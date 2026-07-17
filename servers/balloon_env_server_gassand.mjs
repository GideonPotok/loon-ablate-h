#!/usr/bin/env node
/**
 * balloon_env_server_gassand.mjs — NDJSON IPC bridge for the HELIUM/SAND
 * zero-pressure gas-balloon variant of BalloonEnv.
 *
 * Same protocol and wind/sensing stack as balloon_env_server.mjs (v1), but the
 * altitude actuator is different: instead of a reversible air-ballast pump, the
 * balloon carries FINITE, IRREVERSIBLE reserves of lift gas and ballast:
 *
 *     drop SAND   → shed mass → RISE
 *     vent HELIUM → lose lift → SINK
 *
 * A decision releases a metered amount of one resource (or nothing). The amount
 * sets the size of the buoyancy imbalance, and drag turns that into a vertical
 * speed — a tiny release drifts slowly, a large one climbs/dives fast. See
 * js/balloon_gassand.js for the two-regime lift model.
 *
 * Action space (11 discrete), ordered by altitude rate (low index = up):
 *     0..4  drop sand   (largest → smallest = fast up  → slow up)
 *     5     FLOAT       (release nothing)
 *     6..10 vent helium (smallest → largest = slow down → fast down)
 *   Releasing a resource you have run out of is a no-op (the reserve is empty).
 *
 * State vector (21-dim float32):
 *   [0]     dist / STATION_RADIUS_M
 *   [1]     sin(bearing),  [2] cos(bearing)
 *   [3]     (alt_m - altBandLow) / altBandRange   clamped [0,1]
 *   [4]     vv_m_s / MAX_VV
 *   [5]     helium_kg / HELIUM_CAPACITY_KG   (lift-gas fuel gauge)
 *   [6]     sand_kg   / SAND_CAPACITY_KG     (ballast fuel gauge)
 *   [7]     wind_u_cur / 20,  [8] wind_v_cur / 20
 *   [9..20] 4 × (u/20, v/20, uncertainty/10)
 *           at alts 16625, 17125, 17625, 18125 m
 *
 * REWARD defaults to the v1 station-keeping shape (distance only), with the
 * resource economy exposed in `info` (helium_kg, sand_kg, helium_vented_kg,
 * sand_dropped_kg). Passing use_resource_reward at reset switches on the
 * resource-aware terms (per-release cost, depletion/floor penalties, terminal
 * reserve bonus — see handleReset); with the flag off the reward path is
 * byte-identical to the physics-only baseline. Physics is untouched either way.
 */

import readline from 'readline';
import { runtime } from '../js/config.js';
import { recalculateDerived } from '../js/atmosphere.js';
import { haversine, bearingFlat } from '../js/geo.js';
import { getWind, getBaseWind, WIND_PRESETS } from '../js/wind.js';
import {
    createState, applyRelease, physicsStep, recalculateGassandDerived, findCeiling,
} from '../js/balloon_gassand.js';
import { WindObservationStore } from '../js/wind_observer.js';
import { WindEKF } from '../js/wind_ekf.js';
import { ForecastDegrader } from '../js/wind_degrader.js';

// Initialise gas-balloon derived constants (envelope radius, float band).
recalculateGassandDerived();
// Populate the MAIN-platform derived values too (altBand*_m etc.) — harmless,
// and keeps the shared wind/sensing modules happy. Does not affect gassand
// physics, which uses runtime.gassand exclusively.
recalculateDerived();

// ── Constants ────────────────────────────────────────────────────────────────

const STATE_DIM        = 21;
const NAV_INTERVAL_S   = 300;   // 5-minute decision interval
const PHYSICS_DT_S      = 60;   // 1-minute physics step (physicsStep sub-steps internally)
const PHYSICS_PER_NAV  = Math.round(NAV_INTERVAL_S / PHYSICS_DT_S);  // 5

const WIND_SAMPLE_ALTS = [16625, 17125, 17625, 18125];  // compact state alts
const MAX_UNCERTAINTY  = 10.0;

const TARGET_LAT       = 0;
const TARGET_LON       = 170;
const SPAWN_OFFSET_KM  = 30;

// ── Discrete release action table (built once from the platform config) ──────
// Each entry: { ventHe_kg, dropSand_kg, label }. Ordered fast-up → float → fast-down.
const ACTIONS = (() => {
    const gs = runtime.gassand;
    const out = [];
    for (let i = gs.SAND_RELEASE_KG.length - 1; i >= 0; i--)          // largest sand first
        out.push({ ventHe: 0, dropSand: gs.SAND_RELEASE_KG[i], label: `+sand ${gs.SAND_RELEASE_KG[i]}kg` });
    out.push({ ventHe: 0, dropSand: 0, label: 'float' });
    for (let j = 0; j < gs.HELIUM_RELEASE_KG.length; j++)             // smallest helium first
        out.push({ ventHe: gs.HELIUM_RELEASE_KG[j], dropSand: 0, label: `-He ${gs.HELIUM_RELEASE_KG[j]}kg` });
    return out;
})();
const ACTION_DIM  = ACTIONS.length;                 // 11
const FLOAT_INDEX = runtime.gassand.SAND_RELEASE_KG.length;   // 5

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
        s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s = s >>> 0;
        return (s & 0x7FFFFFFF) / 0x80000000;
    };
}

/**
 * Extract the 21-dim state vector (see file header for layout).
 */
function extractState(bState, getWindFn, time_s, targetLat, targetLon, getUncertaintyFn) {
    const gs   = runtime.gassand;
    const dist = haversine(bState.lat, bState.lon, targetLat, targetLon);
    const brng = bearingFlat(bState.lat, bState.lon, targetLat, targetLon);
    const rad  = brng * Math.PI / 180;

    const s = new Float64Array(STATE_DIM);
    s[0] = dist / gs.STATION_RADIUS_M;
    s[1] = Math.sin(rad);
    s[2] = Math.cos(rad);
    s[3] = Math.max(0, Math.min(1,
        (bState.alt_m - gs.altBandLow_m) / (gs.altBandHigh_m - gs.altBandLow_m)));
    s[4] = (bState.vv_m_s || 0) / gs.MAX_VV_M_S;
    s[5] = (bState.helium_kg || 0) / gs.HELIUM_CAPACITY_KG;
    s[6] = (bState.sand_kg   || 0) / gs.SAND_CAPACITY_KG;

    const wCur = getWindFn(bState.alt_m, time_s);
    s[7] = wCur.u / 20;
    s[8] = wCur.v / 20;

    for (let i = 0; i < 4; i++) {
        const w     = getWindFn(WIND_SAMPLE_ALTS[i], time_s);
        const sigma = getUncertaintyFn ? getUncertaintyFn(WIND_SAMPLE_ALTS[i]) : 0.0;
        const base  = 9 + i * 3;
        s[base]     = w.u / 20;
        s[base + 1] = w.v / 20;
        s[base + 2] = Math.min(1.0, sigma / MAX_UNCERTAINTY);
    }

    return Array.from(s);
}

/**
 * Station-keeping reward (v1 smooth shape). Resource cost is NOT included here;
 * it is applied in stepAction when use_resource_reward is set (and always
 * exposed in `info` regardless).
 */
function computeReward(dist_m) {
    const R       = runtime.gassand.STATION_RADIUS_M;
    const tau     = R * 2.0;
    const inside  = dist_m <= R ? 0.5 : 0.0;
    const soft    = 0.5 * Math.exp(-dist_m / tau);
    const urgency = dist_m > R ? -0.05 : 0.0;
    return inside + soft + urgency;
}

// ── Episode state (reset on each 'reset' command) ────────────────────────────

let ep = null;

// ── Command handlers ─────────────────────────────────────────────────────────

function handleReset(req) {
    const { preset, duration_s, seed } = req;
    const gs = runtime.gassand;
    const spawnOffsetKm = (req.spawn_offset_km != null) ? +req.spawn_offset_km : SPAWN_OFFSET_KM;

    const layers = WIND_PRESETS[preset]?.layers;
    if (!layers) return { ok: false, error: `Unknown preset: ${preset}` };

    // ── Realism flags (per-episode stochastic wind + sensing randomization) ──
    // Ported verbatim from the v2 server (balloon_env_server_v2.mjs) so an
    // R_Gassand run trains/evals under the SAME realism bundle as R. All four
    // are flag-gated and default off, so with no flags this env behaves exactly
    // as the deterministic baseline (the plain gassand demonstrator is
    // unaffected). See ablate_r_gassand_train.py.
    const windPhaseJitter  = !!req.wind_phase_jitter;    // φ_igw, φ_pw ~ U[0,2π)
    const windEpisodeNoise = !!req.wind_episode_noise;   // episode seed into noise hash
    const windParamJitter  = !!req.wind_param_jitter;    // IGW/PW amplitude × logU[0.7,1.4]
    const domainRand       = !!req.domain_rand;          // degrader σ-scale + forecast lag

    // Per-episode wind mods on a DEDICATED RNG stream (seed+424243) — separate
    // from the spawn RNG (makeRng(seed) below), whose draw order must stay
    // untouched. Draw order matches v2 exactly so a given seed yields the same
    // φ/amp/noise realization in both servers.
    let windMods = null;
    if (windPhaseJitter || windEpisodeNoise || windParamJitter) {
        const windRng  = makeRng((seed + 424243) >>> 0);
        const d1 = windRng(), d2 = windRng(), d3 = windRng(), d4 = windRng(), d5 = windRng();
        const logAmp = (x) => Math.exp(Math.log(0.7) + x * (Math.log(1.4) - Math.log(0.7)));
        windMods = {
            igwPhaseOffset: windPhaseJitter  ? d1 * 2 * Math.PI : 0,
            pwPhaseOffset:  windPhaseJitter  ? d2 * 2 * Math.PI : 0,
            igwAmpScale:    windParamJitter  ? logAmp(d3) : 1,
            pwAmpScale:     windParamJitter  ? logAmp(d4) : 1,
            noiseSeed:      windEpisodeNoise ? (d5 * 0x7FFFFFFF) | 0 : 0,
        };
    }

    // ── Resource-aware reward (flag-gated, default OFF) ──
    // With use_resource_reward the plain station-keeping reward is extended by:
    //   • per-release cost: −sand_cost_per_kg·sand − helium_cost_per_kg·He.
    //     Helium is priced above its 6.236× lift equivalence because venting is
    //     doubly irreversible: it lowers the ceiling permanently AND arresting
    //     the resulting descent costs sand.
    //   • depletion_penalty: one-time, the first time either reserve runs dry.
    //   • floor_penalty: per step while pinned at ALT_MIN (absorbing failure).
    //   • terminal_reserve_bonus · mean(He gauge, sand gauge) at episode end.
    // All coefficients are per-reset tunable. Physics is untouched; with the
    // flag off the reward path is byte-identical to the baseline.
    const resourceReward = req.use_resource_reward ? {
        sandCostPerKg:        req.sand_cost_per_kg       != null ? +req.sand_cost_per_kg       : 2.0,
        heliumCostPerKg:      req.helium_cost_per_kg     != null ? +req.helium_cost_per_kg     : 25.0,
        terminalReserveBonus: req.terminal_reserve_bonus != null ? +req.terminal_reserve_bonus : 25.0,
        depletionPenalty:     req.depletion_penalty      != null ? +req.depletion_penalty      : 25.0,
        floorPenalty:         req.floor_penalty          != null ? +req.floor_penalty          : 0.1,
    } : null;

    // Wind functions — every consumer (physics, degrader, observer, EKF) goes
    // through truthWindFn, so the modded wind stays self-consistent everywhere.
    const truthWindFn  = (alt_m, t) => getWind(layers, alt_m, t, windMods);
    const baseWindFn   = (alt_m)    => getBaseWind(layers, alt_m);

    // Sensing stack: ForecastDegrader → WindObserver → WindEKF (v1 stack).
    // Domain randomization (v2 port): per-episode σ-scale on the calibrated
    // bias/noise sigmas + a random forecast lag, from a third dedicated RNG
    // stream (seed+848487).
    const degraderSeed = 7777 + ((seed >>> 0) % 100000);
    const degraderOpts = { SEED: degraderSeed };
    if (domainRand) {
        const drRng      = makeRng((seed + 848487) >>> 0);
        const sigmaScale = Math.exp(Math.log(0.5) + drRng() * (Math.log(2.0) - Math.log(0.5)));
        degraderOpts.BIAS_SIGMA     = 0.71 * sigmaScale;   // calibrated defaults from
        degraderOpts.NOISE_SIGMA    = 2.93 * sigmaScale;   // wind_degrader.DEGRADER_DEFAULTS
        degraderOpts.STALENESS_MODE = 'lagged';
        degraderOpts.LAG_S          = drRng() * 21600;     // U[0, 6h]
    }
    const degrader     = new ForecastDegrader(truthWindFn, baseWindFn, degraderOpts);
    const forecastFn   = (alt_m, t) => degrader.getForecastWind(alt_m, t);
    const observer     = new WindObservationStore();
    const ekf          = new WindEKF();
    ekf.initialize((alt_m) => forecastFn(alt_m, 0));

    function bestWindFn(alt_m, t) {
        if (!ekf.initialized) return forecastFn(alt_m, t);
        const ekfW = ekf.getWind(alt_m);
        const ekfS = ekf.getUncertainty(alt_m);
        if (ekfS < 5.0) return ekfW;
        const rawW = forecastFn(alt_m, t);
        const rawS = degrader.getUncertainty(alt_m);
        const wE = 1 / (ekfS * ekfS), wR = 1 / (rawS * rawS), wT = wE + wR;
        return { u: (ekfW.u * wE + rawW.u * wR) / wT,
                 v: (ekfW.v * wE + rawW.v * wR) / wT };
    }
    function uncertaintyFn(alt_m) {
        return ekf.initialized ? ekf.getUncertainty(alt_m) : MAX_UNCERTAINTY;
    }
    function stepSensing(state, prev, dt_s, t) {
        ekf.predict(dt_s);
        if (prev) {
            const obs = observer.observe(state, prev, dt_s, t);
            if (obs) ekf.update(obs.alt_m, obs.u_obs, obs.v_obs);
        }
    }

    // Spawn: horizontal offset as in v1; altitude near the gas ceiling (the
    // natural launch float with a full envelope + all ballast aboard).
    const rng    = makeRng(seed);
    const angle  = rng() * 2 * Math.PI;
    const cosLat = Math.cos(TARGET_LAT * Math.PI / 180) || 1;
    const spawnLat = TARGET_LAT + (spawnOffsetKm / 111.32) * Math.cos(angle);
    const spawnLon = TARGET_LON + (spawnOffsetKm / (111.32 * cosLat)) * Math.sin(angle);
    const ceiling  = findCeiling(gs.HELIUM_INIT_KG);
    const spawnAlt = ceiling - 300 + rng() * 400;   // ceiling-300 .. ceiling+100

    const balloon = createState(spawnLat, spawnLon, spawnAlt);

    ep = {
        balloon,
        prevBalloon:      null,
        time_s:           0,
        physicsStepCount: 0,
        totalPhysics:     Math.ceil(duration_s / PHYSICS_DT_S),
        inRadiusSteps:    0,
        totalNavSteps:    0,
        targetLat:        TARGET_LAT,
        targetLon:        TARGET_LON,
        sensing:          { bestWindFn, uncertaintyFn, stepSensing, truthWindFn },
        resourceReward,
        heDepleted:       false,
        sandDepleted:     false,
    };

    const dist    = haversine(balloon.lat, balloon.lon, TARGET_LAT, TARGET_LON);
    const statVec = extractState(balloon, bestWindFn, 0, TARGET_LAT, TARGET_LON, uncertaintyFn);

    return {
        ok: true,
        state: statVec,
        n_actions: ACTION_DIM,
        info: {
            dist_m: dist, alt_m: balloon.alt_m, lat: balloon.lat, lon: balloon.lon, time_s: 0,
            helium_kg: balloon.helium_kg, sand_kg: balloon.sand_kg,
            helium_vented_kg: 0, sand_dropped_kg: 0,
        },
    };
}

/** Apply an action index and run one NAV_INTERVAL of physics. */
function stepAction(actionIdx) {
    const act = ACTIONS[actionIdx];
    if (!act) throw new Error(`action ${actionIdx} out of range [0,${ACTION_DIM - 1}]`);

    const { sensing } = ep;
    let { balloon, prevBalloon, time_s, physicsStepCount } = ep;

    // Metered release: apply the decision once at the start of the interval.
    const rel = applyRelease(balloon, act.ventHe, act.dropSand);
    balloon = rel.state;

    // Integrate physics for the nav interval (sensing/observer at the 60 s tick).
    // Track the peak vertical speed over the interval — for a decision that
    // reaches a nearby float within one interval, the end-of-interval velocity
    // has already decayed, so the peak is the honest "how fast did the release
    // make it move" signal (see demo_gassand.py).
    let vvPeak = 0;
    const stepsThisNav = Math.min(PHYSICS_PER_NAV, ep.totalPhysics - physicsStepCount);
    for (let ps = 0; ps < stepsThisNav; ps++) {
        const wind   = sensing.truthWindFn(balloon.alt_m, time_s);
        const before = balloon;
        balloon  = physicsStep(balloon, wind, PHYSICS_DT_S);
        if (Math.abs(balloon.vv_m_s) > Math.abs(vvPeak)) vvPeak = balloon.vv_m_s;
        time_s  += PHYSICS_DT_S;
        sensing.stepSensing(balloon, before, PHYSICS_DT_S, time_s);
        prevBalloon = before;
    }

    ep.balloon           = balloon;
    ep.prevBalloon       = prevBalloon;
    ep.time_s            = time_s;
    ep.physicsStepCount += stepsThisNav;

    const dist   = haversine(balloon.lat, balloon.lon, ep.targetLat, ep.targetLon);
    const done   = ep.physicsStepCount >= ep.totalPhysics;
    let reward   = computeReward(dist);

    // Resource-aware terms (flag-gated; coefficients parsed in handleReset).
    let resourceCost = 0;
    if (ep.resourceReward) {
        const rr = ep.resourceReward;
        const gs = runtime.gassand;
        resourceCost = rr.sandCostPerKg   * rel.sand_released_kg +
                       rr.heliumCostPerKg * rel.helium_released_kg;
        reward -= resourceCost;
        if (!ep.heDepleted && balloon.helium_kg <= 1e-9) {
            ep.heDepleted = true;   reward -= rr.depletionPenalty;
        }
        if (!ep.sandDepleted && balloon.sand_kg <= 1e-9) {
            ep.sandDepleted = true; reward -= rr.depletionPenalty;
        }
        if (balloon.alt_m <= gs.ALT_MIN_M + 1) reward -= rr.floorPenalty;
        if (done) reward += rr.terminalReserveBonus * 0.5 *
            (balloon.helium_kg / gs.HELIUM_CAPACITY_KG +
             balloon.sand_kg   / gs.SAND_CAPACITY_KG);
    }

    if (dist < runtime.gassand.STATION_RADIUS_M) ep.inRadiusSteps++;
    ep.totalNavSteps++;
    const twr50 = ep.totalNavSteps > 0 ? ep.inRadiusSteps / ep.totalNavSteps : 0;

    const stateVec = extractState(
        balloon, sensing.bestWindFn, time_s,
        ep.targetLat, ep.targetLon, sensing.uncertaintyFn,
    );

    return {
        ok: true,
        state:  stateVec,
        reward,
        done,
        info: {
            dist_m: dist, twr50, time_s, alt_m: balloon.alt_m, lat: balloon.lat, lon: balloon.lon,
            vv_m_s: balloon.vv_m_s, vv_peak_m_s: vvPeak,
            helium_kg: balloon.helium_kg, sand_kg: balloon.sand_kg,
            helium_vented_kg: balloon.helium_vented_kg, sand_dropped_kg: balloon.sand_dropped_kg,
            helium_released_kg: rel.helium_released_kg, sand_released_kg: rel.sand_released_kg,
            ...(ep.resourceReward ? { resource_cost: resourceCost } : {}),
        },
    };
}

function handleStep(req) {
    if (!ep) return { ok: false, error: 'no active episode — call reset first' };
    return stepAction(req.action | 0);
}

/**
 * Pick the altitude (within the gas-balloon float band) whose wind carries the
 * balloon most directly toward the station. Self-contained — does NOT use the
 * v1 navigator, which is coupled to the ballast-pump physics and cannot read a
 * gas-balloon state.
 */
function pickBestApproachAlt(getWindFn, t, brngRad) {
    const gs = runtime.gassand;
    const cosB = Math.cos(brngRad), sinB = Math.sin(brngRad);
    let bestAlt = gs.altBandLow_m, bestRate = -Infinity;
    const N = 10;
    for (let i = 0; i < N; i++) {
        const alt = gs.altBandLow_m + (i / (N - 1)) * (gs.altBandHigh_m - gs.altBandLow_m);
        const w   = getWindFn(alt, t);
        const rate = w.u * sinB + w.v * cosB;   // projection onto bearing-to-station
        if (rate > bestRate) { bestRate = rate; bestAlt = alt; }
    }
    return bestAlt;
}

/**
 * Naive gas/sand demonstrator: aim for the best wind-approach altitude and
 * chase it by releasing the resource that moves toward it, magnitude by the
 * altitude error. NOT resource-optimal — provided for interface parity /
 * smoke tests / behavioural-cloning seeds.
 */
function handleHeuristicStep() {
    if (!ep) return { ok: false, error: 'no active episode — call reset first' };
    const gs = runtime.gassand;
    const { sensing } = ep;

    const brngRad   = bearingFlat(ep.balloon.lat, ep.balloon.lon, ep.targetLat, ep.targetLon);
    const targetAlt = pickBestApproachAlt(sensing.bestWindFn, ep.time_s, brngRad);
    const err = targetAlt - ep.balloon.alt_m;   // + → need to rise

    // Map |err| to a release-magnitude level (bigger error → bigger release).
    const nLvl = gs.SAND_RELEASE_KG.length;
    const level = Math.min(nLvl - 1, Math.floor(Math.abs(err) / 150));  // ~150 m per level
    let actionIdx;
    if (err > 75)       actionIdx = (nLvl - 1) - level;         // drop sand (rise); larger err → lower idx
    else if (err < -75) actionIdx = FLOAT_INDEX + 1 + level;    // vent helium (sink)
    else                actionIdx = FLOAT_INDEX;                // float

    const stepResp = stepAction(actionIdx);
    if (!stepResp.ok) return stepResp;
    return { ...stepResp, action: actionIdx };
}

// ── Main: read NDJSON lines from stdin, write NDJSON lines to stdout ─────────

const rl = readline.createInterface({ input: process.stdin, terminal: false });

rl.on('line', (line) => {
    line = line.trim();
    if (!line) return;

    let req;
    try { req = JSON.parse(line); }
    catch (e) {
        process.stdout.write(JSON.stringify({ ok: false, error: `JSON parse error: ${e.message}` }) + '\n');
        return;
    }

    let resp;
    try {
        if      (req.cmd === 'reset')          resp = handleReset(req);
        else if (req.cmd === 'step')           resp = handleStep(req);
        else if (req.cmd === 'heuristic_step') resp = handleHeuristicStep();
        else if (req.cmd === 'close') { process.exit(0); return; }
        else resp = { ok: false, error: `Unknown command: ${req.cmd}` };
    } catch (e) {
        resp = { ok: false, error: e.message };
    }

    process.stdout.write(JSON.stringify(resp) + '\n');
});

rl.on('close', () => process.exit(0));

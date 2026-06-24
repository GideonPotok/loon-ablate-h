#!/usr/bin/env node
/**
 * balloon_env_server.mjs — NDJSON IPC bridge for Python BalloonEnv.
 *
 * Each Python BalloonEnv instance spawns one copy of this process.
 * The process stays alive across multiple reset/step cycles (one episode per
 * reset) so there is no subprocess-spawn overhead per episode.
 *
 * Protocol (newline-delimited JSON on stdin / stdout):
 *
 *   reset:
 *     → {"cmd":"reset","preset":"tropical","duration_s":21600,"seed":42}
 *     ← {"ok":true,"state":[...20 floats...],"info":{"dist_m":…,"alt_m":…,…}}
 *
 *   step:
 *     → {"cmd":"step","action":5}
 *     ← {"ok":true,"state":[...],"reward":0.45,"done":false,
 *         "info":{"dist_m":…,"twr50":…,"time_s":…,"alt_m":…}}
 *
 *   heuristic_step:
 *     → {"cmd":"heuristic_step"}
 *     ← {"ok":true,"action":5,"state":[...],"reward":0.45,"done":false,"info":{…}}
 *     Uses the navigator heuristic to pick the action, then steps the env.
 *     Returns the chosen action index (0–16) alongside the normal step output.
 *     Used for behavioral cloning demo collection.
 *
 *   close:
 *     → {"cmd":"close"}
 *     ← (process exits 0)
 *
 * On any error the response is {"ok":false,"error":"<message>"}.
 * All stderr output is free-form diagnostic text (not JSON).
 *
 * State vector (20-dim float32) matches QRAgent / DQNAgent extractState
 * compact mode exactly:
 *   [0]     dist / STATION_RADIUS_M
 *   [1]     sin(bearing)
 *   [2]     cos(bearing)
 *   [3]     (alt_m - altBandLow) / altBandRange   clamped [0,1]
 *   [4]     vv_m_s / 2.5
 *   [5]     ballast_kg / BALLOON_BALLAST_CAPACITY_KG
 *   [6-7]   u,v wind at current alt / 20
 *   [8-19]  4 × (u/20, v/20, sigma/MAX_UNCERTAINTY)
 *           at alts 16625, 17125, 17625, 18125 m
 */

import readline from 'readline';
import { runtime } from '../js/config.js';
import { recalculateDerived } from '../js/atmosphere.js';
import { haversine, bearingFlat } from '../js/geo.js';
import { getWind, getBaseWind, WIND_PRESETS } from '../js/wind.js';
import { createState, physicsStep } from '../js/balloon.js';
import { WindObservationStore } from '../js/wind_observer.js';
import { WindEKF } from '../js/wind_ekf.js';
import { ForecastDegrader } from '../js/wind_degrader.js';
import { chooseAction } from '../js/navigator.js';
import { indexFromTargetAlt } from '../js/rl_agent.js';

// Initialise derived platform constants (altBandLow_m, altBandHigh_m, etc.)
recalculateDerived();

// ── Constants ────────────────────────────────────────────────────────────────

const NAV_INTERVAL_S   = 300;   // 5-minute decision interval
const PHYSICS_DT_S     = 60;    // 1-minute physics step
const PHYSICS_PER_NAV  = Math.round(NAV_INTERVAL_S / PHYSICS_DT_S);  // 5

const WIND_SAMPLE_ALTS = [16625, 17125, 17625, 18125];  // compact state alts
const MAX_UNCERTAINTY  = 10.0;

// ── v2 expanded-state constants ──────────────────────────────────────────────
const N_WIND_ALTS_V2          = 10;     // wind probes spanning altBandLow..altBandHigh
const PROJECTION_HORIZONS_S   = [3600, 10800, 21600];  // +1h, +3h, +6h
const PROJECTION_DT_S         = 900;    // 15-min chunks for trajectory projection
const HEUR_BEST_ALT_INTERVAL_S = 1800;  // re-pick best altitude every 30 min during heuristic projection
let _windAltsV2Cache = null;
function getWindAltsV2() {
    if (_windAltsV2Cache === null) {
        const lo = runtime.altBandLow_m;
        const hi = runtime.altBandHigh_m;
        const out = new Array(N_WIND_ALTS_V2);
        for (let i = 0; i < N_WIND_ALTS_V2; i++) {
            out[i] = lo + (i / (N_WIND_ALTS_V2 - 1)) * (hi - lo);
        }
        _windAltsV2Cache = out;
    }
    return _windAltsV2Cache;
}

const TARGET_LAT       = 0;
const TARGET_LON       = 170;
const SPAWN_OFFSET_KM  = 30;
const SPAWN_ALT_MIN_M  = 16800;
const SPAWN_ALT_MAX_M  = 18200;

// ── Helpers ──────────────────────────────────────────────────────────────────

function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
        s ^= s << 13;
        s ^= s >>> 17;
        s ^= s << 5;
        s = s >>> 0;
        return (s & 0x7FFFFFFF) / 0x80000000;
    };
}

/** Bang-bang altitude chase (mirrors rl_trainer.js::chaseAction). */
function chaseAction(currentAlt_m, targetAlt_m, tol = 50) {
    const d = targetAlt_m - currentAlt_m;
    if (Math.abs(d) < tol) return 0;
    return d > 0 ? 1 : -1;
}

/**
 * Extract the 20-dim compact state vector.
 * Mirrors DQNAgent / QRAgent extractState (compact mode) exactly.
 */
function extractState(bState, getWindFn, time_s, targetLat, targetLon, getUncertaintyFn) {
    const p   = runtime.platform;
    const dist = haversine(bState.lat, bState.lon, targetLat, targetLon);
    const brng = bearingFlat(bState.lat, bState.lon, targetLat, targetLon);
    const rad  = brng * Math.PI / 180;

    const s = new Float64Array(20);
    s[0] = dist / p.STATION_RADIUS_M;
    s[1] = Math.sin(rad);
    s[2] = Math.cos(rad);
    s[3] = Math.max(0, Math.min(1,
        (bState.alt_m - runtime.altBandLow_m) /
        (runtime.altBandHigh_m - runtime.altBandLow_m)));
    s[4] = (bState.vv_m_s  || 0) / 2.5;
    s[5] = (bState.ballast_kg || 0) / p.BALLOON_BALLAST_CAPACITY_KG;

    const wCur = getWindFn(bState.alt_m, time_s);
    s[6] = wCur.u / 20;
    s[7] = wCur.v / 20;

    for (let i = 0; i < 4; i++) {
        const w     = getWindFn(WIND_SAMPLE_ALTS[i], time_s);
        const sigma = getUncertaintyFn ? getUncertaintyFn(WIND_SAMPLE_ALTS[i]) : 0.0;
        const base  = 8 + i * 3;
        s[base]     = w.u / 20;
        s[base + 1] = w.v / 20;
        s[base + 2] = Math.min(1.0, sigma / MAX_UNCERTAINTY);
    }

    return Array.from(s);  // plain JS array for JSON serialisation
}

// ── v2 expanded state: 52-dim with wind layers + dual forecast trajectories ──

/** Pick altitude with highest approach rate toward the station bearing. */
function pickBestApproachAlt(getWindFn, t, brngRad, candidateAlts) {
    let bestAlt = candidateAlts[0];
    let bestRate = -Infinity;
    const cosB = Math.cos(brngRad);
    const sinB = Math.sin(brngRad);
    for (const alt of candidateAlts) {
        const w = getWindFn(alt, t);
        const rate = w.u * sinB + w.v * cosB;  // wind projection onto bearing-to-station
        if (rate > bestRate) {
            bestRate = rate;
            bestAlt = alt;
        }
    }
    return bestAlt;
}

/**
 * Simulate forward under action=0 (FLOAT) and snapshot the balloon state at each
 * horizon. `horizons` is sorted ascending. Returns array of snapshots.
 */
function projectFloatSnapshots(state, time_s, horizons, getWindFn) {
    let s = state;
    let cur_t = time_s;
    const dt = PROJECTION_DT_S;
    const snapshots = new Array(horizons.length).fill(null);
    let nextHorizonIdx = 0;
    const maxT = time_s + horizons[horizons.length - 1];
    while (cur_t < maxT - dt / 2 && nextHorizonIdx < horizons.length) {
        const w = getWindFn(s.alt_m, cur_t);
        s = physicsStep(s, 0, w, dt);
        cur_t += dt;
        while (nextHorizonIdx < horizons.length &&
               cur_t >= time_s + horizons[nextHorizonIdx]) {
            snapshots[nextHorizonIdx] = s;
            nextHorizonIdx++;
        }
    }
    for (let i = 0; i < horizons.length; i++) {
        if (snapshots[i] === null) snapshots[i] = s;
    }
    return snapshots;
}

/**
 * Simulate forward under a cheap heuristic (re-pick best-approach altitude every
 * HEUR_BEST_ALT_INTERVAL_S) and snapshot at each horizon.
 */
function projectHeuristicSnapshots(state, time_s, horizons, getWindFn,
                                   targetLat, targetLon, windAlts) {
    let s = state;
    let cur_t = time_s;
    const dt = PROJECTION_DT_S;
    let lastPickT = -Infinity;
    let targetAlt = s.alt_m;
    const snapshots = new Array(horizons.length).fill(null);
    let nextHorizonIdx = 0;
    const maxT = time_s + horizons[horizons.length - 1];
    while (cur_t < maxT - dt / 2 && nextHorizonIdx < horizons.length) {
        if (cur_t - lastPickT >= HEUR_BEST_ALT_INTERVAL_S) {
            const brngRad = bearingFlat(s.lat, s.lon, targetLat, targetLon);
            targetAlt = pickBestApproachAlt(getWindFn, cur_t, brngRad, windAlts);
            lastPickT = cur_t;
        }
        const action = targetAlt > s.alt_m + 50 ? 1
                     : targetAlt < s.alt_m - 50 ? -1
                     : 0;
        const w = getWindFn(s.alt_m, cur_t);
        s = physicsStep(s, action, w, dt);
        cur_t += dt;
        while (nextHorizonIdx < horizons.length &&
               cur_t >= time_s + horizons[nextHorizonIdx]) {
            snapshots[nextHorizonIdx] = s;
            nextHorizonIdx++;
        }
    }
    for (let i = 0; i < horizons.length; i++) {
        if (snapshots[i] === null) snapshots[i] = s;
    }
    return snapshots;
}

/**
 * 52-dim state vector for v2.
 *
 * Layout:
 *   [0]      dist / R
 *   [1-2]    sin/cos(bearing*π/180)            (matches v1 quirk for encoding consistency)
 *   [3]      (alt - altBandLow) / range         clamped [0,1]
 *   [4]      vv_m_s / 2.5
 *   [5]      ballast_kg / capacity
 *   [6-7]    current wind u,v / 20
 *   [8-37]   10 × (u/20, v/20, σ/MAX) at altitudes spaced across navigable band
 *   [38-40]  FLOAT projection dist / R  at +1h, +3h, +6h
 *   [41-43]  Heuristic projection dist / R at +1h, +3h, +6h
 *   [44-45]  sin/cos(bearing_float_at_+1h * π/180)
 *   [46-47]  sin/cos(bearing_heur_at_+1h  * π/180)
 *   [48]     time_in_episode fraction
 *   [49]     running TWR-50 estimate so far
 *   [50]     (dist - prev_dist) / R              (sign = approaching/receding)
 *   [51]     (alt - best_approach_alt_now) / range   ("am I where heuristic would send me?")
 */
function extractStateV2(bState, getWindFn, time_s, targetLat, targetLon,
                       getUncertaintyFn, physicsStepCount, totalPhysics,
                       twr50, prevDist) {
    const p   = runtime.platform;
    const R   = p.STATION_RADIUS_M;
    const dist = haversine(bState.lat, bState.lon, targetLat, targetLon);
    const brng = bearingFlat(bState.lat, bState.lon, targetLat, targetLon);
    const rad  = brng * Math.PI / 180;   // match v1 quirk for encoding consistency
    const windAlts = getWindAltsV2();

    const s = new Float64Array(52);
    s[0] = dist / R;
    s[1] = Math.sin(rad);
    s[2] = Math.cos(rad);
    s[3] = Math.max(0, Math.min(1,
        (bState.alt_m - runtime.altBandLow_m) /
        (runtime.altBandHigh_m - runtime.altBandLow_m)));
    s[4] = (bState.vv_m_s  || 0) / 2.5;
    s[5] = (bState.ballast_kg || 0) / p.BALLOON_BALLAST_CAPACITY_KG;

    const wCur = getWindFn(bState.alt_m, time_s);
    s[6] = wCur.u / 20;
    s[7] = wCur.v / 20;

    // 10 altitude wind probes
    for (let i = 0; i < N_WIND_ALTS_V2; i++) {
        const alt   = windAlts[i];
        const w     = getWindFn(alt, time_s);
        const sigma = getUncertaintyFn ? getUncertaintyFn(alt) : 0.0;
        const base  = 8 + i * 3;
        s[base]     = w.u / 20;
        s[base + 1] = w.v / 20;
        s[base + 2] = Math.min(1.0, sigma / MAX_UNCERTAINTY);
    }
    // s[8..37] filled

    // Trajectory projections (FLOAT + heuristic) snapshotted at +1h, +3h, +6h
    const floatSnaps = projectFloatSnapshots(bState, time_s, PROJECTION_HORIZONS_S, getWindFn);
    const heurSnaps  = projectHeuristicSnapshots(bState, time_s, PROJECTION_HORIZONS_S,
                                                  getWindFn, targetLat, targetLon, windAlts);
    for (let i = 0; i < 3; i++) {
        s[38 + i] = haversine(floatSnaps[i].lat, floatSnaps[i].lon, targetLat, targetLon) / R;
        s[41 + i] = haversine(heurSnaps[i].lat,  heurSnaps[i].lon,  targetLat, targetLon) / R;
    }

    // Bearing to station at +1h for both projections (sin/cos with v1 quirk)
    const brngF1 = bearingFlat(floatSnaps[0].lat, floatSnaps[0].lon, targetLat, targetLon);
    const brngH1 = bearingFlat(heurSnaps[0].lat,  heurSnaps[0].lon,  targetLat, targetLon);
    s[44] = Math.sin(brngF1 * Math.PI / 180);
    s[45] = Math.cos(brngF1 * Math.PI / 180);
    s[46] = Math.sin(brngH1 * Math.PI / 180);
    s[47] = Math.cos(brngH1 * Math.PI / 180);

    // Episode-progress + recent-trajectory features
    s[48] = totalPhysics > 0 ? physicsStepCount / totalPhysics : 0;
    s[49] = twr50 || 0;
    s[50] = (prevDist != null) ? (dist - prevDist) / R : 0;

    // Current best-approach altitude offset (signal: alignment with heuristic)
    const heurBestNow = pickBestApproachAlt(getWindFn, time_s, brng, windAlts);
    s[51] = (bState.alt_m - heurBestNow) /
            (runtime.altBandHigh_m - runtime.altBandLow_m);

    return Array.from(s);
}

/**
 * Reward computation.
 *
 *  - Default (v1-compatible smooth shape):
 *      r = 0.5·1[d≤R] + 0.5·exp(−d/(2R)) + (-0.05 if d > R else 0)
 *
 *  - When flags.useRewardFix === true (Phase v2 reward fix, step 3):
 *      r = 1[d ≤ R]                   (per-step; matches TWR-50 metric exactly)
 *    Terminal TWR bonus is added at episode end inside handleStep, not here.
 *
 *  - Shaping (step 4, flags.useShaping) is added on top in handleStep.
 */
function computeReward(dist_m, flags) {
    const R = runtime.platform.STATION_RADIUS_M;

    if (flags && flags.useRewardFix) {
        // Per-step indicator. Sums over episode = inRadiusSteps; sum / totalNavSteps = TWR-50.
        return dist_m <= R ? 1.0 : 0.0;
    }

    // v1-compatible smooth shape
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
    const spawnOffsetKm = (req.spawn_offset_km != null) ? +req.spawn_offset_km : SPAWN_OFFSET_KM;

    // ── v2 feature flags (passed per-episode from the Python trainer) ────────
    const useRewardFix    = !!req.use_reward_fix;
    const useShaping      = !!req.use_shaping;
    const useExpandedState = !!req.use_expanded_state;  // wired in step 5
    const shapingBeta     = (req.shaping_beta != null) ? +req.shaping_beta : 0.5;
    const shapingGamma    = (req.shaping_gamma != null) ? +req.shaping_gamma : 0.97;
    const terminalTwrBonus = (req.terminal_twr_bonus != null) ? +req.terminal_twr_bonus : 50.0;

    const layers = WIND_PRESETS[preset]?.layers;
    if (!layers) return { ok: false, error: `Unknown preset: ${preset}` };

    // Wind functions
    const truthWindFn  = (alt_m, t) => getWind(layers, alt_m, t);
    const baseWindFn   = (alt_m)    => getBaseWind(layers, alt_m);

    // Sensing stack: ForecastDegrader → WindObserver → WindEKF
    const degraderSeed = 7777 + ((seed >>> 0) % 100000);
    const degrader     = new ForecastDegrader(truthWindFn, baseWindFn, { SEED: degraderSeed });
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

    // Spawn position (mirrors rl_trainer.js)
    const rng    = makeRng(seed);
    const angle  = rng() * 2 * Math.PI;
    const cosLat = Math.cos(TARGET_LAT * Math.PI / 180) || 1;
    const spawnLat = TARGET_LAT + (spawnOffsetKm / 111.32) * Math.cos(angle);
    const spawnLon = TARGET_LON + (spawnOffsetKm / (111.32 * cosLat)) * Math.sin(angle);
    const spawnAlt = SPAWN_ALT_MIN_M + rng() * (SPAWN_ALT_MAX_M - SPAWN_ALT_MIN_M);

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

        // v2 flags + sub-knobs (frozen for the episode)
        flags: {
            useRewardFix,
            useShaping,
            useExpandedState,
            shapingBeta,
            shapingGamma,
            terminalTwrBonus,
        },
        prevDist: haversine(balloon.lat, balloon.lon, TARGET_LAT, TARGET_LON),
    };

    const dist    = haversine(balloon.lat, balloon.lon, TARGET_LAT, TARGET_LON);
    const statVec = useExpandedState
        ? extractStateV2(balloon, bestWindFn, 0, TARGET_LAT, TARGET_LON,
                         uncertaintyFn, 0, ep.totalPhysics, 0, dist /* prevDist == dist on reset */)
        : extractState(balloon, bestWindFn, 0, TARGET_LAT, TARGET_LON, uncertaintyFn);

    return {
        ok: true,
        state: statVec,
        info: { dist_m: dist, alt_m: balloon.alt_m, lat: balloon.lat, lon: balloon.lon, time_s: 0 },
    };
}

function handleStep(req) {
    if (!ep) return { ok: false, error: 'no active episode — call reset first' };

    const { action } = req;

    // Map action index (0–16) to target altitude
    const targetAlt_m =
        runtime.altBandLow_m +
        (action / 16) * (runtime.altBandHigh_m - runtime.altBandLow_m);

    const { sensing } = ep;
    let { balloon, prevBalloon, time_s, physicsStepCount } = ep;

    // Run one NAV_INTERVAL of physics steps with bang-bang altitude chase
    const stepsThisNav = Math.min(PHYSICS_PER_NAV, ep.totalPhysics - physicsStepCount);
    for (let ps = 0; ps < stepsThisNav; ps++) {
        const stepAcs = chaseAction(balloon.alt_m, targetAlt_m);
        const wind    = sensing.truthWindFn(balloon.alt_m, time_s);
        const before  = balloon;
        balloon  = physicsStep(balloon, stepAcs, wind, PHYSICS_DT_S);
        time_s  += PHYSICS_DT_S;
        sensing.stepSensing(balloon, before, PHYSICS_DT_S, time_s);
        prevBalloon = before;
    }

    ep.balloon       = balloon;
    ep.prevBalloon   = prevBalloon;
    ep.time_s        = time_s;
    ep.physicsStepCount += stepsThisNav;

    const dist   = haversine(balloon.lat, balloon.lon, ep.targetLat, ep.targetLon);
    let   reward = computeReward(dist, ep.flags);
    const done   = ep.physicsStepCount >= ep.totalPhysics;

    if (dist < runtime.platform.STATION_RADIUS_M) ep.inRadiusSteps++;
    ep.totalNavSteps++;
    const twr50 = ep.totalNavSteps > 0 ? ep.inRadiusSteps / ep.totalNavSteps : 0;

    // Terminal TWR bonus (Phase v2 reward fix, step 3).
    // Added exactly once at episode end so the agent's return correlates with the eval metric.
    if (done && ep.flags.useRewardFix) {
        reward += ep.flags.terminalTwrBonus * twr50;
    }

    // Potential-based reward shaping (Ng/Harada/Russell 1999) — Phase v2 step 4.
    // Φ(s) = β · exp(-d/(2R)). Adds F = γ·Φ(s') − Φ(s), which is policy-invariant.
    // For the terminal state we follow Ng et al.: Φ(s_terminal) = 0, so on the last step
    // shaping reduces to F = -Φ(s) (still cheap to evaluate; preserves optimality).
    if (ep.flags.useShaping) {
        const R       = runtime.platform.STATION_RADIUS_M;
        const tau     = 2.0 * R;
        const beta    = ep.flags.shapingBeta;
        const phiPrev = beta * Math.exp(-ep.prevDist / tau);
        const phiNext = done ? 0.0 : beta * Math.exp(-dist / tau);
        const shaping = ep.flags.shapingGamma * phiNext - phiPrev;
        reward += shaping;
    }

    // Build the state vector first — v2 expanded state needs ep.prevDist (the
    // previous step's distance), which we have not yet overwritten.
    const stateVec = ep.flags.useExpandedState
        ? extractStateV2(
            balloon, sensing.bestWindFn, time_s,
            ep.targetLat, ep.targetLon, sensing.uncertaintyFn,
            ep.physicsStepCount, ep.totalPhysics, twr50, ep.prevDist,
          )
        : extractState(
            balloon, sensing.bestWindFn, time_s,
            ep.targetLat, ep.targetLon, sensing.uncertaintyFn,
          );

    // Now update prevDist for next-step shaping / diagnostics.
    ep.prevDist = dist;

    return {
        ok: true,
        state:  stateVec,
        reward,
        done,
        info: { dist_m: dist, twr50, time_s, alt_m: balloon.alt_m },
    };
}

function handleHeuristicStep() {
    if (!ep) return { ok: false, error: 'no active episode — call reset first' };

    const { sensing } = ep;

    // Ask the navigator heuristic what altitude to target
    const navResult = chooseAction(
        ep.balloon,
        sensing.bestWindFn,
        ep.time_s,
        ep.targetLat,
        ep.targetLon,
        sensing.uncertaintyFn,
    );

    // Map navigator's chosen altitude to one of the 17 action bins
    const targetAlt_m = navResult.targetAlt != null ? navResult.targetAlt : ep.balloon.alt_m;
    const actionIdx   = indexFromTargetAlt(targetAlt_m, runtime.altBandLow_m, runtime.altBandHigh_m);

    // Step the environment with that action (reuse handleStep logic)
    const stepResp = handleStep({ action: actionIdx });
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

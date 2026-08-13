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
import { resolveWindSource } from '../js/wind_source.js';
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
    const useRewardFix     = !!req.use_reward_fix;
    const useShaping       = !!req.use_shaping;
    const useExpandedState = !!req.use_expanded_state;
    const useTimeFeatures  = !!req.use_time_features;   // append 4 Fourier scalars → +4 dim
    const shapingBeta      = (req.shaping_beta   != null) ? +req.shaping_beta   : 0.5;
    const shapingGamma     = (req.shaping_gamma  != null) ? +req.shaping_gamma  : 0.97;
    const shapingLinear    = !!req.shaping_linear;
    const shapingDMax      = (req.shaping_D_max  != null) ? +req.shaping_D_max  : 500_000;
    const terminalTwrBonus = (req.terminal_twr_bonus != null) ? +req.terminal_twr_bonus : 50.0;

    // ── Realism flags (per-episode stochastic wind + sensing randomization) ──
    const windPhaseJitter  = !!req.wind_phase_jitter;    // φ_igw, φ_pw ~ U[0,2π)
    const windEpisodeNoise = !!req.wind_episode_noise;   // episode seed into noise hash
    const windParamJitter  = !!req.wind_param_jitter;    // IGW/PW amplitude × logU[0.7,1.4]
    const domainRand       = !!req.domain_rand;          // degrader σ-scale + forecast lag
    const useEstimatedPhaseFeatures = !!req.use_estimated_phase_features;
    if (useEstimatedPhaseFeatures && useTimeFeatures) {
        return { ok: false, error: 'use_estimated_phase_features and use_time_features are mutually exclusive' };
    }

    // ── Navigation mode (Ablation U): spawn near station A, target B at
    //    navigation_distance_km in a per-episode random direction. ────────────
    const useNavigation        = !!req.use_navigation;
    const navigationDistanceKm = (req.navigation_distance_km != null) ? +req.navigation_distance_km : 100;
    const arrivalBonus         = (req.arrival_bonus != null) ? +req.arrival_bonus : 0;

    // ── Wind source: synthetic preset (default) or real ERA5 reanalysis ─────
    // 'preset' keeps every existing ablation bit-identical. 'era5' moves the
    // station to wherever the archive sample landed, so TARGET_LAT/TARGET_LON
    // below become defaults rather than constants.
    const windSourceName = req.wind_source || 'preset';
    const era5Dir        = req.era5_dir || process.env.LOON_ERA5_DIR || null;
    const era5MinShear   = (req.era5_min_shear_ms != null) ? +req.era5_min_shear_ms : 0;

    if (windSourceName === 'preset' && !WIND_PRESETS[preset]?.layers) {
        return { ok: false, error: `Unknown preset: ${preset}` };
    }

    // Per-episode wind mods. Dedicated RNG stream (not the spawn RNG, whose
    // draw order must stay untouched for old ablations). All five values are
    // drawn unconditionally so a given seed yields the same φ/scale/noise
    // realization regardless of which sub-flags are enabled.
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

    // Wind functions — every consumer (physics, degrader, observer, EKF) goes
    // through truthWindFn, so the modded wind stays self-consistent everywhere.
    // ERA5 episode selection gets its own RNG stream (like windMods above), so
    // enabling it cannot shift the spawn draws that older ablations depend on.
    let wind;
    try {
        wind = resolveWindSource({
            source:   windSourceName,
            preset,
            windMods,
            era5Dir,
            rng:      makeRng((seed + 313131) >>> 0),
            duration_s,
            defaultTargetLat: TARGET_LAT,
            defaultTargetLon: TARGET_LON,
            era5Opts: { minShear_ms: era5MinShear },
        });
    } catch (e) {
        return { ok: false, error: `wind source: ${e.message}` };
    }
    const { truthWindFn, baseWindFn } = wind;
    const targetLat = wind.targetLat;
    const targetLon = wind.targetLon;

    // Sensing stack: ForecastDegrader → WindObserver → WindEKF
    // Domain randomization (port of js/rl_trainer.js createSensingStack drOpts):
    // per-episode σ-scale on the calibrated bias/noise sigmas + a random
    // forecast lag, sampled from a third dedicated RNG stream.
    const degraderSeed = 7777 + ((seed >>> 0) % 100000);
    const degraderOpts = { SEED: degraderSeed };
    if (domainRand) {
        const drRng      = makeRng((seed + 848487) >>> 0);
        const sigmaScale = Math.exp(Math.log(0.5) + drRng() * (Math.log(2.0) - Math.log(0.5)));
        degraderOpts.BIAS_SIGMA     = 0.71 * sigmaScale;   // calibrated values from
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

    // IGW phase estimator — online complex demodulation of GPS-wind residuals
    // at the nominal IGW frequency. Uses only real-life-available signals:
    // the balloon's own GPS-derived winds and a climatological base wind.
    // Generative model: u = A·cosθ, v = 0.7A·sinθ, θ = ωt − k·alt + φ₀, so
    //   z = r_u + i·r_v/0.7 ≈ A·e^{iθ};  y = z·e^{−i(ωt−k·alt)} ≈ A·e^{iφ₀}.
    // An EMA of y over ~one IGW period gives φ̂₀ = arg(ȳ), Â = |ȳ| (other
    // components — PW, diurnal, noise — rotate after demodulation and average
    // out, standard lock-in behavior).
    const est = { re: 0, im: 0, conf: 0, lastT: null };
    function estClockPhase(alt_m, t) {
        return (2 * Math.PI * t) / runtime.wind.IGW_PERIOD_S -
               (2 * Math.PI * alt_m) / runtime.wind.IGW_VERT_WAVELENGTH_M;
    }
    // The drift-derived obs wind equals the wind the physics sampled at the
    // step-START altitude and time, so attribute it exactly there: the base
    // subtraction becomes exact within a layer (no boundary-crossing pulses,
    // which on strong shear are several m/s — bigger than the IGW itself) and
    // the demodulation basis matches where the wave was actually sampled.
    function estimatorUpdate(obs, altSampled, tSampled) {
        const base = baseWindFn(altSampled);
        const ru = obs.u_obs - base.u;
        const rv = (obs.v_obs - base.v) / 0.7;
        const ph = estClockPhase(altSampled, tSampled);
        const c = Math.cos(ph), s = Math.sin(ph);
        const yre = ru * c + rv * s;      // y = z·e^{−i·ph}
        const yim = rv * c - ru * s;
        const dt = est.lastT == null ? PHYSICS_DT_S : Math.max(1e-9, tSampled - est.lastT);
        // 2× the IGW period: narrower passband so PW/diurnal leakage from the
        // balloon's irregular altitude wander averages out (1× left the phasor
        // beating ±40° on real trajectories).
        const a = 1 - Math.exp(-dt / (2 * runtime.wind.IGW_PERIOD_S));
        est.re   += a * (yre - est.re);
        est.im   += a * (yim - est.im);
        est.conf += a * (1 - est.conf);
        est.lastT = tSampled;
    }
    function phaseFeatures(alt_m, t) {
        const amp  = Math.hypot(est.re, est.im);
        const phi0 = amp > 1e-12 ? Math.atan2(est.im, est.re) : 0;
        const th   = estClockPhase(alt_m, t) + phi0;
        return [Math.sin(th), Math.cos(th),
                Math.min(1, amp / runtime.wind.IGW_AMPLITUDE), est.conf];
    }
    function phaseDebug() {
        const amp  = Math.hypot(est.re, est.im);
        return {
            true_igw_offset: windMods ? windMods.igwPhaseOffset : 0,
            est_igw_offset:  amp > 1e-12 ? Math.atan2(est.im, est.re) : 0,
            est_amp:         amp,
            conf:            est.conf,
        };
    }

    function stepSensing(state, prev, dt_s, t) {
        ekf.predict(dt_s);
        if (prev) {
            const obs = observer.observe(state, prev, dt_s, t);
            if (obs) {
                ekf.update(obs.alt_m, obs.u_obs, obs.v_obs);
                if (useEstimatedPhaseFeatures) {
                    estimatorUpdate(obs, prev.alt_m, t - dt_s);
                }
            }
        }
    }

    // Spawn position (mirrors rl_trainer.js)
    const rng    = makeRng(seed);
    const angle  = rng() * 2 * Math.PI;
    const cosLat = Math.cos(targetLat * Math.PI / 180) || 1;

    let finalTargetLat = targetLat;
    let finalTargetLon = targetLon;

    if (useNavigation) {
        // Navigation mode: target B is navigation_distance_km from the station
        // in a per-episode random direction. Dedicated RNG stream so enabling
        // navigation doesn't shift existing spawn/wind draws.
        const navRng    = makeRng((seed + 191919) >>> 0);
        const navAngle  = navRng() * 2 * Math.PI;
        finalTargetLat = targetLat + (navigationDistanceKm / 111.32) * Math.cos(navAngle);
        finalTargetLon = targetLon + (navigationDistanceKm / (111.32 * cosLat)) * Math.sin(navAngle);
    }

    // In navigation mode: spawn near the station (point A) with spawn_offset_km
    // jitter, NOT near the target (point B). In station-keeping mode: spawn near
    // the target as before (targetLat/targetLon == station).
    const spawnOriginLat = useNavigation ? targetLat : finalTargetLat;
    const spawnOriginLon = useNavigation ? targetLon : finalTargetLon;
    const spawnLat = spawnOriginLat + (spawnOffsetKm / 111.32) * Math.cos(angle);
    const spawnLon = spawnOriginLon + (spawnOffsetKm / (111.32 * cosLat)) * Math.sin(angle);
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
        targetLat:        finalTargetLat,
        targetLon:        finalTargetLon,
        arrivalStep:      null,        // first nav step inside target radius (navigation mode)
        sensing:          { bestWindFn, uncertaintyFn, stepSensing, truthWindFn,
                            phaseFeatures, phaseDebug },

        // v2 flags + sub-knobs (frozen for the episode)
        flags: {
            useRewardFix,
            useShaping,
            useExpandedState,
            useTimeFeatures,
            useEstimatedPhaseFeatures,
            windPhaseJitter,
            windEpisodeNoise,
            windParamJitter,
            domainRand,
            shapingBeta,
            shapingGamma,
            shapingLinear,
            shapingDMax,
            terminalTwrBonus,
            shapingLinear,
            shapingDMax,
            arrivalBonus,
        },
        prevDist: haversine(balloon.lat, balloon.lon, finalTargetLat, finalTargetLon),
    };

    const dist    = haversine(balloon.lat, balloon.lon, finalTargetLat, finalTargetLon);
    let statVec = useExpandedState
        ? extractStateV2(balloon, bestWindFn, 0, finalTargetLat, finalTargetLon,
                         uncertaintyFn, 0, ep.totalPhysics, 0, dist /* prevDist == dist on reset */)
        : extractState(balloon, bestWindFn, 0, finalTargetLat, finalTargetLon, uncertaintyFn);
    if (ep.flags.useTimeFeatures) {
        // time_s=0 on reset: sin(0)=0, cos(0)=1 for both periods — correct initial phase.
        statVec = [...statVec, 0, 1, 0, 1];
    } else if (ep.flags.useEstimatedPhaseFeatures) {
        // No observations yet: φ̂₀=0, amp=0, confidence=0.
        statVec = [...statVec, ...phaseFeatures(balloon.alt_m, 0)];
    }

    return {
        ok: true,
        state: statVec,
        info: {
            dist_m: dist, alt_m: balloon.alt_m, lat: balloon.lat, lon: balloon.lon, time_s: 0,
            // Which wind this episode actually ran on. For era5 this is the grid
            // cell and start time, without which an ERA5 result is unreproducible.
            wind: wind.meta,
            target_lat: finalTargetLat, target_lon: finalTargetLon,
        },
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

    const R_nav = runtime.platform.STATION_RADIUS_M;
    if (dist < R_nav) ep.inRadiusSteps++;
    ep.totalNavSteps++;
    const twr50 = ep.totalNavSteps > 0 ? ep.inRadiusSteps / ep.totalNavSteps : 0;

    // Arrival bonus — one-time reward on first entry into target radius.
    // Non-potential-based; for navigation mode this incentivises reaching B.
    if (ep.arrivalStep === null && dist < R_nav) {
        ep.arrivalStep = ep.totalNavSteps;
        reward += ep.flags.arrivalBonus;
    }

    // Terminal TWR bonus (Phase v2 reward fix, step 3).
    // Added exactly once at episode end so the agent's return correlates with the eval metric.
    if (done && ep.flags.useRewardFix) {
        reward += ep.flags.terminalTwrBonus * twr50;
    }

    // Potential-based reward shaping (Ng/Harada/Russell 1999) — Phase v2 step 4.
    // Linear:      Φ(s) = β · max(0, 1 − d/D_max)   (hard cutoff at D_max)
    // Exponential: Φ(s) = β · exp(−d/τ)              (τ = shaping_D_max; fallback 2R)
    // Adds F = γ·Φ(s') − Φ(s), which is policy-invariant.
    // For the terminal state: Φ(s_terminal) = 0, so shaping reduces to F = −Φ(s).
    if (ep.flags.useShaping) {
        const R    = runtime.platform.STATION_RADIUS_M;
        const beta = ep.flags.shapingBeta;
        let phiPrev, phiNext;
        if (ep.flags.shapingLinear) {
            const D = ep.flags.shapingDMax;
            phiPrev = beta * Math.max(0, 1 - ep.prevDist / D);
            phiNext = done ? 0.0 : beta * Math.max(0, 1 - dist / D);
        } else {
            const tau = ep.flags.shapingDMax || (2.0 * R);
            phiPrev = beta * Math.exp(-ep.prevDist / tau);
            phiNext = done ? 0.0 : beta * Math.exp(-dist / tau);
        }
        const shaping = ep.flags.shapingGamma * phiNext - phiPrev;
        reward += shaping;
    }

    // Build the state vector first — v2 expanded state needs ep.prevDist (the
    // previous step's distance), which we have not yet overwritten.
    let stateVec = ep.flags.useExpandedState
        ? extractStateV2(
            balloon, sensing.bestWindFn, time_s,
            ep.targetLat, ep.targetLon, sensing.uncertaintyFn,
            ep.physicsStepCount, ep.totalPhysics, twr50, ep.prevDist,
          )
        : extractState(
            balloon, sensing.bestWindFn, time_s,
            ep.targetLat, ep.targetLon, sensing.uncertaintyFn,
          );
    if (ep.flags.useTimeFeatures) {
        const IGW_S = 28800;    // 8h IGW period
        const PW_S  = 432000;   // 5-day planetary wave period
        stateVec = [
            ...stateVec,
            Math.sin(2 * Math.PI * time_s / IGW_S),
            Math.cos(2 * Math.PI * time_s / IGW_S),
            Math.sin(2 * Math.PI * time_s / PW_S),
            Math.cos(2 * Math.PI * time_s / PW_S),
        ];
    } else if (ep.flags.useEstimatedPhaseFeatures) {
        stateVec = [...stateVec, ...sensing.phaseFeatures(balloon.alt_m, time_s)];
    }

    // Now update prevDist for next-step shaping / diagnostics.
    ep.prevDist = dist;

    return {
        ok: true,
        state:  stateVec,
        reward,
        done,
        info: {
            dist_m: dist, twr50, time_s, alt_m: balloon.alt_m,
            lat: balloon.lat, lon: balloon.lon,
            ...(ep.arrivalStep != null ? { arrival_step: ep.arrivalStep } : {}),
            ...(ep.flags.useEstimatedPhaseFeatures
                ? { phase_debug: sensing.phaseDebug() } : {}),
        },
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

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

/**
 * Smooth reward shape (Bellemare et al. 2020, QRAgent default).
 * r = 0.5·1[d≤R] + 0.5·exp(−d/(2R))
 */
function computeReward(dist_m) {
    const R       = runtime.platform.STATION_RADIUS_M;
    const tau     = R * 2.0;
    const inside  = dist_m <= R ? 0.5 : 0.0;
    const soft    = 0.5 * Math.exp(-dist_m / tau);
    const urgency = dist_m > R ? -0.05 : 0.0;   // recovery urgency; negative beyond ~5R
    return inside + soft + urgency;
}

// ── Episode state (reset on each 'reset' command) ────────────────────────────

let ep = null;

// ── Command handlers ─────────────────────────────────────────────────────────

function handleReset(req) {
    const { preset, duration_s, seed } = req;
    const spawnOffsetKm = (req.spawn_offset_km != null) ? +req.spawn_offset_km : SPAWN_OFFSET_KM;

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
    };

    const dist    = haversine(balloon.lat, balloon.lon, TARGET_LAT, TARGET_LON);
    const statVec = extractState(balloon, bestWindFn, 0, TARGET_LAT, TARGET_LON, uncertaintyFn);

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
    const reward = computeReward(dist);
    const done   = ep.physicsStepCount >= ep.totalPhysics;

    if (dist < runtime.platform.STATION_RADIUS_M) ep.inRadiusSteps++;
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

/**
 * cem_mpc.js — Cross-Entropy Method Model Predictive Control.
 *
 * Hybrid approach: seeds the CEM population with "constant altitude"
 * sequences (equivalent to heuristic evaluation) and then refines with
 * mixed action sequences. This captures both:
 *   - Single-altitude strategies (what the heuristic does well)
 *   - Multi-step strategies (ascend then descend) that the heuristic misses
 *
 * Based on: Park et al. (2025), "High-Altitude Balloon Station-Keeping
 * with First Order Model Predictive Control" — CEM variant.
 */
import { runtime } from './config.js';
import { haversine, bearingFlat, windApproachRate } from './geo.js';
import { physicsStep } from './balloon.js';

// ── Configuration ────────────────────────────────────────────────────

export const CEM_CONFIG = {
    HORIZON_STEPS:    12,     // 12 × 300s = 1 hour planning horizon
    STEP_DURATION_S:  300,    // Each step = 5 min (nav interval)
    RANDOM_SAMPLES:   48,     // Random sequences per CEM iteration
    ELITE_COUNT:      8,      // K: top sequences kept
    ITERATIONS:       3,      // CEM refinement iterations
    ACTION_SMOOTH:    0.7,    // Smoothing factor for distribution update
};

// ── Seeded PRNG ─────────────────────────────────────────────────────

class SeededRNG {
    constructor(seed = 42) {
        this.state = seed;
    }
    next() {
        this.state |= 0;
        this.state = (this.state + 0x6D2B79F5) | 0;
        let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
}

// ── Core rollout evaluation ─────────────────────────────────────────

/**
 * Evaluate an action sequence by forward simulation.
 * Returns cost (lower = better) and trajectory statistics.
 */
function evaluateSequence(state, actions, getWindFn, time_s, targetLat, targetLon) {
    const physicsDt = runtime.nav.PHYSICS_DT_S;
    const stepDur = CEM_CONFIG.STEP_DURATION_S;
    const stepsPerAction = Math.round(stepDur / physicsDt);

    let s = { ...state };
    let t = time_s;
    let sumDist = 0;
    let minDist = haversine(s.lat, s.lon, targetLat, targetLon);
    let nSteps = 0;

    for (const action of actions) {
        for (let p = 0; p < stepsPerAction; p++) {
            const w = getWindFn(s.alt_m, t);
            s = { ...physicsStep(s, action, w, physicsDt) };
            t += physicsDt;
            const d = haversine(s.lat, s.lon, targetLat, targetLon);
            sumDist += d;
            minDist = Math.min(minDist, d);
            nSteps++;
        }
    }

    const finalDist = haversine(s.lat, s.lon, targetLat, targetLon);
    const avgDist = sumDist / nSteps;

    // Switching penalty: discourage gratuitous action changes
    let switches = 0;
    for (let i = 1; i < actions.length; i++) {
        if (actions[i] !== actions[i - 1]) switches++;
    }

    const cost = finalDist * 0.5 + avgDist * 0.3 + minDist * 0.2
               + switches * 300;

    return { cost, finalDist, minDist, avgDist };
}

/**
 * Build seed sequences: constant-action sequences targeting each reachable altitude.
 * These replicate the heuristic's exhaustive altitude search within CEM.
 */
function buildSeedSequences(state) {
    const H = CEM_CONFIG.HORIZON_STEPS;
    const seeds = [];

    // Float sequence (stay at current altitude)
    seeds.push(new Array(H).fill(0));

    // For each reachable altitude, create a sequence that moves there then floats
    for (const targetAlt of runtime.altitudeLevels) {
        const diff = targetAlt - state.alt_m;
        if (Math.abs(diff) < 50) continue; // Skip current altitude (covered by float)

        const action = diff > 0 ? 1 : -1;
        // Estimate steps to reach target (approx 125m per 5min at full pump)
        const transitionSteps = Math.min(H, Math.ceil(Math.abs(diff) / 125));
        const seq = [];
        for (let i = 0; i < H; i++) {
            seq.push(i < transitionSteps ? action : 0);
        }
        seeds.push(seq);
    }

    // Add a few "oscillation" seeds: up then down, down then up
    const half = Math.floor(H / 2);
    seeds.push([...new Array(half).fill(1), ...new Array(H - half).fill(-1)]);
    seeds.push([...new Array(half).fill(-1), ...new Array(H - half).fill(1)]);

    // Short burst patterns
    const third = Math.floor(H / 3);
    seeds.push([...new Array(third).fill(1), ...new Array(third).fill(0), ...new Array(H - 2 * third).fill(-1)]);
    seeds.push([...new Array(third).fill(-1), ...new Array(third).fill(0), ...new Array(H - 2 * third).fill(1)]);

    return seeds;
}

/**
 * Sample a random action sequence from per-timestep categorical probabilities.
 */
function sampleSequence(probs, rng) {
    const actions = [];
    for (let t = 0; t < probs.length; t++) {
        const r = rng.next();
        const [pAsc, pFloat, pDesc] = probs[t];
        if (r < pAsc) actions.push(1);
        else if (r < pAsc + pFloat) actions.push(0);
        else actions.push(-1);
    }
    return actions;
}

// ── Main CEM-MPC planner ────────────────────────────────────────────

/**
 * CEM-MPC planning step with seeded population.
 *
 * @param {object} state — Current balloon state
 * @param {Function} getWindFn — Wind accessor: (alt_m, time_s) → { u, v }
 * @param {number} time_s — Current simulation time
 * @param {number} targetLat — Target latitude
 * @param {number} targetLon — Target longitude
 * @returns {NavDecision}
 */
export function cemPlan(state, getWindFn, time_s, targetLat, targetLon) {
    const cfg = CEM_CONFIG;
    // Adaptive horizon: shorter when close (avoid over-planning), longer when far
    const dist = haversine(state.lat, state.lon, targetLat, targetLon);
    const distRatio = dist / runtime.platform.STATION_RADIUS_M;
    const H = distRatio < 0.8 ? Math.max(4, Math.floor(cfg.HORIZON_STEPS * 0.5)) :
              distRatio > 2.0 ? Math.min(18, Math.ceil(cfg.HORIZON_STEPS * 1.5)) :
              cfg.HORIZON_STEPS;
    const K = cfg.ELITE_COUNT;
    const smooth = cfg.ACTION_SMOOTH;

    const rng = new SeededRNG(Math.floor(time_s) ^ 0xDEADBEEF);

    // Build seed sequences (constant-altitude + oscillation patterns)
    const seeds = buildSeedSequences(state);

    // Initialize distribution from seeds (biased toward patterns that work)
    let probs = [];
    for (let t = 0; t < H; t++) {
        probs.push([0.33, 0.34, 0.33]);
    }

    let bestSequence = null;
    let bestCost = Infinity;
    let bestResult = null;

    // Evaluate seed sequences first (these are the "heuristic equivalents")
    const seedResults = [];
    for (const seq of seeds) {
        const result = evaluateSequence(state, seq, getWindFn, time_s, targetLat, targetLon);
        seedResults.push({ seq, ...result });
        if (result.cost < bestCost) {
            bestCost = result.cost;
            bestSequence = seq;
            bestResult = result;
        }
    }

    // Initialize CEM distribution from seed elite
    seedResults.sort((a, b) => a.cost - b.cost);
    const seedElite = seedResults.slice(0, K);

    for (let t = 0; t < H; t++) {
        let cAsc = 0, cFloat = 0, cDesc = 0;
        for (const e of seedElite) {
            if (e.seq[t] === 1) cAsc++;
            else if (e.seq[t] === 0) cFloat++;
            else cDesc++;
        }
        const total = seedElite.length;
        // Bias toward seed elite patterns
        probs[t] = [cAsc / total, cFloat / total, cDesc / total];
        // Ensure minimum exploration
        probs[t] = probs[t].map(p => Math.max(0.05, p));
        const sum = probs[t][0] + probs[t][1] + probs[t][2];
        probs[t] = probs[t].map(p => p / sum);
    }

    // CEM iterations: sample around the seed-biased distribution
    for (let iter = 0; iter < cfg.ITERATIONS; iter++) {
        const population = [];
        for (let i = 0; i < cfg.RANDOM_SAMPLES; i++) {
            const seq = sampleSequence(probs, rng);
            const result = evaluateSequence(state, seq, getWindFn, time_s, targetLat, targetLon);
            population.push({ seq, ...result });
        }

        // Include previous best
        population.push({ seq: bestSequence, ...bestResult });

        population.sort((a, b) => a.cost - b.cost);

        if (population[0].cost < bestCost) {
            bestCost = population[0].cost;
            bestSequence = population[0].seq;
            bestResult = population[0];
        }

        const elite = population.slice(0, K);

        const newProbs = [];
        for (let t = 0; t < H; t++) {
            let cAsc = 0, cFloat = 0, cDesc = 0;
            for (const e of elite) {
                if (e.seq[t] === 1) cAsc++;
                else if (e.seq[t] === 0) cFloat++;
                else cDesc++;
            }
            const total = K;
            let pAsc = smooth * (cAsc / total) + (1 - smooth) * probs[t][0];
            let pFloat = smooth * (cFloat / total) + (1 - smooth) * probs[t][1];
            let pDesc = smooth * (cDesc / total) + (1 - smooth) * probs[t][2];
            const sum = pAsc + pFloat + pDesc;
            newProbs.push([pAsc / sum, pFloat / sum, pDesc / sum]);
        }
        probs = newProbs;
    }

    // Extract first action and determine target altitude
    const action = bestSequence[0];

    // Count sustained direction in the best sequence
    let sameCount = 0;
    for (let i = 0; i < Math.min(6, bestSequence.length); i++) {
        if (bestSequence[i] === action) sameCount++;
        else break;
    }

    const altStep = runtime.nav.ALTITUDE_STEP_M;
    let targetAlt = state.alt_m;
    if (action === 1) {
        targetAlt = Math.min(runtime.altBandHigh_m, state.alt_m + altStep * sameCount);
    } else if (action === -1) {
        targetAlt = Math.max(runtime.altBandLow_m, state.alt_m - altStep * sameCount);
    }

    return {
        action,
        targetAlt,
        reason: `CEM-MPC: seq=[${bestSequence.slice(0, 4).join(',')}...] cost=${(bestCost / 1000).toFixed(1)}k proj=${(bestResult.finalDist / 1000).toFixed(0)}km`,
        projectedDist: bestResult.finalDist,
        stage: 'cem_mpc',
        sequence: bestSequence,
    };
}

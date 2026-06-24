/**
 * rl_agent.js — DQN agent for balloon station-keeping (Q2.2).
 *
 * Encapsulates the DQN learning algorithm:
 *   - State extraction: balloon position/wind → normalized feature vector
 *   - Action selection: epsilon-greedy with annealing
 *   - Reward computation: +1 in radius, distance penalty outside
 *   - Training: mini-batch TD learning with target network
 *
 * The agent works with any wind source via getWindFn callback, making it
 * compatible with both synthetic presets and real forecast data.
 *
 * Architecture follows Bellemare et al. 2020 (Nature 588):
 *   - 20-dim state: distance, bearing(sin/cos), altitude, vv, ballast,
 *     GPS wind at current altitude, EKF wind + uncertainty at 4 levels
 *   - Network: 20 → 64 → 32 → 3 (Q-values for DESCEND/FLOAT/ASCEND)
 *   - Experience replay (10k buffer), target network sync every 100 episodes
 *   - Epsilon: 1.0 → 0.05, decay 0.995 per episode
 */

import { runtime } from './config.js';
import { haversine, bearingFlat } from './geo.js';
import { NeuralNetwork, ReplayBuffer } from './dqn.js';

// ── Agent configuration ─────────────────────────────────────────────

export const RL_DEFAULTS = Object.freeze({
    // 'compact' (default): 20 features — distance, bearing(×2), alt, vv, ballast,
    //   GPS-wind(×2), then 4 sample altitudes × (u, v, sigma). The EKF posterior
    //   is sub-sampled to 4 altitudes; sufficient-statistic-flavoured.
    // 'wide':    74 features — same 8 non-EKF features + full 34-dim EKF mean
    //   (u, v at all 17 altitudes) + 34-dim diagonal covariance (variance per
    //   altitude per component). Per FUTURE_PATHS.md §4: 2-day ablation to test
    //   whether the compact compression is hurting us.
    STATE_MODE:          'compact',
    STATE_DIM:           20,
    HIDDEN_SIZES:        [64, 32],
    ACTION_COUNT:        3,       // DESCEND=-1, FLOAT=0, ASCEND=+1
    LEARNING_RATE:       0.0001,  // Reduced from 0.0003 — prevents Q-value explosion
    GAMMA:               0.95,    // Reduced from 0.97 — shorter horizon, less bootstrap error
    EPSILON_START:       1.0,
    EPSILON_END:         0.05,
    EPSILON_DECAY:       0.995,   // Per-episode multiplicative decay
    REPLAY_CAPACITY:     10000,
    BATCH_SIZE:          32,
    TARGET_UPDATE_FREQ:  10,      // Increased from 25 → more frequent sync for stability
    SEED:                42,

    // Altitude levels for wind state features (4 levels across reachable band)
    WIND_SAMPLE_ALTS:    [16625, 17125, 17625, 18125],

    // Maximum uncertainty value for normalization (m/s)
    MAX_UNCERTAINTY:     10.0,

    // Action space — see actionFromIndex / applyTargetAltAction in this file
    // and rl_trainer.js. Configurable so we can A/B without hardcoding.
    // Options:
    //   'discrete3'   — original: 3 actions, mapped to ACS commands directly:
    //                   0=DESCEND(-1), 1=FLOAT(0), 2=ASCEND(+1). The agent
    //                   commits one ACS command for the whole nav interval.
    //                   ACTION_COUNT must be 3.
    //   'targetAlt17' — agent picks one of 17 target altitudes spanning the
    //                   reachable band at 125 m steps (matches navigator's
    //                   evaluation grid). Trainer applies a bang-bang chase
    //                   over the nav interval. ACTION_COUNT auto-set to 17.
    //                   This makes the agent's action interface IDENTICAL to
    //                   the heuristic's, so the heuristic itself is a valid
    //                   policy in this space (sanity-check the gauntlet).
    ACTION_SPACE:        'discrete3',

    // Reward shape — see computeReward(). Configurable so we can A/B without
    // hardcoding. Options:
    //   'cliff'  — original: +1 inside R, −min(1, d/R) outside. Bimodal,
    //              saturates beyond 2R, no gradient inside. Validated to
    //              poison value learning empirically (see WORKING_NOTES.md).
    //   'smooth' — Bellemare-style (default): 0.5·1[d≤R] + 0.5·exp(−d/(2R)).
    //              Bounded in [0,1], smooth everywhere, never saturates,
    //              preserves the discrete in-radius incentive.
    //   'expOnly'— pure exponential proximity, no in-radius bonus. Smoother
    //              but loses the "I made it" signal entirely.
    REWARD_SHAPE:        'smooth',
    // For 'smooth' / 'expOnly': scale length of exponential in units of R.
    // 2.0 means 50 % decay over 2·R = 100 km. Larger = flatter outside R.
    REWARD_SCALE_R:      2.0,

    // Optimizer for the underlying NN. 'sgd' (legacy) or 'adam'. Adam adapts
    // per-parameter learning rates and removes the need for manual gradient
    // clipping at ±1.0; recommended with the smooth reward shape because the
    // gradient distribution is very different from the cliff reward's, and
    // SGD hyperparameters tuned for cliff don't transfer.
    OPTIMIZER:           'adam',
    ADAM_BETA1:          0.9,
    ADAM_BETA2:          0.999,

    // n-step returns. With N_STEP=1 (default) this is vanilla DQN. With
    // N_STEP=k, the target becomes  Σ_{i=0..k-1} γ^i r_{t+i} + γ^k max Q(s_{t+k})
    // which propagates rewards faster across long episodes. Recommended for the
    // 24h+ curriculum tiers where γ=0.97 alone gives ~2 h effective horizon.
    N_STEP:              1,

    // Replay sampling mode. 'uniform' (default) is vanilla DQN. 'prioritized'
    // is PER (Schaul 2015) — samples transitions with probability ∝ |TD|^α,
    // applies importance-sampling correction with exponent β. ~1.5–2× sample
    // efficient at the cost of one extra dictionary write per training step
    // (to update priorities post-hoc).
    REPLAY_MODE:         'uniform',
    PER_ALPHA:           0.6,    // priority exponent
    PER_BETA0:           0.4,    // initial IS exponent (annealed → 1.0)
    PER_BETA_ANNEAL:     1e-4,   // β increment per training step

    // Polyak (soft) target network update rate. When > 0, the target net is
    // blended toward the policy net after every training batch:
    //   target ← (1-tau)*target + tau*policy
    // This replaces the hard copy every TARGET_UPDATE_FREQ episodes (which is
    // used when POLYAK_TAU=0). tau=0.005 matches the DDPG/TD3 convention and
    // produces smoother target estimates than hard syncs.
    POLYAK_TAU:          0.005,
});

// ── Action mapping ──────────────────────────────────────────────────

// Discrete-3 mapping: 0=DESCEND(-1), 1=FLOAT(0), 2=ASCEND(+1)
export function actionFromIndex(idx) {
    return idx - 1;  // 0→-1, 1→0, 2→+1
}

export function indexFromAction(action) {
    return action + 1;  // -1→0, 0→1, +1→2
}

// targetAlt17 mapping: 17 evenly-spaced target altitudes across the reachable
// band. The trainer's bang-bang chase converts a target into per-step ACS
// commands. This makes the agent's interface match the heuristic's, so the
// heuristic itself is a valid policy in this action space (sanity check).
export const ALT_BIN_COUNT = 17;

/**
 * Map a target-alt action index to a target altitude (m).
 * Bin 0 → altLow_m, bin (N-1) → altHigh_m, evenly spaced.
 */
export function targetAltFromIndex(idx, altLow_m, altHigh_m) {
    const t = idx / (ALT_BIN_COUNT - 1);
    return altLow_m + t * (altHigh_m - altLow_m);
}

/** Find the bin closest to a given target altitude. */
export function indexFromTargetAlt(target_m, altLow_m, altHigh_m) {
    const span = altHigh_m - altLow_m;
    if (span <= 0) return Math.floor((ALT_BIN_COUNT - 1) / 2);
    const t = (target_m - altLow_m) / span;
    return Math.max(0, Math.min(ALT_BIN_COUNT - 1, Math.round(t * (ALT_BIN_COUNT - 1))));
}

/** Resolve ACTION_COUNT from ACTION_SPACE. */
export function actionCountFor(space) {
    return space === 'targetAlt17' ? ALT_BIN_COUNT : 3;
}

/**
 * Resolve STATE_DIM from STATE_MODE. Compact = 20, wide = 74.
 *   wide layout:
 *     [0..7]    same as compact (distance, bearing×2, alt, vv, ballast, GPS-wind×2)
 *     [8..41]   34 features = 17 altitudes × (u, v) — full EKF mean
 *     [42..75]  34 features = 17 altitudes × (sigma_u, sigma_v) — diagonal covariance
 */
export function stateDimFor(mode) {
    return mode === 'wide' ? 74 : 20;
}

/**
 * Wide-mode altitude grid — matches WindEKF's altitudes (17 levels at 125 m steps
 * across the reachable band). Hard-coded because runtime.altitudeLevels may not
 * be initialised when this module loads.
 */
export const WIDE_STATE_ALTITUDES = (() => {
    const out = [];
    for (let i = 0; i < ALT_BIN_COUNT; i++) {
        // 16500 + i*125 covers 16500..18500 inclusive
        out.push(16500 + i * 125);
    }
    return out;
})();

// ── DQN Agent ───────────────────────────────────────────────────────

export class DQNAgent {
    /**
     * @param {Object} [options] - Override RL_DEFAULTS
     */
    constructor(options = {}) {
        this.config = { ...RL_DEFAULTS, ...options };
        const c = this.config;

        // Auto-resolve ACTION_COUNT from ACTION_SPACE unless caller overrode it.
        // Honour explicit ACTION_COUNT in options (used by serialised checkpoints
        // that pre-date the ACTION_SPACE flag).
        if (!('ACTION_COUNT' in options) || options.ACTION_COUNT == null) {
            c.ACTION_COUNT = actionCountFor(c.ACTION_SPACE || 'discrete3');
        }
        // Same for STATE_DIM auto-resolution from STATE_MODE.
        if (!('STATE_DIM' in options) || options.STATE_DIM == null) {
            c.STATE_DIM = stateDimFor(c.STATE_MODE || 'compact');
        }

        // Networks
        const layers = [c.STATE_DIM, ...c.HIDDEN_SIZES, c.ACTION_COUNT];
        const nnOpts = {
            optimizer:  c.OPTIMIZER || 'sgd',
            adamBeta1:  c.ADAM_BETA1,
            adamBeta2:  c.ADAM_BETA2,
        };
        this.policyNet = new NeuralNetwork(layers, c.SEED,     nnOpts);
        this.targetNet = new NeuralNetwork(layers, c.SEED,     nnOpts);
        this.targetNet.copyFrom(this.policyNet);

        // Experience replay
        this.replayBuffer = new ReplayBuffer(c.REPLAY_CAPACITY, c.SEED + 1, {
            mode:     c.REPLAY_MODE || 'uniform',
            perAlpha: c.PER_ALPHA,
            perBeta0: c.PER_BETA0,
        });

        // Exploration
        this.epsilon = c.EPSILON_START;
        this.episodeCount = 0;

        // Simple seeded RNG for action selection
        this._rngState = c.SEED >>> 0 || 1;

        // Training statistics
        this.losses = [];
    }

    /** Xorshift32 RNG for epsilon-greedy. Returns [0, 1). */
    _rand() {
        let x = this._rngState;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this._rngState = x >>> 0;
        return (this._rngState & 0x7FFFFFFF) / 0x80000000;
    }

    /**
     * Extract normalized state vector from balloon observations.
     *
     * The state vector encodes what the balloon actually knows:
     *   - GPS-derived wind at current altitude (high confidence)
     *   - EKF-estimated wind at other altitudes (lower confidence)
     *   - Uncertainty at each sample altitude (tells agent what to trust)
     *
     * The EKF uses vertical correlation: a GPS observation at the current
     * altitude updates the posterior at nearby altitudes, giving the agent
     * realistic partial information about the full wind column.
     *
     * Features (20-dim, roughly in [-1, 1]):
     *   [0]     Distance to target / station radius
     *   [1]     sin(bearing to target)
     *   [2]     cos(bearing to target)
     *   [3]     Altitude normalized [0,1] in [16.5km, 18.5km]
     *   [4]     Vertical velocity / 2.5 m/s
     *   [5]     Ballast fraction [0,1]
     *   [6-7]   u,v wind at current altitude / 20 m/s (GPS-derived, high trust)
     *   [8-10]  u,v wind + uncertainty at sample alt 0
     *   [11-13] u,v wind + uncertainty at sample alt 1
     *   [14-16] u,v wind + uncertainty at sample alt 2
     *   [17-19] u,v wind + uncertainty at sample alt 3
     *
     * @param {Object} balloonState - { lat, lon, alt_m, vv_m_s, ballast_kg }
     * @param {Function} getWindFn - (alt_m, time_s) → { u, v }
     * @param {number} time_s
     * @param {number} targetLat
     * @param {number} targetLon
     * @param {Function|null} [getUncertaintyFn] - (alt_m) → number (sigma in m/s)
     *   If null, all uncertainties default to 0.0 (legacy: perfect wind assumption).
     * @returns {Float64Array} 20-element state vector
     */
    extractState(balloonState, getWindFn, time_s, targetLat, targetLon, getUncertaintyFn = null) {
        const s = new Float64Array(this.config.STATE_DIM);
        const p = runtime.platform;
        const maxUnc = this.config.MAX_UNCERTAINTY;

        // Distance and bearing to target
        const dist = haversine(balloonState.lat, balloonState.lon, targetLat, targetLon);
        const brng = bearingFlat(balloonState.lat, balloonState.lon, targetLat, targetLon);
        const brngRad = brng * Math.PI / 180;

        s[0] = dist / p.STATION_RADIUS_M;
        s[1] = Math.sin(brngRad);
        s[2] = Math.cos(brngRad);
        s[3] = (balloonState.alt_m - runtime.altBandLow_m) /
               (runtime.altBandHigh_m - runtime.altBandLow_m);
        s[3] = Math.max(0, Math.min(1, s[3]));
        s[4] = (balloonState.vv_m_s || 0) / 2.5;
        s[5] = (balloonState.ballast_kg || 0) / p.BALLOON_BALLAST_CAPACITY_KG;
        const windCur = getWindFn(balloonState.alt_m, time_s);
        s[6] = windCur.u / 20;
        s[7] = windCur.v / 20;

        const mode = this.config.STATE_MODE || 'compact';
        if (mode === 'wide') {
            // Wide mode: full 34-dim EKF mean + 34-dim diagonal covariance
            // (per-altitude u, v means then per-altitude sigmas).
            for (let i = 0; i < ALT_BIN_COUNT; i++) {
                const alt_m = WIDE_STATE_ALTITUDES[i];
                const w = getWindFn(alt_m, time_s);
                const sigma = getUncertaintyFn ? getUncertaintyFn(alt_m) : maxUnc;
                s[8  + 2 * i]     = w.u / 20;
                s[8  + 2 * i + 1] = w.v / 20;
                // sigmas — use same value for u and v (EKF averages them in
                // getUncertainty); a future refinement could expose them
                // separately via a getUncertaintyComponents() method.
                s[42 + 2 * i]     = Math.min(1.0, sigma / maxUnc);
                s[42 + 2 * i + 1] = Math.min(1.0, sigma / maxUnc);
            }
        } else {
            // Compact mode: 4 sample altitudes × (u, v, sigma) = 12 features
            const alts = this.config.WIND_SAMPLE_ALTS;
            for (let i = 0; i < alts.length; i++) {
                const w = getWindFn(alts[i], time_s);
                const base = 8 + i * 3;
                s[base]     = w.u / 20;
                s[base + 1] = w.v / 20;
                s[base + 2] = getUncertaintyFn
                    ? Math.min(1.0, getUncertaintyFn(alts[i]) / maxUnc)
                    : 0.0;
            }
        }

        return s;
    }

    /**
     * Select an action using epsilon-greedy policy.
     *
     * @param {Float64Array} stateVec - Normalized state vector
     * @param {boolean} [explore=true] - Use epsilon-greedy (false = greedy)
     * @returns {number} Action index [0, 1, 2]
     */
    selectAction(stateVec, explore = true) {
        if (explore && this._rand() < this.epsilon) {
            return Math.floor(this._rand() * this.config.ACTION_COUNT);
        }

        const qValues = this.policyNet.forward(stateVec);
        let bestIdx = 0;
        for (let i = 1; i < qValues.length; i++) {
            if (qValues[i] > qValues[bestIdx]) bestIdx = i;
        }
        return bestIdx;
    }

    /**
     * Compute reward based on distance to target. Shape is configurable via
     * config.REWARD_SHAPE — see RL_DEFAULTS for options.
     *
     * The default 'smooth' shape mirrors Bellemare et al. 2020 eq. 1 (up to
     * constants):  r = c1·1[d ≤ R] + c2·exp(−d/τ).  This is smooth at every
     * distance, bounded in [0,1], never saturates, and preserves the discrete
     * in-radius incentive that the cliff shape provided.
     *
     * @param {number} distance_m
     * @returns {number} reward
     */
    computeReward(distance_m) {
        const R = runtime.platform.STATION_RADIUS_M;
        const shape = this.config.REWARD_SHAPE || 'smooth';
        const tau   = R * (this.config.REWARD_SCALE_R || 2.0);

        if (shape === 'cliff') {
            // Legacy: +1 inside, -min(1, d/R) outside. Documented bimodal
            // pathology — kept for reproduction of the May 2026 baseline.
            if (distance_m <= R) return 1.0;
            return -Math.min(1.0, distance_m / R);
        }

        if (shape === 'expOnly') {
            // Pure exponential proximity. Smooth but no "I made it" signal.
            return Math.exp(-distance_m / tau);
        }

        // 'smooth' (default): hybrid in-radius indicator + exponential proximity.
        // At d=0:    1.0       (perfect)
        // At d=R:    0.5 + 0.5·exp(-0.5)  ≈ 0.803
        // At d=R+ε:  0.5·exp(-(R+ε)/tau)  — smooth drop to ~0.30 at R, no cliff
        // At d=2R:   0.5·exp(-1)          ≈ 0.184
        // At d=10R:  0.5·exp(-5)          ≈ 0.003 (asymptotically zero)
        const inside    = distance_m <= R ? 1.0 : 0.0;
        const proximity = Math.exp(-distance_m / tau);
        return 0.5 * inside + 0.5 * proximity;
    }

    /**
     * Execute one DQN training step on a mini-batch from replay buffer.
     *
     * DQN update: Q(s,a) ← Q(s,a) + lr × [r + γ × max_a' Q_target(s',a') - Q(s,a)]
     *
     * @returns {number|null} Mean TD loss, or null if buffer too small
     */
    trainBatch() {
        const c = this.config;
        if (this.replayBuffer.length < c.BATCH_SIZE) return null;

        const batch = this.replayBuffer.sample(c.BATCH_SIZE);
        const usePer = this.replayBuffer.mode === 'prioritized';
        const useAdam = (this.policyNet.optimizer === 'adam');
        // SGD's manual TD clipping is necessary because of its fixed-rate
        // gradient updates; Adam's adaptive lr makes the clip unhelpful and it
        // also distorts the priority signal for PER. Skip clipping when Adam.
        const clipTd = !useAdam;

        let totalLoss = 0;
        const perIdxs    = usePer ? new Int32Array(c.BATCH_SIZE) : null;
        const perTdAbs   = usePer ? new Float64Array(c.BATCH_SIZE) : null;

        for (let bi = 0; bi < batch.length; bi++) {
            const tr = batch[bi];
            const { state, action, reward, nextState, done } = tr;
            // For n-step transitions the bootstrap discount is γ^k (k = n-step
            // length actually accumulated). For 1-step transitions it's γ.
            const bootstrapGamma = (tr.effectiveGamma != null) ? tr.effectiveGamma : c.GAMMA;
            const isW = usePer ? (tr._perWeight ?? 1.0) : 1.0;

            // Double DQN: policy net selects action, target net evaluates
            const nextQPolicy = this.policyNet.forward(nextState);
            let bestAction = 0;
            for (let i = 1; i < c.ACTION_COUNT; i++) {
                if (nextQPolicy[i] > nextQPolicy[bestAction]) bestAction = i;
            }
            const nextQTarget = this.targetNet.forward(nextState);
            const targetQ = done ? reward : reward + bootstrapGamma * nextQTarget[bestAction];

            const curQ = this.policyNet.forward(state);
            const tdError = targetQ - curQ[action];
            totalLoss += tdError * tdError;

            // Gradient signal: weighted by IS weight under PER
            const td = clipTd ? Math.max(-1, Math.min(1, tdError)) : tdError;
            const grad = new Float64Array(c.ACTION_COUNT);
            grad[action] = -2 * td * isW;
            this.policyNet.backward(grad, c.LEARNING_RATE);

            if (usePer) {
                perIdxs[bi]  = tr._perIdx;
                perTdAbs[bi] = Math.abs(tdError);
            }
        }

        if (usePer) {
            this.replayBuffer.updatePriorities(perIdxs, perTdAbs);
            this.replayBuffer.annealBeta(1.0, c.PER_BETA_ANNEAL);
        }

        // Polyak soft target update after each batch (replaces periodic hard sync).
        if ((c.POLYAK_TAU || 0) > 0) {
            this.targetNet.polyakUpdate(this.policyNet, c.POLYAK_TAU);
        }

        const meanLoss = totalLoss / c.BATCH_SIZE;
        this.losses.push(meanLoss);
        return meanLoss;
    }

    /**
     * Sync target network to policy network weights.
     * Called periodically (every TARGET_UPDATE_FREQ episodes).
     */
    syncTargetNetwork() {
        this.targetNet.copyFrom(this.policyNet);
    }

    /**
     * Decay epsilon after an episode.
     */
    decayEpsilon() {
        this.epsilon *= this.config.EPSILON_DECAY;
        this.epsilon = Math.max(this.epsilon, this.config.EPSILON_END);
        this.episodeCount++;

        // Hard sync only when Polyak is disabled (POLYAK_TAU=0).
        // With Polyak active, target net is already updated every batch step.
        if (!(this.config.POLYAK_TAU > 0) &&
            this.episodeCount % this.config.TARGET_UPDATE_FREQ === 0) {
            this.syncTargetNetwork();
        }
    }

    /**
     * Get Q-values for a state (for diagnostics / decision logging).
     * @param {Float64Array} stateVec
     * @returns {Float64Array} Q-values for [DESCEND, FLOAT, ASCEND]
     */
    getQValues(stateVec) {
        return this.policyNet.forward(stateVec);
    }

    /**
     * Store a transition in the replay buffer.
     * @param {Object} [extras] - Optional per-transition fields. Currently
     *                             supports `effectiveGamma` (used by n-step
     *                             returns to discount the bootstrap target).
     */
    remember(state, actionIdx, reward, nextState, done, extras = null) {
        this.replayBuffer.push(state, actionIdx, reward, nextState, done, extras);
    }

    /**
     * Serialize agent state for save/load.
     */
    serialize() {
        // Strip non-serializable config fields (e.g. WIND_ARCHIVE object)
        const { WIND_ARCHIVE: _drop, ...serializableConfig } = this.config;
        return {
            policyNet: this.policyNet.serialize(),
            episodeCount: this.episodeCount,
            epsilon: this.epsilon,
            config: serializableConfig,
        };
    }

    /**
     * Load agent state from serialized data.
     */
    deserialize(data) {
        this.policyNet.deserialize(data.policyNet);
        this.targetNet.copyFrom(this.policyNet);
        this.episodeCount = data.episodeCount || 0;
        this.epsilon = data.epsilon || this.config.EPSILON_END;
    }

    /**
     * Reset agent for a new training run (keep config, reset everything else).
     */
    reset() {
        const c = this.config;
        // Use whatever STATE_DIM and ACTION_COUNT were resolved at construction
        // time. Tests sometimes override these manually post-construction (e.g.
        // a.config.ACTION_COUNT = 1; a.reset()) and we must respect that. The
        // constructor's auto-resolution from ACTION_SPACE/STATE_MODE only fires
        // when those fields are *missing* from options.
        const layers = [c.STATE_DIM, ...c.HIDDEN_SIZES, c.ACTION_COUNT];
        const nnOpts = {
            optimizer:  c.OPTIMIZER || 'sgd',
            adamBeta1:  c.ADAM_BETA1,
            adamBeta2:  c.ADAM_BETA2,
        };
        this.policyNet = new NeuralNetwork(layers, c.SEED, nnOpts);
        this.targetNet = new NeuralNetwork(layers, c.SEED, nnOpts);
        this.targetNet.copyFrom(this.policyNet);
        this.replayBuffer.clear();
        this.epsilon = c.EPSILON_START;
        this.episodeCount = 0;
        this.losses = [];
        this._rngState = c.SEED >>> 0 || 1;
    }
}

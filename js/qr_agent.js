/**
 * qr_agent.js — Quantile Regression DQN (Bellemare et al. 2017,
 * "Distributional Reinforcement Learning with Quantile Regression").
 *
 * Same interface as DQNAgent (extractState, selectAction, trainBatch,
 * remember, decayEpsilon, serialize, deserialize) so the trainer can swap
 * one for the other based on config.LEARNING_MODE. The behaviour difference
 * lives entirely in the network output and the Bellman backup:
 *
 *   - Network output: ACTION_COUNT × N_QUANTILES (default 51) instead of
 *     ACTION_COUNT scalars. Each output is a quantile value of the return
 *     distribution Z(s,a) at midpoint τ_i = (i + 0.5) / N.
 *
 *   - Action selection: by default argmax over E[Z(s,a)] = mean of quantiles
 *     (recovers vanilla DQN behaviour). With CVAR_ALPHA < 1.0, action selection
 *     uses Conditional Value-at-Risk: mean of the lowest α-fraction of
 *     quantiles. CVAR_ALPHA = 0.25 → biases toward catastrophe-avoidance,
 *     which matches Loon's published configuration for stratospheric balloons
 *     (rare jet-stream entrainment events dominate worst-case TWR-50).
 *
 *   - Loss: quantile Huber (Bellemare 2017 eq. 10), pinball-weighted with
 *     τ-asymmetry. Bellman target is the empirical distribution of
 *     r + γ · z_j(s', a*) for each quantile j of the next-state distribution
 *     under the next action a* selected by Double-DQN argmax over E[Z].
 *
 * State extraction is identical to DQNAgent — see rl_agent.js.
 */

import { runtime } from './config.js';
import { haversine, bearingFlat } from './geo.js';
import { NeuralNetwork, ReplayBuffer } from './dqn.js';
import { actionCountFor, ALT_BIN_COUNT } from './rl_agent.js';

// ── QR Agent configuration ──────────────────────────────────────────

export const QR_DEFAULTS = Object.freeze({
    STATE_DIM:           20,
    HIDDEN_SIZES:        [128, 64],
    ACTION_COUNT:        17,        // auto-resolved from ACTION_SPACE
    ACTION_SPACE:        'targetAlt17',
    N_QUANTILES:         51,
    HUBER_KAPPA:         1.0,
    LEARNING_RATE:       0.0003,
    GAMMA:               0.97,
    EPSILON_START:       1.0,
    EPSILON_END:         0.05,
    EPSILON_DECAY:       0.995,
    REPLAY_CAPACITY:     30_000,
    BATCH_SIZE:          64,
    TARGET_UPDATE_FREQ:  15,
    SEED:                42,
    WIND_SAMPLE_ALTS:    [16625, 17125, 17625, 18125],
    MAX_UNCERTAINTY:     10.0,
    REWARD_SHAPE:        'smooth',
    REWARD_SCALE_R:      2.0,
    OPTIMIZER:           'adam',
    ADAM_BETA1:          0.9,
    ADAM_BETA2:          0.999,
    N_STEP:              1,
    REPLAY_MODE:         'uniform',
    PER_ALPHA:           0.6,
    PER_BETA0:           0.4,
    PER_BETA_ANNEAL:     1e-4,

    // CVaR α for action selection: 1.0 = vanilla mean-Q-equivalent (greedy
    // expected return), <1.0 = risk-averse (uses lower tail). Loon used 0.25.
    CVAR_ALPHA:          1.0,
});

export class QRAgent {
    constructor(options = {}) {
        this.config = { ...QR_DEFAULTS, ...options };
        const c = this.config;

        if (!('ACTION_COUNT' in options) || options.ACTION_COUNT == null) {
            c.ACTION_COUNT = actionCountFor(c.ACTION_SPACE || 'targetAlt17');
        }

        // Network output = ACTION_COUNT × N_QUANTILES. Stored as a flat output
        // array indexed by [a * N + i].
        const outDim = c.ACTION_COUNT * c.N_QUANTILES;
        const layers = [c.STATE_DIM, ...c.HIDDEN_SIZES, outDim];
        const nnOpts = {
            optimizer:  c.OPTIMIZER || 'adam',
            adamBeta1:  c.ADAM_BETA1,
            adamBeta2:  c.ADAM_BETA2,
        };
        this.policyNet = new NeuralNetwork(layers, c.SEED, nnOpts);
        this.targetNet = new NeuralNetwork(layers, c.SEED, nnOpts);
        this.targetNet.copyFrom(this.policyNet);

        this.replayBuffer = new ReplayBuffer(c.REPLAY_CAPACITY, c.SEED + 1, {
            mode:     c.REPLAY_MODE || 'uniform',
            perAlpha: c.PER_ALPHA,
            perBeta0: c.PER_BETA0,
        });

        this.epsilon      = c.EPSILON_START;
        this.episodeCount = 0;
        this._rngState    = c.SEED >>> 0 || 1;
        this.losses       = [];

        // Quantile midpoints τ_i = (i + 0.5) / N
        this.taus = new Float64Array(c.N_QUANTILES);
        for (let i = 0; i < c.N_QUANTILES; i++) {
            this.taus[i] = (i + 0.5) / c.N_QUANTILES;
        }
    }

    _rand() {
        let x = this._rngState;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        this._rngState = x >>> 0;
        return (this._rngState & 0x7FFFFFFF) / 0x80000000;
    }

    /** Identical to DQNAgent.extractState — kept here so QRAgent stands alone. */
    extractState(balloonState, getWindFn, time_s, targetLat, targetLon, getUncertaintyFn = null) {
        const s = new Float64Array(this.config.STATE_DIM);
        const p = runtime.platform;
        const maxUnc = this.config.MAX_UNCERTAINTY;

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
        return s;
    }

    /**
     * Compute per-action expected return E[Z(s,a)] = mean of action's quantiles.
     */
    _meanQ(quantilesFlat, action_count, n_quantiles) {
        const out = new Float64Array(action_count);
        for (let a = 0; a < action_count; a++) {
            let sum = 0;
            const base = a * n_quantiles;
            for (let i = 0; i < n_quantiles; i++) sum += quantilesFlat[base + i];
            out[a] = sum / n_quantiles;
        }
        return out;
    }

    /**
     * Compute per-action CVaR_α(Z(s,a)) = mean of the lowest α fraction of
     * quantiles. α=1.0 is identical to mean. Quantiles are stored in increasing
     * τ order so the "lowest α" is the prefix [0, α·N).
     */
    _cvarQ(quantilesFlat, action_count, n_quantiles, alpha) {
        if (alpha >= 0.999) return this._meanQ(quantilesFlat, action_count, n_quantiles);
        const k = Math.max(1, Math.floor(alpha * n_quantiles));
        const out = new Float64Array(action_count);
        for (let a = 0; a < action_count; a++) {
            const base = a * n_quantiles;
            // The QR-DQN paper does NOT guarantee monotonicity of quantiles,
            // so we sort to extract the actual lower tail. Fast for small N.
            const slice = Array.from(quantilesFlat.subarray(base, base + n_quantiles)).sort((x, y) => x - y);
            let sum = 0;
            for (let i = 0; i < k; i++) sum += slice[i];
            out[a] = sum / k;
        }
        return out;
    }

    /** Returns Float64Array of length ACTION_COUNT (per-action expected/CVaR Q). */
    getQValues(stateVec) {
        const c = this.config;
        const flat = this.policyNet.forward(stateVec);
        return c.CVAR_ALPHA < 1.0
            ? this._cvarQ(flat, c.ACTION_COUNT, c.N_QUANTILES, c.CVAR_ALPHA)
            : this._meanQ(flat, c.ACTION_COUNT, c.N_QUANTILES);
    }

    selectAction(stateVec, explore = true) {
        if (explore && this._rand() < this.epsilon) {
            return Math.floor(this._rand() * this.config.ACTION_COUNT);
        }
        const q = this.getQValues(stateVec);
        let best = 0;
        for (let i = 1; i < q.length; i++) if (q[i] > q[best]) best = i;
        return best;
    }

    computeReward(distance_m) {
        const R = runtime.platform.STATION_RADIUS_M;
        const shape = this.config.REWARD_SHAPE || 'smooth';
        const tau   = R * (this.config.REWARD_SCALE_R || 2.0);
        if (shape === 'cliff') {
            if (distance_m <= R) return 1.0;
            return -Math.min(1.0, distance_m / R);
        }
        if (shape === 'expOnly') return Math.exp(-distance_m / tau);
        const inside = distance_m <= R ? 1.0 : 0.0;
        return 0.5 * inside + 0.5 * Math.exp(-distance_m / tau);
    }

    /**
     * Quantile Huber loss training step.
     *
     * For each transition:
     *   target distribution z' = r + γ · Z_target(s', a*)  (a* by Double-DQN argmax over E[Z])
     *   predicted distribution q = Z_policy(s, a)
     *   for each (i, j):
     *     δ_ij = z'_j - q_i
     *     ρ_τi(δ_ij) = |τ_i - 1[δ_ij < 0]| · L_κ(δ_ij)
     *   L = mean over (i,j)
     *
     * Gradient at output a*N+i:
     *   ∂L/∂q_i = -(1/N) · Σ_j |τ_i - 1[δ<0]| · clip(δ, -κ, κ) · sign-from-δ-relation
     *   (with derivative of Huber: δ if |δ|<κ, κ·sign(δ) otherwise; chain rule
     *    for ∂δ/∂q_i = -1)
     */
    trainBatch() {
        const c = this.config;
        if (this.replayBuffer.length < c.BATCH_SIZE) return null;

        const batch = this.replayBuffer.sample(c.BATCH_SIZE);
        const N = c.N_QUANTILES;
        const A = c.ACTION_COUNT;
        const kappa = c.HUBER_KAPPA;
        const usePer = this.replayBuffer.mode === 'prioritized';

        let totalLoss = 0;
        const perIdxs  = usePer ? new Int32Array(c.BATCH_SIZE) : null;
        const perTd    = usePer ? new Float64Array(c.BATCH_SIZE) : null;

        for (let bi = 0; bi < batch.length; bi++) {
            const tr = batch[bi];
            const { state, action, reward, nextState, done } = tr;
            const bootstrapGamma = (tr.effectiveGamma != null) ? tr.effectiveGamma : c.GAMMA;
            const isW = usePer ? (tr._perWeight ?? 1.0) : 1.0;

            // Predicted quantiles for s, action.
            const qFlat = this.policyNet.forward(state);   // length A·N

            // Target distribution: r + γ · Z_target(s', a*)
            // a* via Double-DQN: argmax over policy's E[Z(s', ·)]
            const policyNextFlat = this.policyNet.forward(nextState);
            const policyNextQ    = this._meanQ(policyNextFlat, A, N);
            let aStar = 0;
            for (let i = 1; i < A; i++) if (policyNextQ[i] > policyNextQ[aStar]) aStar = i;
            // re-forward through policyNet (we need its activations for backprop on s)
            // means we clobbered _activations cache when we ran nextState forward.
            // Re-forward state to restore the cache for backward().
            const targetNextFlat = this.targetNet.forward(nextState);
            // Build z' (length N): r + γ · z(s', a*)_j (or just r if done)
            const zPrime = new Float64Array(N);
            const baseAStar = aStar * N;
            if (done) {
                for (let j = 0; j < N; j++) zPrime[j] = reward;
            } else {
                for (let j = 0; j < N; j++) zPrime[j] = reward + bootstrapGamma * targetNextFlat[baseAStar + j];
            }

            // Compute the loss + per-output-quantile gradient for action.
            // We accumulate gradient[i] = mean_j of ∂ρ_τi(δ_ij)/∂q_i
            const grad = new Float64Array(A * N);
            const baseAct = action * N;
            let sumLoss = 0;

            // Re-forward state so policyNet's _activations cache is for state
            // (target net was last forward'd through nextState — independent
            // because target.forward doesn't share cache with policy).
            this.policyNet.forward(state);
            const qPred = qFlat;  // already saved; it's the same Float64Array

            for (let i = 0; i < N; i++) {
                const tau = this.taus[i];
                const qi = qPred[baseAct + i];
                let gradI = 0;
                for (let j = 0; j < N; j++) {
                    const delta = zPrime[j] - qi;
                    const indicator = delta < 0 ? 1.0 : 0.0;
                    const tauWeight = Math.abs(tau - indicator);
                    // Huber
                    let huberGrad;
                    let huberLoss;
                    if (Math.abs(delta) <= kappa) {
                        huberGrad = delta;        // ∂L_κ/∂δ
                        huberLoss = 0.5 * delta * delta;
                    } else {
                        huberGrad = kappa * Math.sign(delta);
                        huberLoss = kappa * (Math.abs(delta) - 0.5 * kappa);
                    }
                    sumLoss += tauWeight * huberLoss;
                    // ∂ρ/∂q_i = -tauWeight · ∂L_κ/∂δ
                    gradI += -tauWeight * huberGrad;
                }
                // mean over j; sign already includes the chain-rule (-1 for ∂δ/∂q)
                grad[baseAct + i] = (gradI / N) * isW;
            }

            const meanLoss = sumLoss / (N * N);
            totalLoss += meanLoss;
            this.policyNet.backward(grad, c.LEARNING_RATE);

            if (usePer) {
                perIdxs[bi] = tr._perIdx;
                perTd[bi]   = meanLoss;   // use loss as priority signal
            }
        }

        if (usePer) {
            this.replayBuffer.updatePriorities(perIdxs, perTd);
            this.replayBuffer.annealBeta(1.0, c.PER_BETA_ANNEAL);
        }

        const meanLoss = totalLoss / c.BATCH_SIZE;
        this.losses.push(meanLoss);
        return meanLoss;
    }

    syncTargetNetwork() {
        this.targetNet.copyFrom(this.policyNet);
    }

    decayEpsilon() {
        this.epsilon *= this.config.EPSILON_DECAY;
        this.epsilon = Math.max(this.epsilon, this.config.EPSILON_END);
        this.episodeCount++;
        if (this.episodeCount % this.config.TARGET_UPDATE_FREQ === 0) {
            this.syncTargetNetwork();
        }
    }

    remember(state, actionIdx, reward, nextState, done, extras = null) {
        this.replayBuffer.push(state, actionIdx, reward, nextState, done, extras);
    }

    serialize() {
        const { WIND_ARCHIVE: _drop, ...serializableConfig } = this.config;
        return {
            policyNet:   this.policyNet.serialize(),
            episodeCount: this.episodeCount,
            epsilon:      this.epsilon,
            config:       serializableConfig,
            // Sentinel so the trainer can identify this as a QR-DQN checkpoint
            // without importing the class.
            agentType:    'qr-dqn',
        };
    }

    deserialize(data) {
        this.policyNet.deserialize(data.policyNet);
        this.targetNet.copyFrom(this.policyNet);
        this.episodeCount = data.episodeCount || 0;
        this.epsilon = data.epsilon || this.config.EPSILON_END;
    }

    reset() {
        const c = this.config;
        // Use the already-resolved ACTION_COUNT (constructor handled
        // auto-resolution from ACTION_SPACE if needed).
        const outDim = c.ACTION_COUNT * c.N_QUANTILES;
        const layers = [c.STATE_DIM, ...c.HIDDEN_SIZES, outDim];
        const nnOpts = {
            optimizer:  c.OPTIMIZER || 'adam',
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

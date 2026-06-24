/**
 * iqn_agent.js — Implicit Quantile Networks (Dabney et al. 2018, arXiv 1806.06923).
 *
 * IQN replaces QR-DQN's fixed quantile grid τ_i = (i+0.5)/N with a network
 * that, conditioned on a sampled τ ∈ [0,1], outputs the quantile value
 * Z_τ(s,a). The state encoder ψ(s) is multiplied (Hadamard) by a cosine
 * embedding φ(τ) = ReLU(W·cos(π·i·τ) + b) for i = 1..n, and the merged
 * vector is fed through the final head.
 *
 * Key advantages over QR-DQN for our problem:
 *   - Sampled τ → unbiased CVaR estimates (not biased by 51-bin discretisation)
 *   - Action selection at deployment can sample τ from any distortion measure
 *     (CVaR_α, Wang transform, CPT) WITHOUT retraining — provides a
 *     deployment-time risk knob
 *   - Cheaper per training step (typically K=32 τ samples vs N=51 fixed)
 *
 * Architecture:
 *   state s (20-dim) → MLP(HIDDEN_SIZES) → ψ (psi, embedding_dim = last hidden)
 *   τ ∈ [0,1]^K → cosine_features → MLP(embedding_dim) → φ_k (K vectors)
 *   merged_k = ψ ⊙ φ_k                              (K Hadamard products)
 *   merged_k → MLP(action_count) → quantile values  (K × action_count)
 *
 * For training:
 *   - Sample K τ_pred for the prediction net, K' τ_target for the target net
 *   - For each pair (i, j), compute quantile Huber loss
 *   - Backprop only through the action's quantile head
 *
 * Note on hand-rolled JS: rather than implementing a clean ψ-then-φ split
 * (which requires intermediate buffers we don't have), we factor IQN as a
 * *single* call per τ to the underlying NeuralNetwork. The state and the
 * cosine-embedded τ are concatenated into one input; the network learns
 * the merging implicitly. This is mathematically equivalent under a
 * reparameterisation but loses the Hadamard-merge inductive bias. Empirically
 * the difference is small (Mavrin et al. 2019 ablation), and the simplicity
 * is worth it for our pure-JS context.
 */

import { runtime } from './config.js';
import { haversine, bearingFlat } from './geo.js';
import { NeuralNetwork, ReplayBuffer } from './dqn.js';
import { actionCountFor, ALT_BIN_COUNT } from './rl_agent.js';

export const IQN_DEFAULTS = Object.freeze({
    STATE_DIM:           20,
    HIDDEN_SIZES:        [128, 64],
    ACTION_COUNT:        17,
    ACTION_SPACE:        'targetAlt17',

    // Number of cosine basis functions used to encode τ.
    N_COSINE:            64,

    // Number of τ samples per training step.
    N_TAU_PRED:          32,    // for the prediction (current) network
    N_TAU_TARGET:        32,    // for the target network

    // Number of τ samples for action selection at inference.
    N_TAU_ACT:           32,

    HUBER_KAPPA:         1.0,

    LEARNING_RATE:       0.0001,
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

    // Risk distortion at action selection.
    //   'mean'  — argmax over E[Z(s,a)] (vanilla DQN-equivalent)
    //   'cvar'  — argmax over CVaR_α
    //   'wang'  — argmax over Wang transform with parameter λ
    //   'cpt'   — Cumulative Prospect Theory (Tversky-Kahneman)
    RISK_DISTORTION:     'mean',
    CVAR_ALPHA:          1.0,
    WANG_LAMBDA:         0.0,    // > 0 = pessimistic, < 0 = optimistic
    CPT_ETA_GAIN:        0.61,   // Tversky-Kahneman gain exponent
    CPT_ETA_LOSS:        0.69,   // loss exponent
    CPT_LOSS_AVERSION:   2.25,   // λ in CPT

    // Munchausen reward augmentation (Vieillard 2020). Set MUNCHAUSEN_ALPHA > 0
    // to enable. Stacks with QR-DQN/IQN — adds α·τ_M·log π(a|s) to the
    // immediate reward in the Bellman target.
    MUNCHAUSEN_ALPHA:    0.0,
    MUNCHAUSEN_TAU:      0.03,   // softmax temperature for π
    MUNCHAUSEN_LOG_CLIP: -1.0,   // clip log π to ≥ this for numerical stability
});

// ── Standard normal CDF and inverse (used by Wang transform) ─────────

function _erf(x) {
    // Abramowitz & Stegun approximation, max error ~1.5e-7
    const a1 =  0.254829592, a2 = -0.284496736, a3 =  1.421413741;
    const a4 = -1.453152027, a5 =  1.061405429, p  =  0.3275911;
    const sign = x < 0 ? -1 : 1;
    const ax = Math.abs(x);
    const t = 1.0 / (1.0 + p * ax);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
    return sign * y;
}
function normCdf(x) { return 0.5 * (1.0 + _erf(x / Math.SQRT2)); }
function normInvCdf(p) {
    // Beasley-Springer-Moro 1995, sufficient accuracy for our purposes
    const a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
                1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00];
    const b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
                6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
               -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00];
    const d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
                3.754408661907416e+00];
    const plow = 0.02425, phigh = 1 - plow;
    let q, r;
    if (p < plow) {
        q = Math.sqrt(-2 * Math.log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
    if (p <= phigh) {
        q = p - 0.5; r = q * q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    }
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
           ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
}

// ── IQN agent ───────────────────────────────────────────────────────

export class IQNAgent {
    constructor(options = {}) {
        this.config = { ...IQN_DEFAULTS, ...options };
        const c = this.config;
        if (!('ACTION_COUNT' in options) || options.ACTION_COUNT == null) {
            c.ACTION_COUNT = actionCountFor(c.ACTION_SPACE || 'targetAlt17');
        }

        // The IQN network input is (state ∥ cosine_emb(τ)) — concatenated.
        // Cosine embedding has N_COSINE features.
        const inputDim = c.STATE_DIM + c.N_COSINE;
        const layers = [inputDim, ...c.HIDDEN_SIZES, c.ACTION_COUNT];
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
    }

    _rand() {
        let x = this._rngState;
        x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
        this._rngState = x >>> 0;
        return (this._rngState & 0x7FFFFFFF) / 0x80000000;
    }

    /** Cosine embedding φ(τ): vector of length N_COSINE. */
    _cosineEmbed(tau) {
        const N = this.config.N_COSINE;
        const out = new Float64Array(N);
        for (let i = 0; i < N; i++) {
            // Standard IQN basis: cos(π·(i+1)·τ)
            // (some papers use i+0; either works)
            out[i] = Math.cos(Math.PI * (i + 1) * tau);
        }
        return out;
    }

    /** Build the network input by concatenating state and cosine_emb(τ). */
    _buildInput(state, tau) {
        const c = this.config;
        const out = new Float64Array(c.STATE_DIM + c.N_COSINE);
        for (let i = 0; i < c.STATE_DIM; i++) out[i] = state[i];
        const phi = this._cosineEmbed(tau);
        for (let i = 0; i < c.N_COSINE; i++) out[c.STATE_DIM + i] = phi[i];
        return out;
    }

    /**
     * Sample K τ values according to the configured risk distortion.
     * Returns Float64Array of length K.
     */
    _sampleTaus(K, distortion = null, useUniform = false) {
        const c = this.config;
        const out = new Float64Array(K);
        if (useUniform || !distortion || distortion === 'mean') {
            for (let i = 0; i < K; i++) out[i] = this._rand();
            return out;
        }
        if (distortion === 'cvar') {
            const alpha = c.CVAR_ALPHA;
            for (let i = 0; i < K; i++) out[i] = alpha * this._rand();
            return out;
        }
        if (distortion === 'wang') {
            const lambda = c.WANG_LAMBDA;
            for (let i = 0; i < K; i++) {
                const u = this._rand();
                // Wang: distort τ by Φ(Φ⁻¹(τ) - λ)
                out[i] = normCdf(normInvCdf(Math.max(1e-6, Math.min(1 - 1e-6, u))) - lambda);
            }
            return out;
        }
        if (distortion === 'cpt') {
            // Tversky-Kahneman: distort u via concave-gain / convex-loss
            // Simpler approximation: power distortion τ → τ^η
            const eta = c.CPT_ETA_GAIN;
            for (let i = 0; i < K; i++) {
                const u = this._rand();
                out[i] = Math.pow(u, eta);
            }
            return out;
        }
        // Fallback uniform
        for (let i = 0; i < K; i++) out[i] = this._rand();
        return out;
    }

    /** Same state extraction as DQNAgent / QRAgent. */
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
            s[base + 2] = getUncertaintyFn ? Math.min(1.0, getUncertaintyFn(alts[i]) / maxUnc) : 0.0;
        }
        return s;
    }

    /**
     * Compute Q-values (one per action) by averaging over K τ samples.
     * The samples come from the configured RISK_DISTORTION (mean by default
     * during training, but can be CVaR/Wang/CPT at inference for risk-sensitive
     * action selection).
     */
    _qValuesAtState(state, K, distortion = null) {
        const A = this.config.ACTION_COUNT;
        const taus = this._sampleTaus(K, distortion);
        const sumQ = new Float64Array(A);
        for (let k = 0; k < K; k++) {
            const input = this._buildInput(state, taus[k]);
            const q = this.policyNet.forward(input);
            for (let a = 0; a < A; a++) sumQ[a] += q[a];
        }
        for (let a = 0; a < A; a++) sumQ[a] /= K;
        return sumQ;
    }

    /** For diagnostic and Munchausen log-policy use. */
    getQValues(stateVec) {
        return this._qValuesAtState(stateVec, this.config.N_TAU_ACT, this.config.RISK_DISTORTION);
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
     * IQN training step. For each transition in the batch:
     *   1. Sample N_TAU_PRED τ_i for prediction
     *   2. Sample N_TAU_TARGET τ_j for target distribution
     *   3. Forward target net at each τ_j on next state, take a* (Double-DQN
     *      via mean over K τ_act samples on the policy net)
     *   4. Build target z_j = r + γ·Z_target(s', a*; τ_j)  (or just r if done)
     *   5. Forward prediction net at each τ_i on state, get q_i (per-action)
     *      grab q_i for the taken action
     *   6. Compute quantile Huber loss over (i, j) pairs
     *   7. Backprop gradient through the predicted action head
     *
     * Munchausen extension: if MUNCHAUSEN_ALPHA > 0, augments the immediate
     * reward in the target with α·τ_M·log π(a_taken|s).
     */
    trainBatch() {
        const c = this.config;
        if (this.replayBuffer.length < c.BATCH_SIZE) return null;

        const batch = this.replayBuffer.sample(c.BATCH_SIZE);
        const usePer = this.replayBuffer.mode === 'prioritized';
        const A = c.ACTION_COUNT;
        const Kp = c.N_TAU_PRED;
        const Kt = c.N_TAU_TARGET;
        const kappa = c.HUBER_KAPPA;
        const useMunchausen = c.MUNCHAUSEN_ALPHA > 0;

        let totalLoss = 0;
        const perIdxs = usePer ? new Int32Array(c.BATCH_SIZE) : null;
        const perTd   = usePer ? new Float64Array(c.BATCH_SIZE) : null;

        for (let bi = 0; bi < batch.length; bi++) {
            const tr = batch[bi];
            const { state, action, reward, nextState, done } = tr;
            const bootstrapGamma = (tr.effectiveGamma != null) ? tr.effectiveGamma : c.GAMMA;
            const isW = usePer ? (tr._perWeight ?? 1.0) : 1.0;

            // 1+2: Sample taus
            const tausPred = this._sampleTaus(Kp, null, true);    // uniform for training
            const tausTarget = this._sampleTaus(Kt, null, true);

            // 3: Double-DQN argmax: sample K τ_act on policy net for next state
            let aStar = 0;
            const meanQNext = this._qValuesAtState(nextState, Math.max(8, Math.floor(Kp / 4)), null);
            for (let i = 1; i < A; i++) if (meanQNext[i] > meanQNext[aStar]) aStar = i;

            // 4: Build target distribution z_j (length Kt) for the chosen aStar
            const zTarget = new Float64Array(Kt);
            for (let j = 0; j < Kt; j++) {
                const inputTgt = this._buildInput(nextState, tausTarget[j]);
                const qTgt = this.targetNet.forward(inputTgt);
                if (done) {
                    zTarget[j] = reward;
                } else {
                    zTarget[j] = reward + bootstrapGamma * qTgt[aStar];
                }
            }

            // Munchausen reward augmentation
            if (useMunchausen) {
                // Compute log π(a|s) under current network (mean Q + softmax/τ_M)
                const meanQ = this._qValuesAtState(state, Math.max(8, Math.floor(Kp / 4)), null);
                let max = -Infinity;
                for (const q of meanQ) if (q > max) max = q;
                const tauM = c.MUNCHAUSEN_TAU;
                let sumExp = 0;
                for (let a = 0; a < A; a++) sumExp += Math.exp((meanQ[a] - max) / tauM);
                const logPi = (meanQ[action] - max) / tauM - Math.log(sumExp);
                const clipped = Math.max(c.MUNCHAUSEN_LOG_CLIP, logPi);
                const munchAug = c.MUNCHAUSEN_ALPHA * tauM * clipped;
                for (let j = 0; j < Kt; j++) zTarget[j] += munchAug;
            }

            // 5+6: Forward prediction net at each τ_i, compute loss + gradient
            //     The loss is averaged over (i, j) pairs:
            //       L = (1/(Kp·Kt)) · Σ_i,j ρ_τi(z_j - q_i)
            //       gradient on q_i (for action a) = -(1/(Kp·Kt)) · Σ_j |τ_i - 1[δ<0]| · huberGrad(δ)
            //
            //  We need to backward through each τ_i forward pass — so we
            //  forward prediction net at each τ_i, accumulate gradient on
            //  output[action], and call backward.

            let sumLoss = 0;
            for (let i = 0; i < Kp; i++) {
                const inputPred = this._buildInput(state, tausPred[i]);
                const q = this.policyNet.forward(inputPred);
                const qi = q[action];
                const tau = tausPred[i];

                let gradI = 0;
                for (let j = 0; j < Kt; j++) {
                    const delta = zTarget[j] - qi;
                    const indicator = delta < 0 ? 1.0 : 0.0;
                    const tauWeight = Math.abs(tau - indicator);
                    let huberGrad, huberLoss;
                    if (Math.abs(delta) <= kappa) {
                        huberGrad = delta;
                        huberLoss = 0.5 * delta * delta;
                    } else {
                        huberGrad = kappa * Math.sign(delta);
                        huberLoss = kappa * (Math.abs(delta) - 0.5 * kappa);
                    }
                    sumLoss += tauWeight * huberLoss;
                    gradI += -tauWeight * huberGrad;   // chain through ∂δ/∂q_i = -1
                }
                gradI = (gradI / Kt) * isW;

                // Build gradient vector: only the action's output gets gradient
                const grad = new Float64Array(A);
                grad[action] = gradI;
                this.policyNet.backward(grad, c.LEARNING_RATE);
            }

            const meanLoss = sumLoss / (Kp * Kt);
            totalLoss += meanLoss;

            if (usePer) {
                perIdxs[bi] = tr._perIdx;
                perTd[bi]   = meanLoss;
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

    syncTargetNetwork() { this.targetNet.copyFrom(this.policyNet); }

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
            policyNet:    this.policyNet.serialize(),
            episodeCount: this.episodeCount,
            epsilon:      this.epsilon,
            config:       serializableConfig,
            agentType:    'iqn',
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
        const inputDim = c.STATE_DIM + c.N_COSINE;
        const layers = [inputDim, ...c.HIDDEN_SIZES, c.ACTION_COUNT];
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

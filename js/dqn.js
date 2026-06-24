/**
 * dqn.js — Pure JavaScript neural network and replay buffer for DQN (Q2.1).
 *
 * Implements a feedforward neural network with backpropagation (ReLU hidden
 * layers, linear output) and a circular replay buffer for experience replay.
 *
 * No external dependencies — uses only standard JavaScript. Float64Arrays
 * for weight storage to maximize performance on the ~3000 weights in a
 * 15→64→32→3 network.
 *
 * Based on the Python NeuralNetwork in loon_navigator/navigation/rl_controller.py,
 * adapted for the JS tactical simulator.
 */

// ── PRNG (same xorshift32 as wind_degrader.js) ─────────────────────

class PRNG {
    constructor(seed = 42) {
        this.state = seed >>> 0 || 1;
    }

    next() {
        let x = this.state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this.state = x >>> 0;
        return (this.state & 0x7FFFFFFF) / 0x80000000;
    }

    nextGaussian() {
        const u1 = this.next() || 1e-10;
        const u2 = this.next();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /** Uniform integer in [0, max) */
    nextInt(max) {
        return Math.floor(this.next() * max);
    }
}

// ── Neural Network ──────────────────────────────────────────────────

/**
 * Feedforward neural network with ReLU hidden activations and linear output.
 *
 * Architecture: layerSizes = [input, hidden1, hidden2, ..., output]
 * Weight initialization: Xavier/Glorot (sqrt(2/(in+out)))
 * Training: mini-batch SGD or Adam via backpropagation. Choice via the
 *           `optimizer` field (set after construction or via setOptimizer).
 *
 * Weights are stored as flat Float64Arrays per layer for cache efficiency.
 * Layout: weights[layer] is a Float64Array of size (out × in), row-major.
 * Access pattern: w[neuron * inputSize + input]
 */
export class NeuralNetwork {
    /**
     * @param {number[]} layerSizes - e.g. [15, 64, 32, 3]
     * @param {number} [seed=42] - PRNG seed for weight initialization
     * @param {Object} [options]
     * @param {'sgd'|'adam'} [options.optimizer='sgd'] - Optimization algorithm
     * @param {number} [options.adamBeta1=0.9]
     * @param {number} [options.adamBeta2=0.999]
     * @param {number} [options.adamEps=1e-8]
     */
    constructor(layerSizes, seed = 42, options = {}) {
        this.layerSizes = layerSizes;
        this.numLayers = layerSizes.length;

        this.weights = [];  // Float64Array per layer transition
        this.biases = [];   // Float64Array per layer transition

        const rng = new PRNG(seed);

        for (let l = 0; l < this.numLayers - 1; l++) {
            const inSize = layerSizes[l];
            const outSize = layerSizes[l + 1];
            const xavierStd = Math.sqrt(2.0 / (inSize + outSize));

            const w = new Float64Array(outSize * inSize);
            const b = new Float64Array(outSize);

            for (let i = 0; i < w.length; i++) {
                w[i] = rng.nextGaussian() * xavierStd;
            }
            // biases initialize to 0

            this.weights.push(w);
            this.biases.push(b);
        }

        // Cache for forward/backward pass
        this._activations = [];
        this._zValues = [];

        // Optimizer state (Adam moments). Lazily allocated when optimizer='adam'
        // first runs. Keeps memory zero for SGD-only callers.
        this.optimizer    = options.optimizer || 'sgd';
        this.adamBeta1    = options.adamBeta1 ?? 0.9;
        this.adamBeta2    = options.adamBeta2 ?? 0.999;
        this.adamEps      = options.adamEps   ?? 1e-8;
        this._adamM_w     = null;     // first moment (weights)
        this._adamM_b     = null;     // first moment (biases)
        this._adamV_w     = null;     // second moment (weights)
        this._adamV_b     = null;     // second moment (biases)
        this._adamT       = 0;        // step counter (used for bias correction)
    }

    /**
     * Allocate Adam moment buffers (called lazily on first Adam step).
     */
    _initAdamState() {
        const numTransitions = this.numLayers - 1;
        this._adamM_w = new Array(numTransitions);
        this._adamM_b = new Array(numTransitions);
        this._adamV_w = new Array(numTransitions);
        this._adamV_b = new Array(numTransitions);
        for (let l = 0; l < numTransitions; l++) {
            const wLen = this.weights[l].length;
            const bLen = this.biases[l].length;
            this._adamM_w[l] = new Float64Array(wLen);
            this._adamM_b[l] = new Float64Array(bLen);
            this._adamV_w[l] = new Float64Array(wLen);
            this._adamV_b[l] = new Float64Array(bLen);
        }
    }

    /**
     * Switch optimizer at runtime. Preserves weights, resets moment state.
     */
    setOptimizer(name) {
        this.optimizer = name;
        this._adamT = 0;
        this._adamM_w = this._adamM_b = this._adamV_w = this._adamV_b = null;
    }

    /**
     * Forward pass. Returns output vector.
     * Caches activations and pre-activation values for backprop.
     *
     * @param {Float64Array|number[]} input
     * @returns {Float64Array} output
     */
    forward(input) {
        const act = new Float64Array(input);
        this._activations = [act];
        this._zValues = [];

        let current = act;

        for (let l = 0; l < this.numLayers - 1; l++) {
            const inSize = this.layerSizes[l];
            const outSize = this.layerSizes[l + 1];
            const w = this.weights[l];
            const b = this.biases[l];
            const isOutput = (l === this.numLayers - 2);

            const z = new Float64Array(outSize);
            const a = new Float64Array(outSize);

            for (let j = 0; j < outSize; j++) {
                let sum = b[j];
                const offset = j * inSize;
                for (let i = 0; i < inSize; i++) {
                    sum += w[offset + i] * current[i];
                }
                z[j] = sum;
                a[j] = isOutput ? sum : Math.max(0, sum);  // ReLU or linear
            }

            this._zValues.push(z);
            this._activations.push(a);
            current = a;
        }

        return current;
    }

    /**
     * Backpropagation. Updates weights and biases using the configured optimizer.
     *
     * SGD path: per-parameter gradient clipping at ±1.0 (legacy behavior).
     * Adam path: standard Adam with bias correction; no manual clipping
     *            (Adam's adaptive lr handles scale automatically).
     *
     * @param {Float64Array|number[]} lossGrad - dL/dOutput (length = output size)
     * @param {number} lr - learning rate
     */
    backward(lossGrad, lr) {
        const numTransitions = this.numLayers - 1;
        const useAdam = this.optimizer === 'adam';

        if (useAdam) {
            if (this._adamM_w === null) this._initAdamState();
            this._adamT++;
        }
        const t = this._adamT;
        const b1 = this.adamBeta1;
        const b2 = this.adamBeta2;
        const eps = this.adamEps;
        const b1corr = useAdam ? (1 - Math.pow(b1, t)) : 1;
        const b2corr = useAdam ? (1 - Math.pow(b2, t)) : 1;
        const SGD_GRAD_CLIP = 1.0;

        // Start with output layer delta
        let delta = new Float64Array(lossGrad);

        // Backprop through layers (output → input)
        for (let l = numTransitions - 1; l >= 0; l--) {
            const inSize = this.layerSizes[l];
            const outSize = this.layerSizes[l + 1];
            const activation = this._activations[l];
            const w = this.weights[l];
            const bArr = this.biases[l];

            if (useAdam) {
                const mW = this._adamM_w[l];
                const mB = this._adamM_b[l];
                const vW = this._adamV_w[l];
                const vB = this._adamV_b[l];
                for (let j = 0; j < outSize; j++) {
                    const offset = j * inSize;
                    for (let i = 0; i < inSize; i++) {
                        const grad = delta[j] * activation[i];
                        const k = offset + i;
                        mW[k] = b1 * mW[k] + (1 - b1) * grad;
                        vW[k] = b2 * vW[k] + (1 - b2) * grad * grad;
                        const mHat = mW[k] / b1corr;
                        const vHat = vW[k] / b2corr;
                        w[k] -= lr * mHat / (Math.sqrt(vHat) + eps);
                    }
                    const bGrad = delta[j];
                    mB[j] = b1 * mB[j] + (1 - b1) * bGrad;
                    vB[j] = b2 * vB[j] + (1 - b2) * bGrad * bGrad;
                    const mHatB = mB[j] / b1corr;
                    const vHatB = vB[j] / b2corr;
                    bArr[j] -= lr * mHatB / (Math.sqrt(vHatB) + eps);
                }
            } else {
                // SGD with per-parameter gradient clipping
                for (let j = 0; j < outSize; j++) {
                    const offset = j * inSize;
                    for (let i = 0; i < inSize; i++) {
                        const grad = delta[j] * activation[i];
                        const clipped = Math.max(-SGD_GRAD_CLIP, Math.min(SGD_GRAD_CLIP, grad));
                        w[offset + i] -= lr * clipped;
                    }
                    const bGrad = Math.max(-SGD_GRAD_CLIP, Math.min(SGD_GRAD_CLIP, delta[j]));
                    bArr[j] -= lr * bGrad;
                }
            }

            // Compute delta for previous layer (if not input layer)
            if (l > 0) {
                const prevDelta = new Float64Array(inSize);
                const prevZ = this._zValues[l - 1];

                for (let i = 0; i < inSize; i++) {
                    let sum = 0;
                    for (let j = 0; j < outSize; j++) {
                        sum += w[j * inSize + i] * delta[j];
                    }
                    // ReLU derivative
                    prevDelta[i] = prevZ[i] > 0 ? sum : 0;
                }
                delta = prevDelta;
            }
        }
    }

    /**
     * Copy all weights and biases from another network (for target network sync).
     * @param {NeuralNetwork} other
     */
    copyFrom(other) {
        for (let l = 0; l < this.numLayers - 1; l++) {
            this.weights[l].set(other.weights[l]);
            this.biases[l].set(other.biases[l]);
        }
    }

    /**
     * Polyak (soft) update: this ← (1-tau)*this + tau*source.
     * Use instead of copyFrom for smoother target network tracking.
     * tau=0.005 matches DDPG/TD3 convention.
     * @param {NeuralNetwork} source - Policy network to blend toward
     * @param {number} tau - Blend rate (0=no change, 1=full copy)
     */
    polyakUpdate(source, tau) {
        const oneMinusTau = 1 - tau;
        for (let l = 0; l < this.numLayers - 1; l++) {
            const tw = this.weights[l];
            const sw = source.weights[l];
            for (let i = 0; i < tw.length; i++) tw[i] = oneMinusTau * tw[i] + tau * sw[i];
            const tb = this.biases[l];
            const sb = source.biases[l];
            for (let i = 0; i < tb.length; i++) tb[i] = oneMinusTau * tb[i] + tau * sb[i];
        }
    }

    /**
     * Serialize weights to a plain object (for JSON save/load).
     * @returns {{ weights: number[][], biases: number[][] }}
     */
    serialize() {
        return {
            layerSizes: [...this.layerSizes],
            weights: this.weights.map(w => Array.from(w)),
            biases: this.biases.map(b => Array.from(b)),
        };
    }

    /**
     * Load weights from a serialized object.
     * @param {{ weights: number[][], biases: number[][] }} data
     */
    deserialize(data) {
        for (let l = 0; l < this.numLayers - 1; l++) {
            this.weights[l].set(data.weights[l]);
            this.biases[l].set(data.biases[l]);
        }
    }

    /** Total number of trainable parameters. */
    get paramCount() {
        let count = 0;
        for (let l = 0; l < this.numLayers - 1; l++) {
            count += this.weights[l].length + this.biases[l].length;
        }
        return count;
    }
}

// ── Replay Buffer ───────────────────────────────────────────────────

/**
 * Circular replay buffer for experience replay.
 *
 * Stores (state, action, reward, nextState, done [, extras]) transitions.
 * State vectors are stored as Float64Arrays for memory efficiency.
 *
 * Two sampling modes:
 *   - 'uniform' (default): vanilla DQN behaviour, uniform random sampling
 *   - 'prioritized': PER (Schaul 2015) — sample with probability ∝ priority^α
 *                     where priority = |TD error| + ε. New transitions get max
 *                     priority so they are guaranteed to be replayed at least
 *                     once. Returns importance-sampling weights (β-correction)
 *                     so the agent can compensate for the biased sample.
 *
 * The PER implementation uses a flat priority array and computes the
 * cumulative sum on demand — O(n) per sample. For our scale (replay capacity
 * ~30 K, batch ~64) this is well under 1 ms per sample. A sum-tree would be
 * faster asymptotically but adds correctness risk we don't need.
 */
export class ReplayBuffer {
    /**
     * @param {number} capacity - Maximum number of transitions
     * @param {number} [seed=42] - PRNG seed for sampling
     * @param {Object} [options]
     * @param {'uniform'|'prioritized'} [options.mode='uniform']
     * @param {number} [options.perAlpha=0.6]    Priority exponent (0=uniform, 1=full prioritization)
     * @param {number} [options.perBeta0=0.4]    Initial importance-sampling exponent
     * @param {number} [options.perEps=1e-6]     Minimum priority (so 0-error transitions still sample)
     */
    constructor(capacity, seed = 42, options = {}) {
        this.capacity = capacity;
        this.buffer = [];
        this.position = 0;
        this._rng = new PRNG(seed);

        this.mode    = options.mode || 'uniform';
        this.alpha   = options.perAlpha ?? 0.6;
        this.beta    = options.perBeta0 ?? 0.4;     // anneal toward 1.0 over training
        this.eps     = options.perEps   ?? 1e-6;
        this._priorities = (this.mode === 'prioritized')
            ? new Float64Array(capacity) : null;
        this._maxPriority = 1.0;
    }

    /**
     * Add a transition to the buffer.
     * @param {Float64Array} state
     * @param {number} action - Action index
     * @param {number} reward
     * @param {Float64Array} nextState
     * @param {boolean} done
     * @param {Object} [extras] - Optional per-transition fields (e.g. effectiveGamma
     *                            for n-step returns). Stored verbatim and forwarded
     *                            to the agent's trainBatch.
     */
    push(state, action, reward, nextState, done, extras = null) {
        const transition = {
            state: new Float64Array(state),
            action,
            reward,
            nextState: new Float64Array(nextState),
            done,
        };
        if (extras) Object.assign(transition, extras);

        let writeIdx;
        if (this.buffer.length < this.capacity) {
            writeIdx = this.buffer.length;
            this.buffer.push(transition);
        } else {
            writeIdx = this.position;
            this.buffer[this.position] = transition;
        }
        this.position = (this.position + 1) % this.capacity;

        // PER: new transitions get max priority — guarantees they are sampled
        // at least once before being demoted by their actual TD error.
        if (this._priorities) {
            this._priorities[writeIdx] = this._maxPriority;
        }
    }

    /**
     * Sample a random mini-batch.
     * @param {number} batchSize
     * @returns {Array<Object>} Transitions. In prioritized mode, each transition
     *   is augmented with `_perIdx` (its index in the buffer, used by
     *   updatePriorities) and `_perWeight` (the importance-sampling weight,
     *   normalized so max weight = 1).
     */
    sample(batchSize) {
        if (this.buffer.length < batchSize) {
            throw new Error(`Cannot sample ${batchSize} from buffer of size ${this.buffer.length}`);
        }

        if (this.mode === 'prioritized') {
            return this._sampleProportional(batchSize);
        }

        const indices = new Set();
        while (indices.size < batchSize) {
            indices.add(this._rng.nextInt(this.buffer.length));
        }
        return Array.from(indices).map(i => this.buffer[i]);
    }

    /**
     * PER proportional sampling. Sums priority^α across the buffer, divides
     * the cumulative range into batchSize equal segments, samples one
     * transition per segment (stratified) for a low-variance batch.
     *
     * @private
     */
    _sampleProportional(batchSize) {
        const n = this.buffer.length;
        const pri = this._priorities;
        const alpha = this.alpha;
        // Cumulative sum of priority^α
        const cum = new Float64Array(n);
        let total = 0;
        for (let i = 0; i < n; i++) {
            const p = Math.pow(Math.max(pri[i], this.eps), alpha);
            total += p;
            cum[i] = total;
        }
        if (total <= 0) {
            // Defensive: fall back to uniform if priorities are degenerate
            const indices = new Set();
            while (indices.size < batchSize) indices.add(this._rng.nextInt(n));
            return Array.from(indices).map(i => this.buffer[i]);
        }

        // Stratified sampling: divide [0, total] into batchSize segments,
        // sample one position per segment, find via binary search on cum[].
        const seg = total / batchSize;
        const out = new Array(batchSize);

        // First pass: collect probabilities to compute IS weights.
        // Probability of sample i = priority^α / total.
        // IS weight (Schaul 2015 eq. 9) = (1 / (n · prob))^β / max_w
        let maxW = 0;
        const weights = new Float64Array(batchSize);
        const indices = new Int32Array(batchSize);
        for (let b = 0; b < batchSize; b++) {
            const target = (b + this._rng.next()) * seg;
            // Binary search for first cum[i] ≥ target
            let lo = 0, hi = n - 1;
            while (lo < hi) {
                const mid = (lo + hi) >> 1;
                if (cum[mid] < target) lo = mid + 1;
                else hi = mid;
            }
            const idx = lo;
            indices[b] = idx;
            const priAlpha = (idx === 0 ? cum[0] : cum[idx] - cum[idx - 1]);
            const prob = priAlpha / total;
            const w = Math.pow(1 / (n * Math.max(prob, 1e-12)), this.beta);
            weights[b] = w;
            if (w > maxW) maxW = w;
        }

        for (let b = 0; b < batchSize; b++) {
            const tr = this.buffer[indices[b]];
            // Augment a *shallow copy* so we don't pollute the stored transition's
            // metadata across batches (would cause wrong _perWeight on next sample).
            const sample = Object.assign({}, tr, {
                _perIdx:    indices[b],
                _perWeight: weights[b] / maxW,    // normalize: max IS weight = 1
            });
            out[b] = sample;
        }
        return out;
    }

    /**
     * Update priorities after a learning step. In prioritized mode the agent
     * calls this with the absolute TD errors observed on the sampled batch.
     * Indices come from the `_perIdx` field on the sampled transitions.
     *
     * @param {Int32Array|Array<number>} idxs
     * @param {Float64Array|Array<number>} tdErrors  Absolute TD errors
     */
    updatePriorities(idxs, tdErrors) {
        if (!this._priorities) return;
        for (let i = 0; i < idxs.length; i++) {
            const p = Math.abs(tdErrors[i]) + this.eps;
            this._priorities[idxs[i]] = p;
            if (p > this._maxPriority) this._maxPriority = p;
        }
    }

    /**
     * Anneal β toward 1.0. Schaul recommends β starts at 0.4 and reaches 1.0
     * by end of training. Caller drives this — typically called once per
     * episode or every K steps.
     */
    annealBeta(target = 1.0, fraction = 1e-4) {
        this.beta = Math.min(target, this.beta + fraction * (target - this.beta));
    }

    /** Current number of transitions stored. */
    get length() {
        return this.buffer.length;
    }

    /** Clear all transitions. */
    clear() {
        this.buffer = [];
        this.position = 0;
    }
}

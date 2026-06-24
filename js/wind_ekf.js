/**
 * wind_ekf.js — Extended Kalman Filter for stratospheric wind estimation.
 *
 * 34-state EKF: estimates u and v wind components at each of 17 altitude
 * evaluation levels. Uses in-situ GPS drift observations at the balloon's
 * current altitude to update the full wind column via vertical correlation.
 *
 * State vector: x = [u_1, v_1, u_2, v_2, ..., u_17, v_17]  (34 elements)
 *
 * Process model (prediction):
 *   x(t+dt) = A × x(t) + w(t)
 *   A = diag(exp(-dt/tau))  — first-order Markov decay toward forecast
 *   Q = process noise covariance
 *
 * Measurement model (update):
 *   z = H × x + v
 *   H selects u,v at the observed altitude level
 *   Vertical correlation propagates innovation to nearby altitudes
 *
 * Dependencies: config.js (runtime for altitude levels)
 */
import { runtime } from './config.js';

// ── Configuration ────────────────────────────────────────────────────

export const EKF_CONFIG = Object.freeze({
    /** Time constant for process model decay (seconds).
     *  Wind at unobserved altitudes decays toward the prior (forecast)
     *  with this time constant. Stratospheric winds have ~6-12h decorrelation. */
    TAU_S: 7200,    // 2 hours — conservative, lets observations persist

    /** Process noise standard deviation (m/s per sqrt(second)).
     *  Controls how fast uncertainty grows between observations. */
    PROCESS_NOISE_STD: 0.005,

    /** GPS measurement noise standard deviation (m/s). */
    MEASUREMENT_NOISE_STD: 0.3,

    /** Vertical correlation length for off-diagonal covariance (meters).
     *  Observation at one altitude informs nearby altitudes. */
    VERT_CORRELATION_M: 500,

    /** Initial uncertainty for all wind states (m/s). */
    INITIAL_UNCERTAINTY_STD: 8.0,

    /** Minimum uncertainty floor (m/s) — prevents overconfidence. */
    MIN_UNCERTAINTY_STD: 0.2,

    /** Maximum uncertainty ceiling (m/s). */
    MAX_UNCERTAINTY_STD: 15.0,
});

// ── Matrix utilities (34×34, pure arrays) ────────────────────────────

/**
 * Create an N×N identity matrix as a flat Float64Array.
 */
function eye(n) {
    const m = new Float64Array(n * n);
    for (let i = 0; i < n; i++) m[i * n + i] = 1;
    return m;
}

/**
 * Create an N×N diagonal matrix from a length-N vector.
 */
function diag(vec) {
    const n = vec.length;
    const m = new Float64Array(n * n);
    for (let i = 0; i < n; i++) m[i * n + i] = vec[i];
    return m;
}

/**
 * Matrix × matrix: C = A × B (both n×n).
 */
function matMul(A, B, n) {
    const C = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
    return C;
}

/**
 * Matrix + Matrix (element-wise, in-place: A += B).
 */
function matAdd(A, B, n) {
    for (let i = 0; i < n * n; i++) A[i] += B[i];
    return A;
}

/**
 * Matrix × vector: y = M × x.
 */
function matVecMul(M, x, n) {
    const y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
        let sum = 0;
        for (let j = 0; j < n; j++) sum += M[i * n + j] * x[j];
        y[i] = sum;
    }
    return y;
}

/**
 * Transpose an n×n matrix.
 */
function transpose(M, n) {
    const T = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            T[j * n + i] = M[i * n + j];
        }
    }
    return T;
}

/**
 * Scale a matrix: M *= scalar (in-place).
 */
function matScale(M, s, n) {
    for (let i = 0; i < n * n; i++) M[i] *= s;
    return M;
}

/**
 * Invert a 2×2 matrix (for the innovation covariance).
 * Returns null if singular.
 */
function inv2x2(m) {
    const [a, b, c, d] = m;
    const det = a * d - b * c;
    if (Math.abs(det) < 1e-15) return null;
    const invDet = 1 / det;
    return new Float64Array([d * invDet, -b * invDet, -c * invDet, a * invDet]);
}

// ── WindEKF ──────────────────────────────────────────────────────────

/**
 * Extended Kalman Filter for wind state estimation.
 *
 * Usage:
 *   const ekf = new WindEKF();
 *   ekf.initialize(forecastWindFn);  // optional: seed from forecast
 *   // Each physics step:
 *   ekf.predict(dt_s);
 *   // When observation available:
 *   ekf.update(alt_m, u_obs, v_obs);
 *   // Query:
 *   const {u, v} = ekf.getWind(alt_m);
 *   const sigma = ekf.getUncertainty(alt_m);
 */
export class WindEKF {
    constructor(config = EKF_CONFIG) {
        this.config = config;
        this.N = 0;          // Number of altitude levels (set on init)
        this.dim = 0;        // State dimension (2 × N)
        this.altitudes = []; // Altitude levels
        this.x = null;       // State vector [u1, v1, u2, v2, ...]
        this.P = null;       // Covariance matrix (dim × dim, flat)
        this.x_prior = null; // Prior (forecast) state — used as decay target
        this._initialized = false;
    }

    /**
     * Initialize the EKF with altitude levels and optional forecast.
     *
     * @param {Function|null} forecastFn — Optional: (alt_m) → {u, v}
     *   If provided, seeds the state from the forecast. Otherwise, starts at zero.
     */
    initialize(forecastFn = null) {
        this.altitudes = [...runtime.altitudeLevels];
        this.N = this.altitudes.length;
        this.dim = 2 * this.N;

        // Initialize state vector
        this.x = new Float64Array(this.dim);
        this.x_prior = new Float64Array(this.dim);

        if (forecastFn) {
            for (let i = 0; i < this.N; i++) {
                const w = forecastFn(this.altitudes[i]);
                this.x[2 * i] = w.u;
                this.x[2 * i + 1] = w.v;
                this.x_prior[2 * i] = w.u;
                this.x_prior[2 * i + 1] = w.v;
            }
        }

        // Initialize covariance: diagonal with initial uncertainty
        const sigmaInit = this.config.INITIAL_UNCERTAINTY_STD;
        this.P = diag(new Float64Array(this.dim).fill(sigmaInit * sigmaInit));

        // Add off-diagonal vertical correlation to initial P
        this._addVerticalCorrelation(this.P);

        this._initialized = true;
    }

    /**
     * Reset to uninitialized state.
     */
    reset() {
        this.x = null;
        this.P = null;
        this.x_prior = null;
        this._initialized = false;
    }

    /** Whether the EKF has been initialized. */
    get initialized() {
        return this._initialized;
    }

    /**
     * Prediction step: advance the state forward by dt_s seconds.
     *
     * Process model: x(t+dt) = A × (x(t) - x_prior) + x_prior + w(t)
     * The state decays toward the prior (forecast) with time constant TAU_S.
     * Covariance grows: P = A × P × A' + Q
     *
     * @param {number} dt_s — Time step in seconds
     */
    predict(dt_s) {
        if (!this._initialized) return;

        const n = this.dim;
        const decay = Math.exp(-dt_s / this.config.TAU_S);

        // State prediction: decay toward prior
        for (let i = 0; i < n; i++) {
            this.x[i] = this.x_prior[i] + decay * (this.x[i] - this.x_prior[i]);
        }

        // Covariance prediction: P = decay² × P + Q
        // (A = decay × I, so A P A' = decay² P)
        const decay2 = decay * decay;
        const processVar = this.config.PROCESS_NOISE_STD * this.config.PROCESS_NOISE_STD * dt_s;

        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                this.P[i * n + j] *= decay2;
            }
            // Add process noise on diagonal
            this.P[i * n + i] += processVar;
        }

        // Clamp covariance diagonal
        this._clampCovariance();
    }

    /**
     * Update step: incorporate an in-situ wind observation.
     *
     * The observation at the balloon's current altitude is used to update
     * the full state vector via vertical correlation in the Kalman gain.
     *
     * @param {number} obsAlt_m — Altitude of observation
     * @param {number} u_obs — Observed u component (m/s)
     * @param {number} v_obs — Observed v component (m/s)
     * @param {number} noiseStd — Observation noise (default: GPS noise)
     */
    update(obsAlt_m, u_obs, v_obs, noiseStd = this.config.MEASUREMENT_NOISE_STD) {
        if (!this._initialized) return;

        const n = this.dim;
        const L = this.config.VERT_CORRELATION_M;
        const R_var = noiseStd * noiseStd;

        // Build observation matrix H (2 × dim):
        // H maps the full state to the observed (u, v) at obsAlt_m.
        // Uses Gaussian-weighted contribution from all altitude levels.
        const H = new Float64Array(2 * n);  // 2 rows × n cols

        for (let i = 0; i < this.N; i++) {
            const dAlt = obsAlt_m - this.altitudes[i];
            const w = Math.exp(-(dAlt * dAlt) / (2 * L * L));
            H[0 * n + 2 * i] = w;       // u component
            H[1 * n + 2 * i + 1] = w;   // v component
        }

        // Normalize H rows so they sum to 1 (proper interpolation)
        let sumU = 0, sumV = 0;
        for (let i = 0; i < this.N; i++) {
            sumU += H[0 * n + 2 * i];
            sumV += H[1 * n + 2 * i + 1];
        }
        if (sumU > 1e-10) {
            for (let i = 0; i < this.N; i++) H[0 * n + 2 * i] /= sumU;
        }
        if (sumV > 1e-10) {
            for (let i = 0; i < this.N; i++) H[1 * n + 2 * i + 1] /= sumV;
        }

        // Innovation: y = z - H × x
        const z = new Float64Array([u_obs, v_obs]);
        const Hx = new Float64Array(2);
        for (let j = 0; j < n; j++) {
            Hx[0] += H[0 * n + j] * this.x[j];
            Hx[1] += H[1 * n + j] * this.x[j];
        }
        const y = new Float64Array([z[0] - Hx[0], z[1] - Hx[1]]);

        // Innovation covariance: S = H × P × H' + R (2×2)
        // S = H P H' + R
        const HP = new Float64Array(2 * n);  // 2 × n
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += H[i * n + k] * this.P[k * n + j];
                }
                HP[i * n + j] = sum;
            }
        }

        const S = new Float64Array(4);  // 2×2
        for (let i = 0; i < 2; i++) {
            for (let j = 0; j < 2; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += HP[i * n + k] * H[j * n + k];  // H' = H transposed in k
                }
                S[i * 2 + j] = sum;
            }
        }
        S[0] += R_var;
        S[3] += R_var;

        // Invert S (2×2)
        const Sinv = inv2x2(S);
        if (!Sinv) return;  // singular, skip update

        // Kalman gain: K = P × H' × S⁻¹  (n × 2)
        // First: PH' (n × 2)
        const PHt = new Float64Array(n * 2);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < 2; j++) {
                let sum = 0;
                for (let k = 0; k < n; k++) {
                    sum += this.P[i * n + k] * H[j * n + k];
                }
                PHt[i * 2 + j] = sum;
            }
        }

        // K = PH' × Sinv  (n × 2)
        const K = new Float64Array(n * 2);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < 2; j++) {
                K[i * 2 + j] = PHt[i * 2 + 0] * Sinv[0 * 2 + j]
                              + PHt[i * 2 + 1] * Sinv[1 * 2 + j];
            }
        }

        // State update: x = x + K × y
        for (let i = 0; i < n; i++) {
            this.x[i] += K[i * 2 + 0] * y[0] + K[i * 2 + 1] * y[1];
        }

        // Covariance update: P = (I - K × H) × P
        // Compute KH (n × n)
        const KH = new Float64Array(n * n);
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                KH[i * n + j] = K[i * 2 + 0] * H[0 * n + j]
                              + K[i * 2 + 1] * H[1 * n + j];
            }
        }

        // I_KH = I - KH
        const I_KH = eye(n);
        for (let i = 0; i < n * n; i++) I_KH[i] -= KH[i];

        // P = I_KH × P  (Joseph form would be better for numerical stability,
        // but for 34×34 this is fine)
        this.P = matMul(I_KH, this.P, n);

        // Symmetrize P (numerical stability)
        for (let i = 0; i < n; i++) {
            for (let j = i + 1; j < n; j++) {
                const avg = (this.P[i * n + j] + this.P[j * n + i]) / 2;
                this.P[i * n + j] = avg;
                this.P[j * n + i] = avg;
            }
        }

        this._clampCovariance();
    }

    /**
     * Assimilate a forecast packet as a "soft observation".
     * Forecast observations have higher noise than in-situ (lower confidence).
     *
     * @param {Array<{alt_m: number, u: number, v: number, sigma?: number}>} levels
     */
    assimilateForecast(levels) {
        for (const lvl of levels) {
            const sigma = lvl.sigma || 3.0;  // forecast uncertainty ~3 m/s
            this.update(lvl.alt_m, lvl.u, lvl.v, sigma);
        }
        // Also update the prior (the decay target)
        for (const lvl of levels) {
            const idx = this._nearestLevelIndex(lvl.alt_m);
            if (idx >= 0) {
                this.x_prior[2 * idx] = lvl.u;
                this.x_prior[2 * idx + 1] = lvl.v;
            }
        }
    }

    /**
     * Get the EKF's best wind estimate at a given altitude.
     * Uses linear interpolation between the two nearest altitude levels.
     *
     * @param {number} alt_m
     * @returns {{ u: number, v: number }}
     */
    getWind(alt_m) {
        if (!this._initialized) return { u: 0, v: 0 };

        // Find bracketing levels
        const alts = this.altitudes;
        if (alt_m <= alts[0]) {
            return { u: this.x[0], v: this.x[1] };
        }
        if (alt_m >= alts[alts.length - 1]) {
            return { u: this.x[this.dim - 2], v: this.x[this.dim - 1] };
        }

        for (let i = 0; i < alts.length - 1; i++) {
            if (alt_m >= alts[i] && alt_m <= alts[i + 1]) {
                const t = (alt_m - alts[i]) / (alts[i + 1] - alts[i]);
                return {
                    u: this.x[2 * i] * (1 - t) + this.x[2 * (i + 1)] * t,
                    v: this.x[2 * i + 1] * (1 - t) + this.x[2 * (i + 1) + 1] * t,
                };
            }
        }
        return { u: 0, v: 0 };
    }

    /**
     * Get the uncertainty (standard deviation) at a given altitude.
     * Returns the average of u and v uncertainties.
     *
     * @param {number} alt_m
     * @returns {number} — Standard deviation in m/s
     */
    getUncertainty(alt_m) {
        if (!this._initialized) return this.config.INITIAL_UNCERTAINTY_STD;

        const idx = this._nearestLevelIndex(alt_m);
        if (idx < 0) return this.config.MAX_UNCERTAINTY_STD;

        const varU = this.P[(2 * idx) * this.dim + (2 * idx)];
        const varV = this.P[(2 * idx + 1) * this.dim + (2 * idx + 1)];
        return Math.sqrt((varU + varV) / 2);
    }

    /**
     * Get wind estimates and uncertainties at all altitude levels.
     *
     * @returns {Array<{alt_m: number, u: number, v: number, sigma: number}>}
     */
    getFullColumn() {
        if (!this._initialized) return [];
        return this.altitudes.map((alt_m, i) => ({
            alt_m,
            u: this.x[2 * i],
            v: this.x[2 * i + 1],
            sigma: this.getUncertainty(alt_m),
        }));
    }

    // ── Private helpers ──────────────────────────────────────────────

    _nearestLevelIndex(alt_m) {
        let best = -1, bestDist = Infinity;
        for (let i = 0; i < this.N; i++) {
            const d = Math.abs(this.altitudes[i] - alt_m);
            if (d < bestDist) { bestDist = d; best = i; }
        }
        return best;
    }

    _addVerticalCorrelation(P) {
        const L = this.config.VERT_CORRELATION_M;
        const L2 = 2 * L * L;
        const n = this.dim;

        for (let i = 0; i < this.N; i++) {
            for (let j = i + 1; j < this.N; j++) {
                const dAlt = this.altitudes[i] - this.altitudes[j];
                const corr = Math.exp(-(dAlt * dAlt) / L2);

                // u-u correlation
                const varI_u = P[(2 * i) * n + (2 * i)];
                const varJ_u = P[(2 * j) * n + (2 * j)];
                const cov_u = corr * Math.sqrt(varI_u * varJ_u);
                P[(2 * i) * n + (2 * j)] = cov_u;
                P[(2 * j) * n + (2 * i)] = cov_u;

                // v-v correlation
                const varI_v = P[(2 * i + 1) * n + (2 * i + 1)];
                const varJ_v = P[(2 * j + 1) * n + (2 * j + 1)];
                const cov_v = corr * Math.sqrt(varI_v * varJ_v);
                P[(2 * i + 1) * n + (2 * j + 1)] = cov_v;
                P[(2 * j + 1) * n + (2 * i + 1)] = cov_v;
            }
        }
    }

    _clampCovariance() {
        const n = this.dim;
        const minVar = this.config.MIN_UNCERTAINTY_STD ** 2;
        const maxVar = this.config.MAX_UNCERTAINTY_STD ** 2;
        for (let i = 0; i < n; i++) {
            this.P[i * n + i] = Math.max(minVar, Math.min(maxVar, this.P[i * n + i]));
        }
    }
}

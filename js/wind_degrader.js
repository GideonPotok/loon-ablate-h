/**
 * wind_degrader.js — Forecast degradation model (Q1.5).
 *
 * Simulates realistic forecast errors so the EKF (Q1.2) can demonstrate
 * value by correcting degraded forecasts using in-situ observations.
 *
 * Without degradation, the navigator sees the exact same wind that drives
 * the physics — the EKF can never improve on perfection. With degradation,
 * the navigator's "forecast" diverges from truth over time, and the EKF
 * pulls it back toward reality using GPS-derived wind measurements.
 *
 * Degradation model:
 *   forecast(alt, t) = truth(alt, t_issue) + bias(alt) + noise(alt, t)
 *
 * Components:
 *   1. Temporal staleness: forecast is frozen at issuance time (no temporal
 *      variation), while truth evolves. Gap grows naturally with time.
 *   2. Altitude-dependent bias: systematic model error at each altitude,
 *      drawn from seeded PRNG at initialization. Represents NWP model bias.
 *   3. Stochastic noise: time-varying perturbation that changes every
 *      NOISE_UPDATE_S seconds. Represents irreducible forecast uncertainty.
 *
 * Feature flag: runtime.features.forecastDegradation (default: false)
 * Config knobs: DEGRADER_* in config or passed as options.
 */

/**
 * Deterministic PRNG (xorshift32) for reproducible degradation.
 * Avoids touching global Math.random state.
 */
class PRNG {
    constructor(seed = 42) {
        this.state = seed >>> 0 || 1;
    }

    /** Returns float in [0, 1) */
    next() {
        let x = this.state;
        x ^= x << 13;
        x ^= x >>> 17;
        x ^= x << 5;
        this.state = x >>> 0;
        return (this.state & 0x7FFFFFFF) / 0x80000000;
    }

    /** Returns normally distributed value (Box-Muller) */
    nextGaussian() {
        const u1 = this.next() || 1e-10;
        const u2 = this.next();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

// ── Default degradation parameters ──────────────────────────────────

export const DEGRADER_DEFAULTS = Object.freeze({
    BIAS_SIGMA:         0.71,     // m/s std dev of altitude-dependent bias [calibrated 2026-05-05 from IGRA×ERA5 cross-validation: 15 stations spanning 1°N–47°N, 138,089 collocated obs, 2023-2024 — see tactical/output/cross_validation_0to50n/calibration.txt]
    BIAS_VERT_CORR_M:   1000,    // Vertical correlation length for bias (m)
    NOISE_SIGMA:        2.93,     // m/s std dev of time-varying noise [calibrated 2026-05-05; median noise std at 100-70 hPa — robust statistic, unchanged across 7-, 11-, and 15-station calibrations]
    NOISE_UPDATE_S:     1800,     // Noise changes every 30 minutes
    NOISE_VERT_CORR_M:  750,     // Vertical correlation for noise
    STALENESS_MODE:     'frozen', // 'frozen' = no temporal variation in forecast
                                  // 'lagged' = forecast uses t - lag_s
    LAG_S:              10800,    // 3-hour lag (only used in 'lagged' mode)
    SEED:               12345,    // PRNG seed for reproducibility
});

// ── Altitude grid (matches navigator's 17 levels) ──────────────────

const ALT_MIN = 16500;
const ALT_MAX = 18500;
const ALT_STEP = 125;
const NUM_LEVELS = 17;  // (18500 - 16500) / 125 + 1

function altIndex(alt_m) {
    const idx = Math.floor((alt_m - ALT_MIN) / ALT_STEP);
    return Math.max(0, Math.min(NUM_LEVELS - 2, idx));  // -2 so idx+1 is always valid
}

function altForIndex(i) {
    return ALT_MIN + i * ALT_STEP;
}

// ── ForecastDegrader ────────────────────────────────────────────────

export class ForecastDegrader {
    /**
     * @param {Function} truthWindFn - (alt_m, time_s) → { u, v } — the true wind
     * @param {Function} baseWindFn  - (alt_m) → { u, v } — base wind without temporal variation
     *                                  (null = use truthWindFn at t=0 as snapshot)
     * @param {Object} options       - Override DEGRADER_DEFAULTS
     */
    constructor(truthWindFn, baseWindFn = null, options = {}) {
        this.truthFn = truthWindFn;
        this.baseFn = baseWindFn;
        this.opts = { ...DEGRADER_DEFAULTS, ...options };

        this.rng = new PRNG(this.opts.SEED);

        // Generate altitude-dependent bias with vertical correlation
        this.biasU = new Float64Array(NUM_LEVELS);
        this.biasV = new Float64Array(NUM_LEVELS);
        this._generateCorrelatedBias();

        // Noise cache: regenerated every NOISE_UPDATE_S
        this.noiseU = new Float64Array(NUM_LEVELS);
        this.noiseV = new Float64Array(NUM_LEVELS);
        this._noiseSlot = -1;

        // Snapshot: frozen forecast at t=0
        this.snapshot = null;
        if (baseWindFn) {
            this._takeSnapshot();
        }
    }

    /**
     * Take a frozen snapshot of the base wind at all altitude levels.
     * This represents the forecast issued at t=0.
     */
    _takeSnapshot() {
        this.snapshot = [];
        for (let i = 0; i < NUM_LEVELS; i++) {
            const alt = altForIndex(i);
            const w = this.baseFn(alt);
            this.snapshot.push({ u: w.u, v: w.v });
        }
    }

    /**
     * Generate vertically-correlated bias using Cholesky-like approach.
     * Adjacent altitude levels share correlated bias; distant levels are independent.
     */
    _generateCorrelatedBias() {
        const sigma = this.opts.BIAS_SIGMA;
        const L = this.opts.BIAS_VERT_CORR_M;

        // Generate independent samples
        const rawU = new Float64Array(NUM_LEVELS);
        const rawV = new Float64Array(NUM_LEVELS);
        for (let i = 0; i < NUM_LEVELS; i++) {
            rawU[i] = this.rng.nextGaussian() * sigma;
            rawV[i] = this.rng.nextGaussian() * sigma;
        }

        // Apply vertical smoothing (Gaussian kernel)
        for (let i = 0; i < NUM_LEVELS; i++) {
            let sumU = 0, sumV = 0, wTotal = 0;
            for (let j = 0; j < NUM_LEVELS; j++) {
                const dAlt = (i - j) * ALT_STEP;
                const w = Math.exp(-(dAlt * dAlt) / (2 * L * L));
                sumU += rawU[j] * w;
                sumV += rawV[j] * w;
                wTotal += w;
            }
            this.biasU[i] = sumU / wTotal;
            this.biasV[i] = sumV / wTotal;
        }
    }

    /**
     * Generate time-varying noise for a given time slot.
     * Changes every NOISE_UPDATE_S seconds with vertical correlation.
     */
    _updateNoise(time_s) {
        const slot = Math.floor(time_s / this.opts.NOISE_UPDATE_S);
        if (slot === this._noiseSlot) return;
        this._noiseSlot = slot;

        const sigma = this.opts.NOISE_SIGMA;
        const L = this.opts.NOISE_VERT_CORR_M;

        // Seed PRNG deterministically from slot
        const slotRng = new PRNG(this.opts.SEED * 7 + slot * 104729);

        const rawU = new Float64Array(NUM_LEVELS);
        const rawV = new Float64Array(NUM_LEVELS);
        for (let i = 0; i < NUM_LEVELS; i++) {
            rawU[i] = slotRng.nextGaussian() * sigma;
            rawV[i] = slotRng.nextGaussian() * sigma;
        }

        // Vertical smoothing
        for (let i = 0; i < NUM_LEVELS; i++) {
            let sumU = 0, sumV = 0, wTotal = 0;
            for (let j = 0; j < NUM_LEVELS; j++) {
                const dAlt = (i - j) * ALT_STEP;
                const w = Math.exp(-(dAlt * dAlt) / (2 * L * L));
                sumU += rawU[j] * w;
                sumV += rawV[j] * w;
                wTotal += w;
            }
            this.noiseU[i] = sumU / wTotal;
            this.noiseV[i] = sumV / wTotal;
        }
    }

    /**
     * Get the degraded forecast wind at (alt_m, time_s).
     *
     * When snapshot is available (synthetic presets):
     *   forecast = snapshot(alt) + bias(alt) + noise(alt, t)
     *   (snapshot is frozen at t=0, missing temporal variation)
     *
     * When no snapshot (real GFS data):
     *   forecast = truth(alt, t - lag) + bias(alt) + noise(alt, t)
     *   (uses lagged truth to simulate forecast staleness)
     *
     * @returns {{ u: number, v: number }}
     */
    getForecastWind(alt_m, time_s) {
        this._updateNoise(time_s);

        const idx = altIndex(alt_m);
        const frac = (alt_m - altForIndex(idx)) / ALT_STEP;

        // Interpolate bias and noise between grid levels
        const idx2 = Math.min(idx + 1, NUM_LEVELS - 1);
        const t = Math.max(0, Math.min(1, frac));

        const biasU = this.biasU[idx] * (1 - t) + this.biasU[idx2] * t;
        const biasV = this.biasV[idx] * (1 - t) + this.biasV[idx2] * t;
        const noiseU = this.noiseU[idx] * (1 - t) + this.noiseU[idx2] * t;
        const noiseV = this.noiseV[idx] * (1 - t) + this.noiseV[idx2] * t;

        // Base wind: frozen snapshot or lagged truth
        let baseU, baseV;
        if (this.snapshot) {
            // Synthetic mode: frozen at t=0
            baseU = this.snapshot[idx].u * (1 - t) + this.snapshot[idx2].u * t;
            baseV = this.snapshot[idx].v * (1 - t) + this.snapshot[idx2].v * t;
        } else {
            // Real data mode: lagged
            const laggedT = Math.max(0, time_s - this.opts.LAG_S);
            const w = this.truthFn(alt_m, laggedT);
            baseU = w.u;
            baseV = w.v;
        }

        return {
            u: baseU + biasU + noiseU,
            v: baseV + biasV + noiseV,
        };
    }

    /**
     * Get the forecast uncertainty (sigma) at an altitude.
     * Returns the expected RMS error of the degraded forecast.
     * Useful for initializing the EKF's raw sigma parameter.
     */
    getUncertainty(alt_m) {
        const idx = altIndex(alt_m);
        const bSigma = Math.sqrt(this.biasU[idx] ** 2 + this.biasV[idx] ** 2) / Math.SQRT2;
        const nSigma = this.opts.NOISE_SIGMA;
        // RMS of bias + noise + temporal staleness (~1-2 m/s from temporal variation)
        const stalenessSigma = 1.5;  // approximate from temporal variation amplitude
        return Math.sqrt(bSigma ** 2 + nSigma ** 2 + stalenessSigma ** 2);
    }

    /**
     * Reset with new seed (for re-randomization between runs).
     */
    reseed(seed) {
        this.rng = new PRNG(seed);
        this._generateCorrelatedBias();
        this._noiseSlot = -1;
        if (this.baseFn) this._takeSnapshot();
    }

    /**
     * Get diagnostic info about degradation at each altitude level.
     */
    getDiagnostics(time_s) {
        this._updateNoise(time_s);
        const levels = [];
        for (let i = 0; i < NUM_LEVELS; i++) {
            levels.push({
                alt_m: altForIndex(i),
                biasU: this.biasU[i],
                biasV: this.biasV[i],
                noiseU: this.noiseU[i],
                noiseV: this.noiseV[i],
                totalBias: Math.sqrt(this.biasU[i] ** 2 + this.biasV[i] ** 2),
            });
        }
        return levels;
    }
}

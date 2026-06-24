/**
 * wind_observer.js — In-situ wind observation from GPS drift.
 *
 * The balloon continuously measures wind at its current altitude by
 * differencing successive GPS positions. This module:
 *   1. Derives wind (u, v) from position deltas (GPS drift method)
 *   2. Stores observations in a time-decaying ring buffer
 *   3. Provides quality-weighted wind estimates at any altitude
 *
 * The observation store is the foundation for all Bayesian wind
 * estimation (EKF, EnKF) in later roadmap stages.
 *
 * Dependencies: config.js (runtime constants)
 */
import { EARTH_RADIUS_M, runtime } from './config.js';

// ── Configuration ────────────────────────────────────────────────────

export const OBSERVER_CONFIG = Object.freeze({
    /** Maximum observations stored (ring buffer). */
    MAX_OBSERVATIONS: 2000,

    /** Time constant for exponential quality decay (seconds).
     *  After TAU_S seconds, an observation has quality = 1/e ≈ 0.37. */
    TAU_S: 3600,

    /** Minimum quality threshold — observations below this are ignored. */
    MIN_QUALITY: 0.05,

    /** Minimum time between successive observations (seconds).
     *  Prevents storing redundant data at every physics step. */
    MIN_INTERVAL_S: 30,

    /** Altitude bin width for binned retrieval (meters). */
    ALT_BIN_M: 250,

    /** Vertical correlation length for interpolation (meters).
     *  Observations at altitude A inform estimates at A ± L with
     *  Gaussian weighting: w = exp(-(dAlt)² / (2 * L²)). */
    VERT_CORRELATION_M: 500,

    /** GPS velocity noise standard deviation (m/s).
     *  Typical for u-blox ZED-F9P at 10Hz, after smoothing to 1Hz. */
    GPS_NOISE_STD: 0.3,
});

// ── WindObservation data type ────────────────────────────────────────

/**
 * A single wind observation derived from GPS drift.
 * @typedef {Object} WindObservation
 * @property {number} alt_m — Altitude at which wind was observed
 * @property {number} time_s — Simulation time of observation
 * @property {number} u_obs — Observed eastward wind component (m/s)
 * @property {number} v_obs — Observed northward wind component (m/s)
 * @property {number} noise_std — Estimated measurement noise (m/s)
 */

// ── WindObservationStore ─────────────────────────────────────────────

/**
 * Ring buffer of wind observations with time-decaying quality.
 *
 * Observations are derived from GPS position differences:
 *   u = dLon/dt × R_earth × cos(lat)
 *   v = dLat/dt × R_earth
 *
 * Quality decays exponentially: quality(t) = exp(-(t_now - t_obs) / tau)
 *
 * Usage:
 *   const store = new WindObservationStore();
 *   // Each physics step, feed current balloon state:
 *   store.observe(state, prevState, dt_s);
 *   // Query best wind estimate at any altitude:
 *   const est = store.getEstimate(17500, currentTime);
 */
export class WindObservationStore {
    constructor(config = OBSERVER_CONFIG) {
        this.config = config;
        this.observations = [];      // WindObservation[]
        this._head = 0;              // Next write position in ring buffer
        this._lastObsTime = -Infinity;
    }

    /** Number of stored observations. */
    get size() {
        return this.observations.length;
    }

    /** Clear all observations. */
    reset() {
        this.observations = [];
        this._head = 0;
        this._lastObsTime = -Infinity;
    }

    /**
     * Derive wind from two successive balloon states and store the observation.
     *
     * @param {Object} state — Current balloon state { lat, lon, alt_m }
     * @param {Object} prevState — Previous balloon state { lat, lon, alt_m }
     * @param {number} dt_s — Time elapsed between states (seconds)
     * @param {number} time_s — Current simulation time
     * @returns {WindObservation|null} — The observation, or null if skipped
     */
    observe(state, prevState, dt_s, time_s) {
        // Rate-limit: don't store faster than MIN_INTERVAL_S
        if (time_s - this._lastObsTime < this.config.MIN_INTERVAL_S) {
            return null;
        }

        if (dt_s <= 0) return null;

        // GPS drift → wind estimation
        const dLat_deg = state.lat - prevState.lat;
        const dLon_deg = state.lon - prevState.lon;
        const cosLat = Math.cos(state.lat * Math.PI / 180);

        const v_obs = (dLat_deg * Math.PI / 180) * EARTH_RADIUS_M / dt_s;
        const u_obs = (dLon_deg * Math.PI / 180) * EARTH_RADIUS_M * cosLat / dt_s;

        const obs = {
            alt_m: (state.alt_m + prevState.alt_m) / 2,  // midpoint altitude
            time_s,
            u_obs,
            v_obs,
            noise_std: this.config.GPS_NOISE_STD,
        };

        // Store in ring buffer
        if (this.observations.length < this.config.MAX_OBSERVATIONS) {
            this.observations.push(obs);
        } else {
            this.observations[this._head] = obs;
        }
        this._head = (this._head + 1) % this.config.MAX_OBSERVATIONS;
        this._lastObsTime = time_s;

        return obs;
    }

    /**
     * Compute the quality of an observation at a given time.
     * Quality decays exponentially with age.
     *
     * @param {WindObservation} obs
     * @param {number} currentTime_s
     * @returns {number} — Quality in [0, 1]
     */
    quality(obs, currentTime_s) {
        const age = currentTime_s - obs.time_s;
        if (age < 0) return 0;
        return Math.exp(-age / this.config.TAU_S);
    }

    /**
     * Get a quality-weighted wind estimate at a target altitude.
     *
     * Uses Gaussian vertical correlation: nearby-altitude observations
     * contribute with weight proportional to exp(-dAlt² / (2L²)).
     *
     * @param {number} alt_m — Target altitude
     * @param {number} currentTime_s — Current time (for decay computation)
     * @returns {{ u: number, v: number, quality: number, n: number } | null}
     *   Weighted wind estimate, overall quality score, and observation count.
     *   Returns null if no valid observations exist.
     */
    getEstimate(alt_m, currentTime_s) {
        const L = this.config.VERT_CORRELATION_M;
        const L2 = 2 * L * L;
        const minQ = this.config.MIN_QUALITY;

        let sumU = 0, sumV = 0, sumW = 0;
        let count = 0;

        for (const obs of this.observations) {
            const timeQuality = this.quality(obs, currentTime_s);
            if (timeQuality < minQ) continue;

            const dAlt = alt_m - obs.alt_m;
            const altWeight = Math.exp(-(dAlt * dAlt) / L2);

            // Combined weight: time quality × altitude proximity
            const w = timeQuality * altWeight;
            if (w < minQ * 0.1) continue;  // skip negligible contributions

            sumU += obs.u_obs * w;
            sumV += obs.v_obs * w;
            sumW += w;
            count++;
        }

        if (sumW < 1e-10) return null;

        return {
            u: sumU / sumW,
            v: sumV / sumW,
            quality: Math.min(1, sumW),  // capped at 1.0
            n: count,
        };
    }

    /**
     * Get wind estimates at all navigator altitude levels.
     * Returns an array of estimates (one per altitude level), with null
     * entries where no data is available.
     *
     * @param {number} currentTime_s
     * @returns {Array<{ alt_m: number, u: number, v: number, quality: number } | null>}
     */
    getColumnEstimates(currentTime_s) {
        return runtime.altitudeLevels.map(alt_m => {
            const est = this.getEstimate(alt_m, currentTime_s);
            if (!est) return null;
            return { alt_m, ...est };
        });
    }

    /**
     * Get the uncertainty (standard deviation) of the wind estimate at an altitude.
     * Combines GPS measurement noise with information decay.
     *
     * Lower quality → higher uncertainty.
     *
     * @param {number} alt_m — Target altitude
     * @param {number} currentTime_s
     * @returns {number} — Estimated standard deviation in m/s
     */
    getUncertainty(alt_m, currentTime_s) {
        const est = this.getEstimate(alt_m, currentTime_s);
        if (!est || est.quality < 0.01) {
            // No data: return maximum uncertainty (prior)
            return 10.0;  // m/s — typical stratospheric wind variability
        }

        // Uncertainty decreases with observation quality:
        // sigma = GPS_noise / sqrt(quality × n_effective)
        // where n_effective is bounded by quality (stale obs contribute less)
        const nEff = Math.max(1, est.quality * est.n);
        const baseSigma = this.config.GPS_NOISE_STD / Math.sqrt(nEff);

        // Floor at GPS noise, ceiling at prior
        return Math.max(this.config.GPS_NOISE_STD, Math.min(10.0, baseSigma / est.quality));
    }

    /**
     * Get all observations within an altitude band, sorted by time (newest first).
     * Useful for debugging and visualization.
     *
     * @param {number} altMin
     * @param {number} altMax
     * @param {number} currentTime_s
     * @returns {Array<WindObservation & { quality: number }>}
     */
    getObservationsInBand(altMin, altMax, currentTime_s) {
        return this.observations
            .filter(obs => obs.alt_m >= altMin && obs.alt_m <= altMax)
            .map(obs => ({ ...obs, quality: this.quality(obs, currentTime_s) }))
            .filter(obs => obs.quality >= this.config.MIN_QUALITY)
            .sort((a, b) => b.time_s - a.time_s);
    }

    /**
     * Summary statistics for debugging.
     * @param {number} currentTime_s
     * @returns {{ total: number, valid: number, altRange: [number, number], avgQuality: number }}
     */
    summary(currentTime_s) {
        let valid = 0, sumQ = 0;
        let minAlt = Infinity, maxAlt = -Infinity;

        for (const obs of this.observations) {
            const q = this.quality(obs, currentTime_s);
            if (q >= this.config.MIN_QUALITY) {
                valid++;
                sumQ += q;
                minAlt = Math.min(minAlt, obs.alt_m);
                maxAlt = Math.max(maxAlt, obs.alt_m);
            }
        }

        return {
            total: this.observations.length,
            valid,
            altRange: valid > 0 ? [minAlt, maxAlt] : [0, 0],
            avgQuality: valid > 0 ? sumQ / valid : 0,
        };
    }
}

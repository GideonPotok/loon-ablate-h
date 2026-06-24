/**
 * wind.js — Wind field model with multiple data sources.
 *
 * Supports: synthetic presets, real GFS/ERA5 data, manual profiles.
 * All sources produce a common interface: getWind(alt_m, time_s) → { u, v }
 *
 * Temporal variation layers (diurnal, IGW, planetary wave, noise) are
 * applied on top of any base wind source.
 */
import { runtime } from './config.js';

// ── Synthetic wind presets ──────────────────────────────────────────
export const WIND_PRESETS = {
    tropical: {
        name: 'Tropical Pacific',
        layers: [
            { alt_min: 15000, alt_max: 16500, u: -8,   v: -3   },
            { alt_min: 16500, alt_max: 17500, u:  7,   v:  7   },
            { alt_min: 17500, alt_max: 18500, u: -9.9, v: -7   },
            { alt_min: 18500, alt_max: 20000, u:  5,   v:  3   },
            { alt_min: 20000, alt_max: 22000, u: -6,   v: -4   },
        ],
    },
    'strong-shear': {
        name: 'Strong Shear',
        layers: [
            { alt_min: 15000, alt_max: 16500, u: -12,   v:  5   },
            { alt_min: 16500, alt_max: 17500, u:  10,   v:  7.07},
            { alt_min: 17500, alt_max: 18500, u: -14.1, v: -7.07},
            { alt_min: 18500, alt_max: 20000, u:  8,    v:  5   },
            { alt_min: 20000, alt_max: 22000, u: -10,   v: -6   },
        ],
    },
    calm: {
        name: 'Calm Stratosphere',
        layers: [
            { alt_min: 15000, alt_max: 16500, u: -3,  v: -2   },
            { alt_min: 16500, alt_max: 17500, u:  3,  v:  2.12},
            { alt_min: 17500, alt_max: 18500, u: -4,  v: -2.83},
            { alt_min: 18500, alt_max: 20000, u:  2,  v:  1   },
            { alt_min: 20000, alt_max: 22000, u: -2,  v: -1   },
        ],
    },
    'jet-crossing': {
        name: 'Jet Crossing',
        layers: [
            { alt_min: 15000, alt_max: 16500, u:  3,  v:  1   },
            { alt_min: 16500, alt_max: 17500, u: 18,  v:  2   },
            { alt_min: 17500, alt_max: 18500, u:  5,  v: -1   },
            { alt_min: 18500, alt_max: 20000, u: -3,  v:  0   },
            { alt_min: 20000, alt_max: 22000, u:  2,  v:  1   },
        ],
    },
    uniform: {
        name: 'Uniform (Impossible)',
        layers: [
            { alt_min: 15000, alt_max: 16500, u: 8, v: 0 },
            { alt_min: 16500, alt_max: 17500, u: 8, v: 0 },
            { alt_min: 17500, alt_max: 18500, u: 8, v: 0 },
            { alt_min: 18500, alt_max: 20000, u: 8, v: 0 },
            { alt_min: 20000, alt_max: 22000, u: 8, v: 0 },
        ],
    },
};

/**
 * Get base wind from static layers at a given altitude.
 * Uses exclusive upper bounds for all but the last layer to avoid
 * boundary ambiguity.
 */
export function getBaseWind(layers, alt_m) {
    for (let i = 0; i < layers.length; i++) {
        const isLast = i === layers.length - 1;
        if (alt_m >= layers[i].alt_min &&
            (isLast ? alt_m <= layers[i].alt_max : alt_m < layers[i].alt_max)) {
            return { u: layers[i].u, v: layers[i].v };
        }
    }
    // Fallback: clamp to nearest layer
    if (alt_m < layers[0].alt_min) return { u: layers[0].u, v: layers[0].v };
    const last = layers[layers.length - 1];
    return { u: last.u, v: last.v };
}

/**
 * Apply temporal variation to base wind.
 * Includes diurnal cycle, inertia-gravity waves, planetary waves, and noise.
 */
export function applyTemporalVariation(baseU, baseV, alt_m, time_s) {
    const w = runtime.wind;
    let u = baseU, v = baseV;

    // 1. Diurnal thermal tide (24h period)
    const diurnalPhase = (2 * Math.PI * time_s) / 86400;
    const diurnalMod = 1 + w.DIURNAL_AMPLITUDE * Math.sin(diurnalPhase);
    const diurnalRot = w.DIURNAL_AMPLITUDE * 0.67 * Math.sin(diurnalPhase + Math.PI / 4);
    const cosR = Math.cos(diurnalRot), sinR = Math.sin(diurnalRot);
    const u1 = u * diurnalMod, v1 = v * diurnalMod;
    u = u1 * cosR - v1 * sinR;
    v = u1 * sinR + v1 * cosR;

    // 2. Inertia-gravity wave (IGW) — altitude-dependent phase
    const igwPhase = (2 * Math.PI * time_s) / w.IGW_PERIOD_S -
                     (2 * Math.PI * alt_m) / w.IGW_VERT_WAVELENGTH_M;
    u += w.IGW_AMPLITUDE * Math.cos(igwPhase);
    v += w.IGW_AMPLITUDE * 0.7 * Math.sin(igwPhase);  // elliptical hodograph

    // 3. Planetary wave (multi-day, altitude-dependent)
    const pwPhase = (2 * Math.PI * time_s) / w.PW_PERIOD_S -
                    (2 * Math.PI * alt_m) / w.PW_VERT_WAVELENGTH_M;
    u += w.PW_AMPLITUDE * Math.sin(pwPhase);
    v += w.PW_AMPLITUDE * 0.5 * Math.cos(pwPhase);

    // 4. Stochastic gravity wave background (deterministic from time+alt seed)
    const slot = Math.floor(time_s / 1800);
    const altBin = Math.floor(alt_m / 500);
    const seed = ((slot * 7919 + altBin * 104729) & 0x7FFFFFFF) / 0x7FFFFFFF;
    const seed2 = ((slot * 104729 + altBin * 7919) & 0x7FFFFFFF) / 0x7FFFFFFF;
    u += w.NOISE_STD * (seed * 2 - 1);
    v += w.NOISE_STD * (seed2 * 2 - 1);

    return { u, v };
}

/**
 * Get wind at (altitude, time) for a given set of layers.
 * Combines base wind + temporal variation.
 */
export function getWind(layers, alt_m, time_s) {
    const base = getBaseWind(layers, alt_m);
    return applyTemporalVariation(base.u, base.v, alt_m, time_s);
}

/**
 * Get wind column — wind at multiple altitudes for a given time.
 * Returns array of { alt_m, u, v, speed, direction_deg }.
 */
export function getWindColumn(layers, time_s, altitudes) {
    return altitudes.map(alt_m => {
        const { u, v } = getWind(layers, alt_m, time_s);
        const speed = Math.sqrt(u * u + v * v);
        const direction_deg = (Math.atan2(u, v) * 180 / Math.PI + 360) % 360;
        return { alt_m, u, v, speed, direction_deg };
    });
}

// ── Real wind data container ────────────────────────────────────────

/**
 * WindProfile holds wind data from real sources (GFS, ERA5).
 * Provides interpolation across altitude and time.
 */
export class WindProfile {
    /**
     * @param {Array<{time_s: number, levels: Array<{alt_m: number, u: number, v: number}>}>} snapshots
     *   Time-ordered array of wind soundings. Each has a time_s and levels array.
     */
    constructor(snapshots) {
        this.snapshots = snapshots;
        this.minTime = snapshots[0]?.time_s ?? 0;
        this.maxTime = snapshots[snapshots.length - 1]?.time_s ?? 0;
    }

    /**
     * Bilinear interpolation: time × altitude → { u, v }
     */
    getWind(alt_m, time_s) {
        if (this.snapshots.length === 0) return { u: 0, v: 0 };
        if (this.snapshots.length === 1) return this._interpAlt(this.snapshots[0].levels, alt_m);

        // Find bracketing time snapshots
        let i0 = 0, i1 = this.snapshots.length - 1;
        for (let i = 0; i < this.snapshots.length - 1; i++) {
            if (time_s >= this.snapshots[i].time_s && time_s <= this.snapshots[i + 1].time_s) {
                i0 = i; i1 = i + 1; break;
            }
        }
        if (time_s <= this.snapshots[0].time_s) { i0 = 0; i1 = 0; }
        if (time_s >= this.snapshots[this.snapshots.length - 1].time_s) {
            i0 = i1 = this.snapshots.length - 1;
        }

        const w0 = this._interpAlt(this.snapshots[i0].levels, alt_m);
        if (i0 === i1) return w0;
        const w1 = this._interpAlt(this.snapshots[i1].levels, alt_m);

        // Linear time interpolation
        const dt = this.snapshots[i1].time_s - this.snapshots[i0].time_s;
        const t = (time_s - this.snapshots[i0].time_s) / dt;
        return {
            u: w0.u * (1 - t) + w1.u * t,
            v: w0.v * (1 - t) + w1.v * t,
        };
    }

    _interpAlt(levels, alt_m) {
        if (levels.length === 0) return { u: 0, v: 0 };
        if (levels.length === 1) return { u: levels[0].u, v: levels[0].v };

        // Clamp to range
        if (alt_m <= levels[0].alt_m) return { u: levels[0].u, v: levels[0].v };
        if (alt_m >= levels[levels.length - 1].alt_m) {
            const last = levels[levels.length - 1];
            return { u: last.u, v: last.v };
        }

        // Find bracketing levels
        for (let i = 0; i < levels.length - 1; i++) {
            if (alt_m >= levels[i].alt_m && alt_m <= levels[i + 1].alt_m) {
                const dAlt = levels[i + 1].alt_m - levels[i].alt_m;
                const t = (alt_m - levels[i].alt_m) / dAlt;
                return {
                    u: levels[i].u * (1 - t) + levels[i + 1].u * t,
                    v: levels[i].v * (1 - t) + levels[i + 1].v * t,
                };
            }
        }
        const last = levels[levels.length - 1];
        return { u: last.u, v: last.v };
    }
}

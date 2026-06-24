/**
 * wind_archive.js — ERA5 reanalysis wind archive for DQN training.
 *
 * Loads pre-converted ERA5 JSON files (produced by era5_to_json.py) and
 * provides wind lookups with trilinear interpolation (time × altitude × space).
 *
 * Interface compatible with the synthetic wind preset system:
 *   archive.sampleEpisode(rng, {duration_s}) →
 *     { truthWindFn, baseWindFn, targetLat, targetLon, meta }
 *
 * truthWindFn(alt_m, time_s) → { u, v }  — same as getWind(layers, alt_m, time_s)
 * baseWindFn(alt_m)          → { u, v }  — same as getBaseWind(layers, alt_m)
 *
 * Vertical interpolation: altitude → pressure (US Std Atmo, isothermal 11-20km)
 *   then log-pressure linear between ERA5 levels.
 * Temporal interpolation: linear between the two nearest 00Z/12Z steps.
 * Spatial: uses the nearest grid point (2.5° grid spacing >> 30 km spawn radius,
 *   so bilinear spatial interp adds negligible accuracy vs cost of implementation).
 *
 * Dependencies: none (standalone, no imports).
 * Node.js usage only (uses fs.readFileSync).
 */

import { readFileSync, readdirSync } from 'fs';
import { join } from 'path';

// ── US Standard Atmosphere (11-20 km isothermal tropopause) ──────────────────
// Used to convert balloon altitude to ERA5 pressure level.

const _T11  = 216.65;   // K  — temperature (isothermal above 11 km)
const _P11  = 22632.1;  // Pa — pressure at 11 000 m
const _H11  = 11000;    // m  — tropopause base altitude
const _g    = 9.80665;  // m/s²
const _R    = 287.058;  // J/(kg·K) dry air gas constant
const _Hscale = _R * _T11 / _g;  // scale height ≈ 6341 m

/**
 * Convert altitude to pressure (hPa).
 * Valid for 11-32 km; the balloon operates 16.5-18.5 km so this is exact.
 */
function altToPressureHPa(alt_m) {
    // Below 11 km: standard troposphere (not needed but handled gracefully)
    if (alt_m < _H11) {
        const T0 = 288.15, L = 0.0065, P0 = 101325;
        return (P0 * Math.pow(1 - L * alt_m / T0, _g / (_R * L))) / 100;
    }
    // 11-20 km isothermal
    if (alt_m <= 20000) {
        return (_P11 * Math.exp(-(alt_m - _H11) / _Hscale)) / 100;
    }
    // 20-32 km stratosphere lapse (+1 K/km), T = 216.65 + (h-20000)*0.001
    const T20 = 216.65, L20 = 0.001, P20 = 5474.89;
    return (P20 * Math.pow(T20 / (T20 + L20 * (alt_m - 20000)),
        _g / (_R * L20))) / 100;
}

// ── WindArchive ───────────────────────────────────────────────────────────────

export class WindArchive {
    constructor() {
        /** @type {Array<MonthData>} sorted by time */
        this._months = [];
        /** Flat sorted array of all unix timestamps across all months */
        this._allTimes = [];
        /** Whether the archive is loaded */
        this.loaded = false;
    }

    /**
     * Load all era5_wind_YYYY_MM.json files from a directory.
     * @param {string} dir - Path to era5_json/ directory
     */
    load(dir) {
        const files = readdirSync(dir)
            .filter(f => f.startsWith('era5_wind_') && f.endsWith('.json'))
            .sort();

        if (files.length === 0) {
            throw new Error(`No ERA5 JSON files found in ${dir}`);
        }

        for (const file of files) {
            const raw = JSON.parse(readFileSync(join(dir, file), 'utf8'));
            this._months.push(new MonthData(raw));
        }

        // Build sorted global time index
        const seen = new Set();
        const times = [];
        for (const m of this._months) {
            for (const t of m.times) {
                if (!seen.has(t)) { seen.add(t); times.push(t); }
            }
        }
        times.sort((a, b) => a - b);
        this._allTimes = times;

        // Cache grid metadata from first month (same across all months)
        const first = this._months[0];
        this.lats   = first.lats;
        this.lons   = first.lons;
        this.levels = first.levels;
        this.nLats  = first.lats.length;
        this.nLons  = first.lons.length;
        this.nLevels = first.levels.length;

        this.loaded = true;
        return this;
    }

    /**
     * Get wind at exact (lat, lon360, press_hpa, unix_t).
     * Uses nearest-grid-point + linear temporal interpolation.
     *
     * @param {number} lat       - degrees north (-20 to 20)
     * @param {number} lon360    - degrees east (100 to 260)
     * @param {number} press_hpa - pressure in hPa (clamped to available levels)
     * @param {number} unix_t    - unix timestamp (seconds)
     * @returns {{ u: number, v: number }}
     */
    getWindAt(lat, lon360, press_hpa, unix_t) {
        // Nearest grid point (2.5° grid >> 30 km spawn radius)
        const iLat = this._nearestIdx(this.lats, lat);
        const iLon = this._nearestIdx(this.lons, lon360);

        // Log-pressure interpolation between bracketing ERA5 levels
        const { iLo, iHi, wHi } = this._levelWeights(press_hpa);

        // Temporal interpolation between bracketing time steps
        return this._interpTime(iLat, iLon, iLo, iHi, wHi, unix_t);
    }

    /**
     * Sample a random episode from the archive.
     *
     * @param {Function} rng - seeded random function () → [0,1)
     * @param {Object}   opts
     * @param {number}   opts.duration_s - Episode length in seconds (default 7200)
     * @returns {{ truthWindFn, baseWindFn, targetLat, targetLon, meta }}
     */
    sampleEpisode(rng, {
        duration_s   = 7200,
        minShear_ms  = 0,      // Minimum |u(16500) - u(18500)| AND both sides must have opposite signs.
                               // Set to e.g. 8 to require genuinely opposing winds ≥4 m/s on each side.
        latRange     = null,   // [latMin, latMax] to restrict spatial sampling (degrees)
        lonRange     = null,   // [lon360Min, lon360Max] to restrict spatial sampling (degrees east)
        maxAttempts  = 50,     // Max rejection-sampling attempts before falling back
    } = {}) {
        if (!this.loaded) throw new Error('WindArchive not loaded');

        // Pre-filter lat/lon indices if ranges specified
        const latIdxs = latRange
            ? this.lats.map((v, i) => v >= latRange[0] && v <= latRange[1] ? i : -1).filter(i => i >= 0)
            : null;
        const lonIdxs = lonRange
            ? this.lons.map((v, i) => v >= lonRange[0] && v <= lonRange[1] ? i : -1).filter(i => i >= 0)
            : null;

        if (latIdxs && latIdxs.length === 0) throw new Error('latRange matches no grid points');
        if (lonIdxs && lonIdxs.length === 0) throw new Error('lonRange matches no grid points');

        const minStepsNeeded = Math.ceil(duration_s / 43200) + 1;
        const maxStartIdx    = this._allTimes.length - minStepsNeeded;
        if (maxStartIdx < 0) throw new Error('Archive too short for requested duration');

        // Sample altitudes for shear check: bottom and top of reachable band
        const ALT_LO = 16500;   // m  — lower edge of reachable band
        const ALT_HI = 18500;   // m  — upper edge of reachable band

        let chosen = null;
        for (let attempt = 0; attempt < maxAttempts; attempt++) {
            // Pick a random grid point (from filtered set or full grid)
            const iLat = latIdxs
                ? latIdxs[Math.floor(rng() * latIdxs.length)]
                : Math.floor(rng() * this.nLats);
            const iLon = lonIdxs
                ? lonIdxs[Math.floor(rng() * lonIdxs.length)]
                : Math.floor(rng() * this.nLons);
            const lat    = this.lats[iLat];
            const lon360 = this.lons[iLon];

            // Pick a random start time
            const startIdx  = Math.floor(rng() * (maxStartIdx + 1));
            const startUnix = this._allTimes[startIdx];

            // Shear filter: require genuinely opposing u-winds across the altitude band.
            // Both sides must exceed half the threshold (e.g. minShear_ms=8 → each ≥4 m/s),
            // and they must have opposite signs (otherwise same-direction shear doesn't help).
            if (minShear_ms > 0) {
                const pLo = altToPressureHPa(ALT_LO);
                const pHi = altToPressureHPa(ALT_HI);
                const uLo = this.getWindAt(lat, lon360, pLo, startUnix).u;
                const uHi = this.getWindAt(lat, lon360, pHi, startUnix).u;
                const minEach = minShear_ms / 2;  // each side must be at least this large
                const hasOpposing = (uLo > minEach && uHi < -minEach) ||
                                    (uLo < -minEach && uHi > minEach);
                if (!hasOpposing) continue;
            }

            chosen = { iLat, iLon, lat, lon360, startIdx, startUnix };
            break;
        }

        // Fallback: unconstrained random sample if no candidate met the shear threshold
        if (!chosen) {
            const iLat = Math.floor(rng() * this.nLats);
            const iLon = Math.floor(rng() * this.nLons);
            const startIdx = Math.floor(rng() * (maxStartIdx + 1));
            chosen = {
                iLat, iLon,
                lat:    this.lats[iLat],
                lon360: this.lons[iLon],
                startIdx,
                startUnix: this._allTimes[startIdx],
            };
        }

        const { lat, lon360, startIdx, startUnix } = chosen;
        const targetLon = lon360 > 180 ? lon360 - 360 : lon360;

        const self = this;
        const truthWindFn = (alt_m, time_s) => {
            const press_hpa = altToPressureHPa(alt_m);
            return self.getWindAt(lat, lon360, press_hpa, startUnix + time_s);
        };
        const baseWindFn = (alt_m) => truthWindFn(alt_m, 0);

        return {
            truthWindFn,
            baseWindFn,
            targetLat: lat,
            targetLon,
            meta: { lat, lon360, startUnix, startIdx },
        };
    }

    // ── Private helpers ─────────────────────────────────────────────────

    /** Find index of nearest value in a sorted array. */
    _nearestIdx(arr, val) {
        let best = 0, bestDist = Math.abs(arr[0] - val);
        for (let i = 1; i < arr.length; i++) {
            const d = Math.abs(arr[i] - val);
            if (d < bestDist) { bestDist = d; best = i; }
            if (arr[i] > val && arr[0] < arr[1]) break; // ascending, past target
        }
        return best;
    }

    /** Compute log-pressure interpolation weights between ERA5 levels. */
    _levelWeights(press_hpa) {
        const levels = this.levels; // e.g. [300, 250, 200, 150, 100, 70, 50, 30]
        // Levels are in descending order (300 hPa at index 0, 30 hPa at end).
        // Higher pressure = lower altitude. We want the two levels bracketing press_hpa.
        let iLo = 0, iHi = 1;

        if (press_hpa >= levels[0]) {
            // Above highest pressure (lowest altitude) — clamp
            return { iLo: 0, iHi: 0, wHi: 0 };
        }
        if (press_hpa <= levels[levels.length - 1]) {
            // Below lowest pressure (highest altitude) — clamp
            const n = levels.length - 1;
            return { iLo: n, iHi: n, wHi: 0 };
        }

        // Find bracket: levels[iLo] > press_hpa > levels[iHi]
        for (let i = 0; i < levels.length - 1; i++) {
            if (levels[i] >= press_hpa && levels[i + 1] <= press_hpa) {
                iLo = i; iHi = i + 1; break;
            }
        }

        // Log-pressure interpolation weight
        const logP   = Math.log(press_hpa);
        const logPLo = Math.log(levels[iLo]);
        const logPHi = Math.log(levels[iHi]);
        const wHi    = (logPLo - logP) / (logPLo - logPHi);

        return { iLo, iHi, wHi: Math.max(0, Math.min(1, wHi)) };
    }

    /**
     * Interpolate wind at a grid point between two ERA5 time steps.
     * Finds the two time steps in the archive bracketing unix_t.
     */
    _interpTime(iLat, iLon, iLo, iHi, wHi, unix_t) {
        // Find bracketing time steps across all months
        const allTimes = this._allTimes;
        let tIdx = 0;
        // Binary search for unix_t in allTimes
        let lo = 0, hi = allTimes.length - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (allTimes[mid] < unix_t) lo = mid + 1; else hi = mid;
        }
        tIdx = lo;

        // Clamp to valid range
        const t0Idx = Math.max(0, Math.min(allTimes.length - 2, tIdx > 0 && allTimes[tIdx] > unix_t ? tIdx - 1 : tIdx));
        const t1Idx = t0Idx + 1;

        const t0 = allTimes[t0Idx];
        const t1 = allTimes[t1Idx];
        const wT  = t1 > t0 ? (unix_t - t0) / (t1 - t0) : 0;
        const wT0 = 1 - Math.max(0, Math.min(1, wT));
        const wT1 = 1 - wT0;

        // Get wind at both time steps, both levels
        const w00 = this._getGridPoint(t0, iLat, iLon, iLo);
        const w01 = this._getGridPoint(t0, iLat, iLon, iHi);
        const w10 = this._getGridPoint(t1, iLat, iLon, iLo);
        const w11 = this._getGridPoint(t1, iLat, iLon, iHi);

        // Bilinear interp: time × pressure level
        const u = wT0 * ((1 - wHi) * w00.u + wHi * w01.u) +
                  wT1 * ((1 - wHi) * w10.u + wHi * w11.u);
        const v = wT0 * ((1 - wHi) * w00.v + wHi * w01.v) +
                  wT1 * ((1 - wHi) * w10.v + wHi * w11.v);

        return { u, v };
    }

    /**
     * Look up u, v at an exact (unix_t, iLat, iLon, iLevel) grid point.
     * Finds the correct month and indexes into its flat array.
     */
    _getGridPoint(unix_t, iLat, iLon, iLevel) {
        // Find the month that contains unix_t
        for (const m of this._months) {
            const tIdx = m.timeIndex.get(unix_t);
            if (tIdx === undefined) continue;
            const idx = m.flatIndex(tIdx, iLevel, iLat, iLon);
            return { u: m.u[idx], v: m.v[idx] };
        }
        // Fallback: return zeros (shouldn't happen if archive is complete)
        return { u: 0, v: 0 };
    }

    /** Summary string for logging. */
    toString() {
        const n = this._months.length;
        const tStart = new Date(this._allTimes[0] * 1000).toISOString().slice(0, 10);
        const tEnd   = new Date(this._allTimes[this._allTimes.length - 1] * 1000).toISOString().slice(0, 10);
        return `WindArchive(${n} months, ${this._allTimes.length} steps, ${tStart}–${tEnd}, ${this.nLats}×${this.nLons} grid)`;
    }
}

// ── MonthData ─────────────────────────────────────────────────────────────────

class MonthData {
    constructor(raw) {
        this.year    = raw.year;
        this.month   = raw.month;
        this.lats    = raw.lats;
        this.lons    = raw.lons;
        this.levels  = raw.levels_hpa;
        this.times   = raw.times_unix.map(Number);
        this.shape   = raw.shape;   // [T, L, Lat, Lon]

        // Store u, v as Float32Arrays for memory efficiency
        this.u = new Float32Array(raw.u);
        this.v = new Float32Array(raw.v);

        // Fast time lookup: unix_t → time index
        this.timeIndex = new Map();
        for (let i = 0; i < this.times.length; i++) {
            this.timeIndex.set(this.times[i], i);
        }

        const [T, L, Lat, Lon] = this.shape;
        this._L = L; this._Lat = Lat; this._Lon = Lon;
    }

    /** Flat array index for (tIdx, lIdx, latIdx, lonIdx). */
    flatIndex(tIdx, lIdx, latIdx, lonIdx) {
        return tIdx * (this._L * this._Lat * this._Lon)
             + lIdx * (this._Lat * this._Lon)
             + latIdx * this._Lon
             + lonIdx;
    }
}

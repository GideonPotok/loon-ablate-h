#!/usr/bin/env node
/**
 * make_era5_fixture.mjs — write a small ERA5 JSON archive in the exact schema
 * era5_to_json.py produces, so wind_archive/wind_source can be tested without
 * mounting the 169 MB real archive.
 *
 * This is a FIXTURE, not data. The wind is analytic, and deliberately shaped
 * so the balloon band has something to work with:
 *
 *   mag(lon, t) = 6 + 5·cos(2π t / P + lon·π/180)        → sweeps 1 … 11 m/s
 *   sign(level) = +1 at ≥100 hPa,  −1 at ≤70 hPa         → reversal in the band
 *   u = sign · mag + lat/8
 *   v = sign · 0.4 · mag · sin(2π t / P)
 *
 * The 100/70 hPa sign flip is the point: those are the only two levels that
 * bracket the 16.5–18.5 km band, so this exercises the exact log-pressure
 * interpolation a real ERA5 lookup does there. Because mag sweeps through 4
 * m/s, some cells clear the minShear_ms=8 filter and some do not, which is
 * what makes the rejection sampler testable.
 *
 * Usage: node scratch/make_era5_fixture.mjs <outdir> [nMonths]
 */
import { writeFileSync, mkdirSync } from 'fs';
import { join } from 'path';

const OUT = process.argv[2];
const N_MONTHS = +(process.argv[3] || 2);
if (!OUT) { console.error('usage: make_era5_fixture.mjs <outdir> [nMonths]'); process.exit(1); }

mkdirSync(OUT, { recursive: true });

// Real grid, matching era5_json/: 2.5°, 20°N→20°S, 100°E→260°E, 8 levels, 12-hourly.
const lats = Array.from({ length: 17 }, (_, i) => 20.0 - i * 2.5);
const lons = Array.from({ length: 65 }, (_, i) => 100.0 + i * 2.5);
const levels_hpa = [300, 250, 200, 150, 100, 70, 50, 30];
const PERIOD_S = 5 * 86400;

const daysIn = (y, m) => new Date(Date.UTC(y, m, 0)).getUTCDate();

let written = 0;
for (let k = 0; k < N_MONTHS; k++) {
    const year  = 2023 + Math.floor(k / 12);
    const month = (k % 12) + 1;

    const t0 = Date.UTC(year, month - 1, 1) / 1000;
    const times_unix = [];
    for (let d = 0; d < daysIn(year, month); d++) {
        times_unix.push(t0 + d * 86400);
        times_unix.push(t0 + d * 86400 + 43200);
    }

    const T = times_unix.length, L = levels_hpa.length;
    const Lat = lats.length, Lon = lons.length;
    const u = new Array(T * L * Lat * Lon);
    const v = new Array(T * L * Lat * Lon);

    let n = 0;
    for (let ti = 0; ti < T; ti++) {
        const ph = (2 * Math.PI * times_unix[ti]) / PERIOD_S;
        for (let li = 0; li < L; li++) {
            // levels_hpa is descending: index 4 is 100 hPa, index 5 is 70 hPa.
            const sign = levels_hpa[li] >= 100 ? 1 : -1;
            for (let ai = 0; ai < Lat; ai++) {
                const latTerm = lats[ai] / 8;
                for (let oi = 0; oi < Lon; oi++) {
                    const mag = 6 + 5 * Math.cos(ph + lons[oi] * Math.PI / 180);
                    u[n] = +(sign * mag + latTerm).toFixed(4);
                    v[n] = +(sign * 0.4 * mag * Math.sin(ph)).toFixed(4);
                    n++;
                }
            }
        }
    }

    const name = `era5_wind_${year}_${String(month).padStart(2, '0')}.json`;
    writeFileSync(join(OUT, name), JSON.stringify({
        year, month, lats, lons, levels_hpa, times_unix,
        shape: [T, L, Lat, Lon], u, v,
    }));
    written++;
}

console.log(`wrote ${written} fixture month(s) to ${OUT}`);
console.log(`grid ${lats.length}×${lons.length}, ${levels_hpa.length} levels, 12-hourly`);

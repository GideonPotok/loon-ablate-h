#!/usr/bin/env node
/**
 * validate_era5_archive.mjs — standalone smoke test for js/wind_archive.js
 * against the real ERA5 JSON files, before any of it is wired into a server.
 *
 * Checks, in order:
 *   1. The archive loads every month and reports a sane grid + time span.
 *   2. Wind lookups at balloon altitudes return plausible values.
 *   3. sampleEpisode() produces episodes inside the tile with usable duration.
 *   4. The minShear_ms rejection filter actually biases toward opposing winds.
 *   5. In-band shear (16.5 vs 18.5 km) is reported so the ERA5 ceiling is
 *      visible before anyone reads a policy score off this env.
 *
 * Usage:
 *   node scratch/validate_era5_archive.mjs [era5_json_dir]
 */
import { WindArchive } from '../js/wind_archive.js';

const DIR = process.argv[2] || '/Volumes/Gideon SDD/data/era5_json';

const ALT_LO = 16500;
const ALT_HI = 18500;

function fail(msg) { console.error(`FAIL: ${msg}`); process.exitCode = 1; }
function ok(msg)   { console.log(`  ok   ${msg}`); }

console.log(`ERA5 archive validation — ${DIR}\n`);

// ── 1. Load ──────────────────────────────────────────────────────────────────
const t0 = Date.now();
const archive = new WindArchive().load(DIR);
const loadMs = Date.now() - t0;

console.log('1. load');
console.log(`     ${archive}`);
console.log(`     loaded in ${loadMs} ms, heap ${(process.memoryUsage().heapUsed / 1e6).toFixed(0)} MB`);
// The real era5_json/ archive is 24 months (2023-2024); a fixture is smaller.
// Only an empty load is a failure — a short archive just gets called out.
const nMonths = archive._months.length;
if (nMonths === 0) fail('archive loaded zero months');
else if (nMonths === 24) ok('24 months (full 2023-2024 archive)');
else console.log(`     NOTE ${nMonths} month(s) — not the full 24-month archive`);

const tStart = new Date(archive._allTimes[0] * 1000).toISOString();
const tEnd   = new Date(archive._allTimes.at(-1) * 1000).toISOString();
console.log(`     span ${tStart} → ${tEnd}  (${archive._allTimes.length} steps)`);

// Cadence: every gap should be identical for a clean archive.
const gaps = new Set();
for (let i = 1; i < archive._allTimes.length; i++) {
    gaps.add(archive._allTimes[i] - archive._allTimes[i - 1]);
}
const gapList = [...gaps].sort((a, b) => a - b);
console.log(`     step gaps (s): ${gapList.join(', ')}`);
if (gaps.size === 1) ok(`uniform cadence ${gapList[0] / 3600} h`);
else fail(`non-uniform cadence — ${gaps.size} distinct gaps, archive may have holes`);

// Grid must be uniform across months, since load() caches it from month 0.
let gridConsistent = true;
for (const m of archive._months) {
    if (m.lats.length !== archive.nLats || m.lons.length !== archive.nLons) gridConsistent = false;
    if (m.lons[0] !== archive.lons[0] || m.lats[0] !== archive.lats[0]) gridConsistent = false;
}
if (gridConsistent) ok(`grid uniform across months (${archive.nLats}×${archive.nLons}, ${archive.nLevels} levels)`);
else fail('months disagree on grid — load() caches month 0 metadata, lookups would be wrong');

console.log(`     lat ${archive.lats[0]} → ${archive.lats.at(-1)}   lon ${archive.lons[0]} → ${archive.lons.at(-1)}`);
console.log(`     levels ${archive.levels.join('/')} hPa`);

// ── 2. Point lookups ─────────────────────────────────────────────────────────
console.log('\n2. wind lookups at the station (lat 0, lon 170)');
const tMid = archive._allTimes[Math.floor(archive._allTimes.length / 2)];
let lookupsFinite = true;
for (const alt of [ALT_LO, 17500, ALT_HI]) {
    const p = pressureAt(alt);
    const w = archive.getWindAt(0, 170, p, tMid);
    const spd = Math.hypot(w.u, w.v);
    if (!Number.isFinite(w.u) || !Number.isFinite(w.v)) lookupsFinite = false;
    if (spd > 150) lookupsFinite = false;
    console.log(`     ${alt} m (${p.toFixed(1)} hPa): u=${w.u.toFixed(1)} v=${w.v.toFixed(1)} |w|=${spd.toFixed(1)} m/s`);
}
if (lookupsFinite) ok('finite and physically plausible');
else fail('lookup returned NaN or an absurd speed');

// Off-grid times must clamp, not explode.
const before = archive.getWindAt(0, 170, 70, archive._allTimes[0] - 86400);
const after  = archive.getWindAt(0, 170, 70, archive._allTimes.at(-1) + 86400);
if (Number.isFinite(before.u) && Number.isFinite(after.u)) ok('out-of-range times clamp cleanly');
else fail('out-of-range time lookup produced NaN');

// ── 3. sampleEpisode ─────────────────────────────────────────────────────────
console.log('\n3. sampleEpisode (72 h episodes, matching the probe protocol)');
const DURATION_S = 72 * 3600;
const N = 200;

const samples = [];
for (let i = 0; i < N; i++) {
    samples.push(archive.sampleEpisode(makeRng(42 + i * 1000003), { duration_s: DURATION_S }));
}

const inTile = samples.every(s =>
    s.meta.lat >= Math.min(...archive.lats) && s.meta.lat <= Math.max(...archive.lats) &&
    s.meta.lon360 >= Math.min(...archive.lons) && s.meta.lon360 <= Math.max(...archive.lons));
if (inTile) ok(`all ${N} episodes inside the tile`);
else fail('an episode sampled outside the grid');

// Every episode must have real wind for its whole duration, not clamped tail.
const lastStart = archive._allTimes.at(-1) - DURATION_S;
const nTruncated = samples.filter(s => s.meta.startUnix > lastStart).length;
if (nTruncated === 0) ok('no episode runs past the end of the archive');
else console.log(`     WARN ${nTruncated}/${N} episodes extend past the archive end and will clamp to the last step`);

const uniqueCells = new Set(samples.map(s => `${s.meta.lat},${s.meta.lon360}`)).size;
const uniqueTimes = new Set(samples.map(s => s.meta.startUnix)).size;
console.log(`     coverage: ${uniqueCells} distinct grid cells, ${uniqueTimes} distinct start times`);
if (uniqueCells > N * 0.5) ok('spatial sampling is well spread');
else fail('episodes are clustering on a few cells');

// Wind must actually change over the episode, or the eval is static.
const drifts = samples.map(s => {
    const a = s.truthWindFn(17500, 0);
    const b = s.truthWindFn(17500, DURATION_S);
    return Math.hypot(b.u - a.u, b.v - a.v);
});
console.log(`     |Δwind| over 72 h at 17.5 km: median ${median(drifts).toFixed(1)} m/s, max ${Math.max(...drifts).toFixed(1)}`);
if (median(drifts) > 1) ok('wind evolves over the episode');
else fail('wind is nearly static over 72 h — check temporal interpolation');

// ── 4. Shear filter ──────────────────────────────────────────────────────────
console.log('\n4. minShear_ms rejection filter');
const opposing = (s) => {
    const uLo = s.truthWindFn(ALT_LO, 0).u;
    const uHi = s.truthWindFn(ALT_HI, 0).u;
    return (uLo > 4 && uHi < -4) || (uLo < -4 && uHi > 4);
};
const fracPlain = samples.filter(opposing).length / N;
const filtered = [];
for (let i = 0; i < N; i++) {
    filtered.push(archive.sampleEpisode(makeRng(42 + i * 1000003), { duration_s: DURATION_S, minShear_ms: 8 }));
}
const fracFiltered = filtered.filter(opposing).length / N;
console.log(`     opposing u (each ≥4 m/s): unfiltered ${(fracPlain * 100).toFixed(1)}%  →  minShear_ms=8 ${(fracFiltered * 100).toFixed(1)}%`);
if (fracFiltered > fracPlain) ok('filter biases toward opposing shear as intended');
else fail('filter had no effect — rejection sampling may be falling through');

// ── 5. In-band shear, the number that bounds any score from this env ─────────
console.log('\n5. in-band shear available to the agent (16.5 → 18.5 km)');
const shear = samples.map(s => {
    const a = s.truthWindFn(ALT_LO, 0);
    const b = s.truthWindFn(ALT_HI, 0);
    return Math.hypot(b.u - a.u, b.v - a.v);
});
console.log(`     |w(18.5km) − w(16.5km)|: median ${median(shear).toFixed(1)}  p90 ${pct(shear, 90).toFixed(1)}  max ${Math.max(...shear).toFixed(1)} m/s`);
console.log(`     for reference — tropical preset step: 21.9 m/s, IGRA soundings: 16.0 m/s median`);

console.log(`\n${process.exitCode ? 'VALIDATION FAILED' : 'all checks passed'}`);

// ── helpers ──────────────────────────────────────────────────────────────────
function makeRng(seed) {
    let s = (seed >>> 0) || 1;
    return function () {
        s ^= s << 13; s ^= s >>> 17; s ^= s << 5; s = s >>> 0;
        return (s & 0x7FFFFFFF) / 0x80000000;
    };
}
function pressureAt(alt_m) {
    return (22632.1 * Math.exp(-(alt_m - 11000) / (287.058 * 216.65 / 9.80665))) / 100;
}
function median(a) { return pct(a, 50); }
function pct(a, p) {
    const s = [...a].sort((x, y) => x - y);
    return s[Math.min(s.length - 1, Math.floor(s.length * p / 100))];
}

/**
 * wind_source.js — one place to build the (truthWindFn, baseWindFn, target)
 * triple that every env server needs.
 *
 * All three servers (v1, v2, gassand) construct these functions identically
 * from a synthetic preset. Adding ERA5 as a second source in each of them
 * separately would mean three copies of the same decision, so it lives here
 * and each server calls resolveWindSource() once during reset.
 *
 * Two sources:
 *
 *   'preset' — the synthetic layered winds in wind.js. Target position comes
 *              from the caller (each server's TARGET_LAT/TARGET_LON constant).
 *              Wired to be byte-identical to the pre-existing inline code:
 *              getWind's `mods` parameter already defaults to null, so passing
 *              windMods=null reproduces the old three-argument call exactly.
 *
 *   'era5'   — real reanalysis from a WindArchive. sampleEpisode() picks the
 *              grid cell and start time, so the STATION MOVES: the target is
 *              wherever the sample landed, and the caller must spawn relative
 *              to the returned targetLat/targetLon rather than its own
 *              constants. Archives are cached per directory, because a
 *              169 MB parse per episode would dominate the episode itself.
 *
 * Caveat worth knowing before reading any score off the era5 source: ERA5's
 * pressure-level product has no level between 70 and 100 hPa, and the balloon
 * band (16.5–18.5 km ≈ 95–69 hPa) sits entirely inside that gap. Every bit of
 * in-band shear is therefore a log-pressure interpolation between exactly two
 * numbers. Measured on the tropical Pacific tile it runs ~10 m/s median, where
 * IGRA soundings over the same region say ~16 m/s and the `tropical` preset
 * hands the agent a 21.9 m/s step. Real wind here is not the same thing as
 * fully-resolved wind.
 */
import { getWind, getBaseWind, WIND_PRESETS } from './wind.js';
import { WindArchive } from './wind_archive.js';

/** dir → WindArchive. Module-level so a long-lived server parses each dir once. */
const _archives = new Map();

/**
 * Load (or return a cached) ERA5 archive.
 * @param {string} dir - directory of era5_wind_YYYY_MM.json files
 * @returns {WindArchive}
 */
export function loadArchive(dir) {
    if (!dir) throw new Error('era5 wind source requires a directory (era5_dir / LOON_ERA5_DIR)');
    let a = _archives.get(dir);
    if (!a) {
        a = new WindArchive().load(dir);
        _archives.set(dir, a);
    }
    return a;
}

/** Drop cached archives. Only useful for tests. */
export function _clearArchiveCache() { _archives.clear(); }

/**
 * Build the wind functions and station position for one episode.
 *
 * @param {Object}   opts
 * @param {string}   opts.source           - 'preset' (default) or 'era5'
 * @param {string}   opts.preset           - preset name, when source==='preset'
 * @param {Object}   opts.windMods         - per-episode IGW/PW mods, or null
 * @param {string}   opts.era5Dir          - archive directory, when source==='era5'
 * @param {Function} opts.rng              - seeded rng for ERA5 episode selection.
 *                                           MUST be a dedicated stream: sharing the
 *                                           spawn rng would shift every historical
 *                                           spawn position.
 * @param {number}   opts.duration_s       - episode length, so the sampler leaves room
 * @param {number}   opts.defaultTargetLat - station lat for the preset source
 * @param {number}   opts.defaultTargetLon - station lon for the preset source
 * @param {Object}   opts.era5Opts         - passthrough to sampleEpisode
 *                                           ({minShear_ms, latRange, lonRange})
 * @returns {{truthWindFn, baseWindFn, targetLat, targetLon, meta}}
 */
export function resolveWindSource({
    source = 'preset',
    preset,
    windMods = null,
    era5Dir = null,
    rng = null,
    duration_s = 7200,
    defaultTargetLat,
    defaultTargetLon,
    era5Opts = {},
} = {}) {
    if (source === 'era5') {
        if (!rng) throw new Error('era5 wind source requires a dedicated rng');
        const archive = loadArchive(era5Dir);
        const sample  = archive.sampleEpisode(rng, { duration_s, ...era5Opts });
        return {
            truthWindFn: sample.truthWindFn,
            baseWindFn:  sample.baseWindFn,
            targetLat:   sample.targetLat,
            targetLon:   sample.targetLon,
            meta: {
                source:    'era5',
                lat:       sample.meta.lat,
                lon360:    sample.meta.lon360,
                startUnix: sample.meta.startUnix,
                startISO:  new Date(sample.meta.startUnix * 1000).toISOString(),
            },
        };
    }

    if (source !== 'preset') throw new Error(`Unknown wind source: ${source}`);

    const layers = WIND_PRESETS[preset]?.layers;
    if (!layers) throw new Error(`Unknown preset: ${preset}`);

    return {
        truthWindFn: (alt_m, t) => getWind(layers, alt_m, t, windMods),
        baseWindFn:  (alt_m)    => getBaseWind(layers, alt_m),
        targetLat:   defaultTargetLat,
        targetLon:   defaultTargetLon,
        meta: { source: 'preset', preset },
    };
}

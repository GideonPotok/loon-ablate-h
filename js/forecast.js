/**
 * forecast.js — Real weather data fetching from Open-Meteo GFS API.
 *
 * Fetches wind speed and direction at stratospheric pressure levels,
 * converts to u/v components and altitude, and builds a WindProfile
 * for the simulator to consume.
 *
 * Also provides ensemble forecast support for uncertainty estimation.
 */
import { GFS_PRESSURE_LEVELS } from './config.js';
import { pressureToAltitude } from './atmosphere.js';
import { WindProfile } from './wind.js';

// ── Open-Meteo API endpoints ────────────────────────────────────────
const BASE_URL = 'https://api.open-meteo.com/v1';
const GFS_URL  = `${BASE_URL}/gfs`;
const ECMWF_URL = `${BASE_URL}/ecmwf`;

/**
 * Fetch GFS wind forecast for a given location.
 * Returns a WindProfile with hourly wind data at multiple altitudes.
 *
 * @param {number} lat — Latitude
 * @param {number} lon — Longitude
 * @param {number} forecastDays — Days of forecast (1–16)
 * @param {string} model — 'gfs' or 'ecmwf'
 * @returns {Promise<WindProfile>}
 */
export async function fetchWindForecast(lat, lon, forecastDays = 7, model = 'gfs') {
    // Build pressure level parameters
    const speedParams = GFS_PRESSURE_LEVELS.map(p => `wind_speed_${p}hPa`).join(',');
    const dirParams = GFS_PRESSURE_LEVELS.map(p => `wind_direction_${p}hPa`).join(',');

    const url = model === 'ecmwf' ? ECMWF_URL : GFS_URL;
    const params = new URLSearchParams({
        latitude: lat.toFixed(4),
        longitude: lon.toFixed(4),
        hourly: `${speedParams},${dirParams}`,
        forecast_days: forecastDays.toString(),
        wind_speed_unit: 'ms',
    });

    const response = await fetch(`${url}?${params}`);
    if (!response.ok) {
        throw new Error(`Open-Meteo API error: ${response.status} ${response.statusText}`);
    }
    const data = await response.json();
    return parseOpenMeteoResponse(data);
}

/**
 * Parse Open-Meteo JSON response into a WindProfile.
 */
function parseOpenMeteoResponse(data) {
    const hourly = data.hourly;
    const times = hourly.time;  // ISO strings

    // Reference time: first forecast hour
    const t0 = new Date(times[0]).getTime() / 1000;

    const snapshots = [];

    for (let i = 0; i < times.length; i++) {
        const time_s = new Date(times[i]).getTime() / 1000 - t0;
        const levels = [];

        for (const hPa of GFS_PRESSURE_LEVELS) {
            const speed = hourly[`wind_speed_${hPa}hPa`]?.[i];
            const dir   = hourly[`wind_direction_${hPa}hPa`]?.[i];

            if (speed == null || dir == null) continue;

            // Convert speed + direction to u, v components
            // Meteorological convention: direction is where wind comes FROM
            const dirRad = (dir + 180) * Math.PI / 180;  // convert to where wind blows TO
            const u = speed * Math.sin(dirRad);
            const v = speed * Math.cos(dirRad);

            // Map pressure level to altitude
            const alt_m = pressureToAltitude(hPa);

            levels.push({ alt_m, u, v, hPa });
        }

        // Sort by altitude
        levels.sort((a, b) => a.alt_m - b.alt_m);

        if (levels.length > 0) {
            snapshots.push({ time_s, iso: times[i], levels });
        }
    }

    return new WindProfile(snapshots);
}

/**
 * Convert a WindProfile into static wind layers (for the navigator).
 * Averages wind over a time window and bins by altitude.
 *
 * @param {WindProfile} profile — Real wind data
 * @param {number} startTime_s — Start of averaging window
 * @param {number} endTime_s — End of averaging window
 * @returns {Array<{alt_min, alt_max, u, v}>} — Static layers
 */
export function profileToLayers(profile, startTime_s = 0, endTime_s = 3600) {
    // Sample wind at multiple times within the window
    const nSamples = Math.max(1, Math.ceil((endTime_s - startTime_s) / 3600));
    const altBins = [
        { alt_min: 15000, alt_max: 16500 },
        { alt_min: 16500, alt_max: 17500 },
        { alt_min: 17500, alt_max: 18500 },
        { alt_min: 18500, alt_max: 20000 },
        { alt_min: 20000, alt_max: 22000 },
    ];

    return altBins.map(bin => {
        const midAlt = (bin.alt_min + bin.alt_max) / 2;
        let uSum = 0, vSum = 0, count = 0;

        for (let i = 0; i <= nSamples; i++) {
            const t = startTime_s + (endTime_s - startTime_s) * i / nSamples;
            const w = profile.getWind(midAlt, t);
            uSum += w.u;
            vSum += w.v;
            count++;
        }

        return { ...bin, u: uSum / count, v: vSum / count };
    });
}

/**
 * Fetch wind data and return both the raw profile and static layers.
 * This is the main entry point for the UI.
 */
export async function loadForecast(lat, lon, forecastDays = 7, model = 'gfs') {
    const profile = await fetchWindForecast(lat, lon, forecastDays, model);

    // Generate layers for the first 6 hours as initial view
    const layers = profileToLayers(profile, 0, 6 * 3600);

    return { profile, layers, model, forecastDays };
}

/**
 * Generate a time-evolving set of layers for trajectory prediction.
 * Returns layers at each hour for the full forecast period.
 *
 * @param {WindProfile} profile
 * @param {number} totalHours
 * @returns {Array<{time_h: number, layers: Array}>}
 */
export function generateTimeEvolvingLayers(profile, totalHours) {
    const result = [];
    for (let h = 0; h < totalHours; h++) {
        const startS = h * 3600;
        const endS = (h + 1) * 3600;
        const layers = profileToLayers(profile, startS, endS);
        result.push({ time_h: h, time_s: startS, layers });
    }
    return result;
}

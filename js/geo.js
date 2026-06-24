/**
 * geo.js — Geodesic math: Haversine distance, bearing, destination point.
 */
import { EARTH_RADIUS_M } from './config.js';

const DEG = Math.PI / 180;
const RAD = 180 / Math.PI;

/**
 * Haversine distance between two lat/lon points (meters).
 */
export function haversine(lat1, lon1, lat2, lon2) {
    const dLat = (lat2 - lat1) * DEG;
    const dLon = (lon2 - lon1) * DEG;
    const a = Math.sin(dLat / 2) ** 2 +
              Math.cos(lat1 * DEG) * Math.cos(lat2 * DEG) * Math.sin(dLon / 2) ** 2;
    return EARTH_RADIUS_M * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/**
 * Initial bearing from point 1 to point 2 (radians, clockwise from north).
 */
export function bearing(lat1, lon1, lat2, lon2) {
    const dLon = (lon2 - lon1) * DEG;
    const y = Math.sin(dLon) * Math.cos(lat2 * DEG);
    const x = Math.cos(lat1 * DEG) * Math.sin(lat2 * DEG) -
              Math.sin(lat1 * DEG) * Math.cos(lat2 * DEG) * Math.cos(dLon);
    return Math.atan2(y, x);
}

/**
 * Bearing from (lat, lon) to target using flat-earth approximation.
 * Faster than full spherical for short distances. Returns radians.
 */
export function bearingFlat(lat, lon, targetLat, targetLon) {
    return Math.atan2(
        (targetLon - lon) * Math.cos(lat * DEG),
        targetLat - lat
    );
}

/**
 * Destination point given start, bearing (radians), and distance (meters).
 */
export function destination(lat, lon, brng, dist_m) {
    const d = dist_m / EARTH_RADIUS_M;
    const lat1 = lat * DEG;
    const lon1 = lon * DEG;
    const lat2 = Math.asin(
        Math.sin(lat1) * Math.cos(d) + Math.cos(lat1) * Math.sin(d) * Math.cos(brng)
    );
    const lon2 = lon1 + Math.atan2(
        Math.sin(brng) * Math.sin(d) * Math.cos(lat1),
        Math.cos(d) - Math.sin(lat1) * Math.sin(lat2)
    );
    return { lat: lat2 * RAD, lon: lon2 * RAD };
}

/**
 * Compute wind approach rate (m/s toward target).
 * Positive = approaching, negative = receding.
 */
export function windApproachRate(u, v, brng) {
    return v * Math.cos(brng) + u * Math.sin(brng);
}

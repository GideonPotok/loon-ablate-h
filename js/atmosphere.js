/**
 * atmosphere.js — US Standard Atmosphere 1976 model.
 *
 * Pure functions for temperature, pressure, density at any altitude.
 * Also provides equilibrium altitude computation and derived value
 * recalculation for the runtime config.
 */
import { G, R_AIR, runtime } from './config.js';

// ── US Standard Atmosphere 1976 — piecewise linear temperature ──────
// Each layer: [base_alt_m, base_temp_K, lapse_rate_K_per_m, base_pressure_Pa]
const ATMO_LAYERS = [
    [     0, 288.15, -0.0065,  101325   ],   // Troposphere
    [ 11000, 216.65,  0.0,      22632.1 ],   // Tropopause
    [ 20000, 216.65,  0.001,     5474.89],   // Stratosphere 1
    [ 32000, 228.65,  0.0028,     868.02],   // Stratosphere 2
    [ 47000, 270.65,  0.0,        110.91],   // Stratopause
];

/**
 * Compute atmospheric properties at a given altitude.
 * @param {number} alt_m — Altitude in meters
 * @returns {{ T_K: number, P_Pa: number, rho: number }}
 */
export function atmosphereAt(alt_m) {
    let layer = ATMO_LAYERS[0];
    for (let i = ATMO_LAYERS.length - 1; i >= 0; i--) {
        if (alt_m >= ATMO_LAYERS[i][0]) { layer = ATMO_LAYERS[i]; break; }
    }
    const [h0, T0, L, P0] = layer;
    const dh = alt_m - h0;
    let T_K, P_Pa;

    if (Math.abs(L) < 1e-10) {
        // Isothermal layer
        T_K = T0;
        P_Pa = P0 * Math.exp(-G * dh / (R_AIR * T0));
    } else {
        T_K = T0 + L * dh;
        P_Pa = P0 * Math.pow(T_K / T0, -G / (L * R_AIR));
    }
    const rho = P_Pa / (R_AIR * T_K);
    return { T_K, P_Pa, rho };
}

/**
 * Find the equilibrium altitude for a given total mass + balloon volume
 * via bisection. At equilibrium, buoyancy = weight.
 */
export function findEquilibrium(totalMass_kg, volume_m3) {
    let lo = runtime.platform.ALT_MIN_M;
    let hi = runtime.platform.ALT_MAX_M;
    for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        const { rho } = atmosphereAt(mid);
        const buoyancy = (rho * volume_m3 - totalMass_kg) * G;
        if (buoyancy > 0) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}

/**
 * Convert pressure (hPa) to altitude (m) using standard atmosphere.
 * Bisection on atmosphereAt.
 */
export function pressureToAltitude(hPa) {
    const targetPa = hPa * 100;
    let lo = 0, hi = 50000;
    for (let i = 0; i < 50; i++) {
        const mid = (lo + hi) / 2;
        const { P_Pa } = atmosphereAt(mid);
        if (P_Pa > targetPa) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
}

/**
 * Recompute derived values after any platform parameter change.
 * Must be called whenever balloon volume, mass, or ballast changes.
 */
export function recalculateDerived() {
    const p = runtime.platform;
    runtime.balloonRadius_m = Math.cbrt(3 * p.BALLOON_VOLUME_M3 / (4 * Math.PI));
    runtime.balloonArea_m2 = Math.PI * runtime.balloonRadius_m ** 2;

    // Find equilibrium altitudes
    const heavyMass = p.BALLOON_DRY_MASS_KG + p.BALLOON_BALLAST_CAPACITY_KG;
    const lightMass = p.BALLOON_DRY_MASS_KG;
    runtime.altBandLow_m  = Math.max(findEquilibrium(heavyMass, p.BALLOON_VOLUME_M3), p.ALT_MIN_M);
    runtime.altBandHigh_m = Math.min(findEquilibrium(lightMass, p.BALLOON_VOLUME_M3), p.ALT_MAX_M);

    // Build altitude evaluation levels
    // Use floor/ceil to include slightly outside reachable band — the physics
    // will clamp naturally, and including boundary altitudes (e.g., wind layer
    // transitions) is important for correct evaluation.
    const step = runtime.nav.ALTITUDE_STEP_M;
    const lo = Math.floor(runtime.altBandLow_m / step) * step;
    const hi = Math.ceil(runtime.altBandHigh_m / step) * step;
    runtime.altitudeLevels = [];
    for (let a = lo; a <= hi; a += step) {
        runtime.altitudeLevels.push(a);
    }
}

/**
 * config.js — Global constants, feature flags, and default configuration.
 *
 * Pure data module — no dependencies on other modules.
 * All tunable parameters live here. Every module imports from config
 * rather than hardcoding values.
 */

// ── Physical constants ──────────────────────────────────────────────
export const EARTH_RADIUS_M = 6_371_000;
export const G = 9.80665;
export const R_AIR = 287.058;        // J/(kg·K) specific gas constant dry air
export const HELIUM_MOLAR_MASS = 4.0026e-3; // kg/mol

// ── Balloon platform defaults ───────────────────────────────────────
export const DEFAULT_PLATFORM = Object.freeze({
    BALLOON_VOLUME_M3:          500,
    BALLOON_DRY_MASS_KG:        56,
    BALLOON_BALLAST_CAPACITY_KG: 20,
    PUMP_RATE_KG_S:             0.05,
    DRAG_COEFFICIENT:           0.47,  // sphere
    ALT_MIN_M:                  15_000,
    ALT_MAX_M:                  22_000,
    STATION_RADIUS_M:           50_000,
});

// ── Navigator defaults ──────────────────────────────────────────────
export const DEFAULT_NAV = Object.freeze({
    PHYSICS_DT_S:        60,
    NAV_INTERVAL_S:      300,
    LOOK_AHEAD_S:        3600,
    ENERGY_WEIGHT:       0.03,
    HYSTERESIS_M:        900,        // Base hysteresis (scaled by distance)
    ALTITUDE_STEP_M:     125,
    COMMITMENT_THRESHOLD_M: 50,
    COOLDOWN_INTERVALS:  1,
    STALL_THRESHOLD_M:   20,
    STALL_INTERVALS:     2,
});

// ── Navigator feature flags (toggleable modules) ────────────────────
export const NAV_FEATURES = {
    insideRadiusFloat:      true,
    windDirectionOverride:  true,
    approachRateRiding:     true,
    forecastAwareScoring:   false,
    distanceScaledUrgency:  true,
    useCemMpc:              false,   // false=heuristic, true=CEM-MPC, 'hybrid'=best-of-both
    windObserver:           true,    // Collect in-situ wind observations from GPS drift (Q1.1)
    windEkf:                false,   // EKF wind state estimation using observations (Q1.2)
    multiHorizon:           false,   // Multi-horizon trajectory scoring (Q1.3)
    uncertaintyScoring:     false,   // EKF-informed risk/exploration in altitude selection (Q1.4)
    forecastDegradation:    false,   // Simulate forecast errors so EKF can correct them (Q1.5)
    useRl:                  false,   // Use DQN RL controller instead of heuristic (Q2)
    gradientRefinement:     false,   // Test midpoint altitudes between grid points (helps smooth wind, hurts sharp layers)
    recommendationStability: false,  // EMA-based stability filter: hold when recommendations are unstable (helps real data, hurts synthetic)
};

// ── Temporal wind variation defaults ────────────────────────────────
export const DEFAULT_WIND_VARIATION = Object.freeze({
    DIURNAL_AMPLITUDE:     0.15,
    IGW_AMPLITUDE:         3.0,
    IGW_PERIOD_S:          28_800,   // 8 hours
    IGW_VERT_WAVELENGTH_M: 2_000,
    PW_AMPLITUDE:          2.5,
    PW_PERIOD_S:           432_000,  // 5 days
    PW_VERT_WAVELENGTH_M:  5_000,
    NOISE_STD:             1.0,
});

// ── Wind data source types ──────────────────────────────────────────
export const WIND_SOURCE = Object.freeze({
    PRESET:   'preset',    // Built-in synthetic wind presets
    GFS_API:  'gfs_api',   // Live GFS data via Open-Meteo
    MANUAL:   'manual',    // User-entered wind profile
    ERA5:     'era5',      // Historical ERA5 reanalysis
});

// ── Map defaults ────────────────────────────────────────────────────
export const DEFAULT_MAP = Object.freeze({
    CENTER_LAT:  0,
    CENTER_LON:  170,    // Pacific (Loon operating area)
    ZOOM:        6,
    TILE_URL:    'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    TILE_ATTR:   '&copy; OpenStreetMap contributors',
});

// ── Pressure levels ↔ approximate altitudes (standard atmosphere) ───
export const PRESSURE_LEVELS = Object.freeze([
    { hPa: 300, alt_m: 9_164  },
    { hPa: 250, alt_m: 10_363 },
    { hPa: 200, alt_m: 11_784 },
    { hPa: 175, alt_m: 12_631 },
    { hPa: 150, alt_m: 13_608 },
    { hPa: 125, alt_m: 14_795 },
    { hPa: 100, alt_m: 16_180 },
    { hPa:  70, alt_m: 18_442 },
    { hPa:  50, alt_m: 20_576 },
    { hPa:  40, alt_m: 21_835 },
    { hPa:  30, alt_m: 23_849 },
    { hPa:  20, alt_m: 26_481 },
]);

// ── Stratospheric pressure levels to request from Open-Meteo ────────
export const GFS_PRESSURE_LEVELS = [300, 250, 200, 175, 150, 125, 100, 70, 50, 40, 30];

// ── Color scales ────────────────────────────────────────────────────
export const ALTITUDE_COLORS = Object.freeze({
    LOW:    '#3388ff',  // 15 km — blue
    MID:    '#33cc33',  // 17 km — green
    HIGH:   '#ff3333',  // 20 km — red
    MAX:    '#cc33ff',  // 22 km — purple
});

// ── Mutable runtime state (set by UI, read by all modules) ──────────
export const runtime = {
    platform: { ...DEFAULT_PLATFORM },
    nav:      { ...DEFAULT_NAV },
    wind:     { ...DEFAULT_WIND_VARIATION },
    features: { ...NAV_FEATURES },

    // Derived values (recomputed when platform changes)
    balloonRadius_m: 0,
    balloonArea_m2:  0,
    altBandLow_m:    0,
    altBandHigh_m:   0,
    altitudeLevels:  [],
};

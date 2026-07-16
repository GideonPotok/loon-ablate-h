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
export const AIR_MOLAR_MASS    = 28.9647e-3; // kg/mol (dry air)

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

// ── Gas-balloon (helium/sand) variant platform ──────────────────────
// Used only by the gassand env server + js/balloon_gassand.js. Altitude is
// controlled by venting finite helium (→ sink) and dropping finite sand
// (→ rise), instead of the reversible air-ballast pump in DEFAULT_PLATFORM.
// Defaults chosen (via the demo probe) to float ~17 km with both reserves
// comfortably stocked and the envelope seated at its zero-pressure ceiling.
export const DEFAULT_GASSAND = Object.freeze({
    V_ENVELOPE_M3:     1000,     // envelope volume (superpressure ceiling)
    DRY_MASS_KG:       100,      // fixed structure + payload (never changes)
    // Launch is deliberately BALANCED: helium·(M_air/M_he − 1) ≈ DRY + sand, so
    // the envelope is exactly full (zero-pressure ceiling) and net buoyancy ≈ 0.
    // This is the ONLY regime where venting helium descends: with any surplus
    // helium the envelope stays full and venting just sheds mass → the balloon
    // would rise instead. 19.24165 = (DRY+SAND) / (28.9647/4.0026 − 1) = 120/6.236471.
    HELIUM_INIT_KG:    19.24165, // lift gas aboard at launch (only depletes)
    SAND_INIT_KG:      20.0,     // ballast sand aboard at launch (only depletes)
    SAND_CAPACITY_KG:  20.0,     // = SAND_INIT_KG; used to normalise the gauge
    HELIUM_CAPACITY_KG:19.24165, // = HELIUM_INIT_KG; used to normalise the gauge
    M_AIR_KG_MOL:      28.9647e-3,
    M_HE_KG_MOL:       4.0026e-3,
    DRAG_COEFFICIENT:  0.47,     // sphere
    MAX_VV_M_S:        2.5,      // vertical-velocity clamp (matches balloon.js)
    ALT_MIN_M:         15_000,
    ALT_MAX_M:         22_000,
    STATION_RADIUS_M:  50_000,
    // Per-decision release magnitudes (kg), smallest→largest. A tiny release
    // gives a slow drift (v ∝ √amount), a large one a fast climb/dive. Helium
    // amounts are the paired sand amounts ÷ 6.236 (net lift per kg He) so an
    // "up" and its mirror "down" produce matched speeds. All tunable.
    SAND_RELEASE_KG:   [0.005, 0.02, 0.08, 0.30, 1.00],   // → ~0.11 .. 1.56 m/s up
    HELIUM_RELEASE_KG: [0.0008, 0.0032, 0.0128, 0.0481, 0.1604], // → ~0.11 .. 1.55 m/s down
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

    // Gas-balloon variant platform + its derived values (see balloon_gassand.js).
    // balloonRadius_m / altBand*_m here are filled by recalculateGassandDerived().
    gassand:  { ...DEFAULT_GASSAND, balloonRadius_m: 0, altBandLow_m: 0, altBandHigh_m: 0 },

    // Derived values (recomputed when platform changes)
    balloonRadius_m: 0,
    balloonArea_m2:  0,
    altBandLow_m:    0,
    altBandHigh_m:   0,
    altitudeLevels:  [],
};

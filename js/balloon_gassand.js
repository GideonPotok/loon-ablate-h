/**
 * balloon_gassand.js — Zero-pressure gas-balloon physics variant.
 *
 * A deliberate departure from balloon.js (the reversible air-ballast pump used
 * by v1/v2). Here altitude is controlled the way a classic manned gas balloon
 * is flown:
 *
 *   • Drop SAND  → lose mass  → rise.
 *   • Vent HELIUM → lose lift  → sink.
 *
 * Both are FINITE and IRREVERSIBLE: once sand is dropped or gas is vented it is
 * gone. There is no pump to reverse either action. The amount released per
 * decision sets the size of the buoyancy imbalance, and drag turns that
 * imbalance into a (roughly sqrt-scaled) terminal vertical velocity — so a tiny
 * release produces a slow drift and a large release produces a fast climb/dive.
 *
 * Lift model (two regimes, emerging naturally from one `min`):
 *
 *   displaced_air_mass = min( helium_kg · (M_air / M_he),   ρ_air(alt) · V_env )
 *
 *   - Bubble regime (envelope not full, low altitude / plentiful gas):
 *       min picks helium_kg · M_air/M_he. Lift ∝ helium, altitude-independent.
 *       This is the free-expansion regime of a zero-pressure balloon below its
 *       ceiling — venting gas here directly and immediately reduces lift.
 *   - Superpressure regime (envelope full, high altitude / less gas):
 *       min picks ρ_air · V_env. Lift falls with altitude → a STABLE float,
 *       exactly like the fixed-volume Loon model in balloon.js.
 *
 * The crossover (the "ceiling") is the altitude where the gas charge just fills
 * the envelope. A balloon started in the superpressure regime floats where
 * ρ_air·V_env = total mass. Dropping sand lowers total mass → the float rises.
 * Venting helium drops the charge below fill → the balloon falls into the
 * bubble regime and sinks to the lower altitude where the smaller charge fills
 * the envelope again. Both controls move between stable floats; the transient is
 * drag-limited; the distance and peak speed both scale with the amount released.
 *
 * State is immutable — every function returns a new frozen state, matching
 * balloon.js so parallel rollouts stay cheap.
 */
import { G, R_AIR, runtime } from './config.js';
import { atmosphereAt } from './atmosphere.js';

// Net-lift ratio: mass of air displaced per kg of helium when the gas is free
// to expand to ambient P,T (zero-pressure / below ceiling). Equals M_air/M_he.
// A kg of helium therefore displaces ~7.2 kg of air; its NET lift is ~6.2 kg
// once the gas's own weight is subtracted.
export const AIR_PER_HELIUM = () => runtime.gassand.M_AIR_KG_MOL / runtime.gassand.M_HE_KG_MOL;

// ── Balloon state ───────────────────────────────────────────────────

/**
 * Create an immutable gas-balloon state.
 *
 * @param {number} helium_kg — Lift gas currently in the envelope (depletes as vented).
 * @param {number} sand_kg   — Ballast sand still aboard (depletes as dropped).
 */
export function createState(lat, lon, alt_m, vv_m_s = 0, helium_kg = null, sand_kg = null) {
    const gs = runtime.gassand;
    return Object.freeze({
        lat,
        lon,
        alt_m,
        vv_m_s,                                   // vertical velocity (m/s, +up)
        helium_kg: helium_kg ?? gs.HELIUM_INIT_KG,
        sand_kg:   sand_kg   ?? gs.SAND_INIT_KG,
        // Cumulative resources spent — handy for reward shaping / diagnostics.
        helium_vented_kg: 0,
        sand_dropped_kg:  0,
    });
}

// ── Release actuator ────────────────────────────────────────────────

/**
 * Release resources: vent `ventHelium_kg` of gas and/or drop `dropSand_kg` of
 * sand. Each is clamped to what is actually still aboard (you cannot release
 * what you do not have — the finite-resource constraint). Returns a new state
 * plus how much was actually released.
 */
export function applyRelease(state, ventHelium_kg = 0, dropSand_kg = 0) {
    const heOut   = Math.max(0, Math.min(ventHelium_kg, state.helium_kg));
    const sandOut = Math.max(0, Math.min(dropSand_kg,  state.sand_kg));

    const next = Object.freeze({
        ...state,
        helium_kg:        state.helium_kg - heOut,
        sand_kg:          state.sand_kg   - sandOut,
        helium_vented_kg: state.helium_vented_kg + heOut,
        sand_dropped_kg:  state.sand_dropped_kg  + sandOut,
    });
    return { state: next, helium_released_kg: heOut, sand_released_kg: sandOut };
}

// ── Buoyancy ────────────────────────────────────────────────────────

/**
 * Air mass displaced by the lift gas at a given altitude (kg).
 * min() of the free-expansion (bubble) and envelope-full (superpressure) limits.
 */
export function displacedAirMass(helium_kg, rho_air) {
    const bubble       = helium_kg * AIR_PER_HELIUM();     // gas-limited (free expansion)
    const superpressure = rho_air * runtime.gassand.V_ENVELOPE_M3;  // envelope-limited (full)
    return Math.min(bubble, superpressure);
}

/** True when the envelope is full (superpressure regime) at this altitude. */
export function envelopeFull(helium_kg, rho_air) {
    return rho_air * runtime.gassand.V_ENVELOPE_M3 <= helium_kg * AIR_PER_HELIUM();
}

// ── Physics step ────────────────────────────────────────────────────

// The vertical buoyancy/drag system is stiff (restoring force ~0.2 N/m, natural
// period ~2–3 min) so integrating it directly at the 60 s physics tick blows up
// into a ±MAX_VV oscillation — the same latent instability lives in balloon.js
// but v1/v2 hide it behind a bang-bang chase that only reads 5-min-averaged
// altitude. Here the vertical velocity IS the controlled quantity (release
// amount → speed), so the inner integration must be sub-stepped fine enough to
// stay stable (dt·ω < 2 ⟹ dt ≲ 50 s; 5 s leaves wide margin).
const VERT_SUBSTEP_S = 5;

/**
 * Single physics integration step (semi-implicit Euler), pure buoyancy/drag —
 * no actuation. Actuation is applied separately via applyRelease() so a
 * "decision" (metered release) is cleanly decoupled from the sub-steps that
 * integrate its effect.
 *
 * Vertical dynamics are integrated with fine VERT_SUBSTEP_S sub-steps for
 * numerical stability; horizontal advection is applied once over the full dt_s
 * (it does not couple to the stiff vertical mode).
 *
 * @param {object} state — Current balloon state.
 * @param {{ u:number, v:number }} wind — Wind at current position.
 * @param {number} dt_s — Time step (seconds).
 * @returns {object} — New frozen balloon state.
 */
export function physicsStep(state, wind, dt_s) {
    const gs = runtime.gassand;

    const totalMass = gs.DRY_MASS_KG + state.sand_kg + state.helium_kg;
    const area      = Math.PI * gs.balloonRadius_m ** 2;

    // ── Vertical: sub-stepped semi-implicit Euler ──
    let alt = state.alt_m;
    let vv  = state.vv_m_s;
    const nSub = Math.max(1, Math.round(dt_s / VERT_SUBSTEP_S));
    const sub  = dt_s / nSub;
    for (let i = 0; i < nSub; i++) {
        const atm = atmosphereAt(alt);
        // Buoyancy: (displaced air mass − total mass) · g
        const buoyancy = (displacedAirMass(state.helium_kg, atm.rho) - totalMass) * G;
        // Drag: −0.5 · Cd · ρ · A · |v| · v  (opposes velocity)
        const drag = -0.5 * gs.DRAG_COEFFICIENT * atm.rho * area * Math.abs(vv) * vv;

        vv += ((buoyancy + drag) / totalMass) * sub;
        vv  = Math.max(-gs.MAX_VV_M_S, Math.min(gs.MAX_VV_M_S, vv));

        alt += vv * sub;
        if (alt <= gs.ALT_MIN_M) { alt = gs.ALT_MIN_M; vv = 0; break; }
        if (alt >= gs.ALT_MAX_M) { alt = gs.ALT_MAX_M; vv = 0; break; }
    }

    // ── Horizontal advection by wind (great-circle approximation) ──
    const dLat = (wind.v * dt_s / 6_371_000) * (180 / Math.PI);
    const dLon = (wind.u * dt_s / (6_371_000 * Math.cos(state.lat * Math.PI / 180))) *
                 (180 / Math.PI);

    return Object.freeze({
        ...state,
        lat:    state.lat + dLat,
        lon:    state.lon + dLon,
        alt_m:  alt,
        vv_m_s: vv,
    });
}

/**
 * Recompute gas-balloon derived values. Call once after any gassand platform
 * change (analogue of atmosphere.recalculateDerived for DEFAULT_PLATFORM).
 *
 * The navigable float band is the span of equilibrium altitudes reachable with
 * the initial gas charge:
 *   - altBandLow_m : float with ALL sand still aboard   (heaviest → lowest).
 *   - altBandHigh_m: float with ALL sand dropped         (lightest → highest).
 * (Venting helium can push below altBandLow but that is irreversible.)
 */
export function recalculateGassandDerived() {
    const gs = runtime.gassand;
    gs.balloonRadius_m = Math.cbrt(3 * gs.V_ENVELOPE_M3 / (4 * Math.PI));

    // Low bound: the ceiling of the initial gas charge. At balanced launch the
    // balloon floats here (envelope just full); it cannot passively rise above
    // it without shedding sand. Depends only on helium, not on mass.
    gs.altBandLow_m  = Math.max(findCeiling(gs.HELIUM_INIT_KG), gs.ALT_MIN_M);
    // High bound: the stable superpressure float once ALL sand is dropped
    // (lightest → highest reachable float).
    const lightMass  = gs.DRY_MASS_KG + gs.HELIUM_INIT_KG;
    gs.altBandHigh_m = Math.min(findGasEquilibrium(gs.HELIUM_INIT_KG, lightMass), gs.ALT_MAX_M);
}

/**
 * Ceiling altitude for a gas charge: the altitude at which the freely-expanding
 * gas exactly fills the envelope (ρ_air·V_env = helium·M_air/M_he). Above it the
 * envelope is full (superpressure); below it the gas is a smaller bubble.
 */
export function findCeiling(helium_kg) {
    const gs = runtime.gassand;
    const target = helium_kg * AIR_PER_HELIUM();   // = ρ_ceiling · V_env
    let lo = gs.ALT_MIN_M, hi = gs.ALT_MAX_M;
    for (let i = 0; i < 60; i++) {
        const mid = (lo + hi) / 2;
        const { rho } = atmosphereAt(mid);
        if (rho * gs.V_ENVELOPE_M3 > target) lo = mid; else hi = mid;  // too dense → go higher
    }
    return (lo + hi) / 2;
}

/**
 * Equilibrium (float) altitude for the current gas charge and total mass, via
 * bisection on net buoyancy. Mirrors atmosphere.findEquilibrium but uses the
 * two-regime displaced-air model.
 */
export function findGasEquilibrium(helium_kg, totalMass_kg) {
    const gs = runtime.gassand;
    let lo = gs.ALT_MIN_M, hi = gs.ALT_MAX_M;
    for (let i = 0; i < 60; i++) {
        const mid = (lo + hi) / 2;
        const { rho } = atmosphereAt(mid);
        const buoyancy = (displacedAirMass(helium_kg, rho) - totalMass_kg) * G;
        if (buoyancy > 0) lo = mid; else hi = mid;   // buoyant → search higher
    }
    return (lo + hi) / 2;
}

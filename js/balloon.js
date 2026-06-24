/**
 * balloon.js — Balloon physics, dynamics, and altitude control.
 *
 * Semi-implicit Euler integration. State is immutable — step() returns
 * a new state. This enables parallel rollouts for planning.
 */
import { G, runtime } from './config.js';
import { atmosphereAt } from './atmosphere.js';

// ── Balloon state ───────────────────────────────────────────────────

/**
 * Create an immutable balloon state.
 */
export function createState(lat, lon, alt_m, vv_m_s = 0, ballast_kg = null) {
    return Object.freeze({
        lat,
        lon,
        alt_m,
        vv_m_s,          // vertical velocity (m/s, positive = up)
        ballast_kg: ballast_kg ?? runtime.platform.BALLOON_BALLAST_CAPACITY_KG / 2,
        energy_used_j: 0,
    });
}

// ── Altitude Control System (ACS) ───────────────────────────────────
// Actions: 1 = ASCEND (pump ballast out), -1 = DESCEND (pump ballast in), 0 = FLOAT

/**
 * Apply a single ACS action for dt seconds.
 * Returns the new ballast_kg and energy cost.
 */
export function applyAction(action, ballast_kg, dt_s, alt_m) {
    const p = runtime.platform;
    let newBallast = ballast_kg;
    let energyCost = 0;

    if (action === 1) {
        // ASCEND: release ballast (reduce mass → rise)
        const released = Math.min(p.PUMP_RATE_KG_S * dt_s, ballast_kg);
        newBallast -= released;
        energyCost = released * G * alt_m * 0.001;  // kJ
    } else if (action === -1) {
        // DESCEND: take on ballast (increase mass → sink)
        const taken = Math.min(p.PUMP_RATE_KG_S * dt_s, p.BALLOON_BALLAST_CAPACITY_KG - ballast_kg);
        newBallast += taken;
        energyCost = taken * G * alt_m * 0.001;
    }

    return { ballast_kg: newBallast, energy_cost: energyCost };
}

// ── Physics step ────────────────────────────────────────────────────

/**
 * Single physics integration step. Semi-implicit Euler:
 * 1. Compute forces at current position
 * 2. Update velocity
 * 3. Update position using new velocity
 *
 * @param {object} state — Current balloon state
 * @param {number} action — ACS action (-1, 0, 1)
 * @param {{ u: number, v: number }} wind — Wind at current position
 * @param {number} dt_s — Time step (seconds)
 * @returns {object} — New balloon state (frozen)
 */
export function physicsStep(state, action, wind, dt_s) {
    const p = runtime.platform;

    // 1. Apply ACS action to ballast
    const acs = applyAction(action, state.ballast_kg, dt_s, state.alt_m);

    // 2. Compute forces
    const totalMass = p.BALLOON_DRY_MASS_KG + acs.ballast_kg;
    const atm = atmosphereAt(state.alt_m);

    // Buoyancy: (displaced air mass - balloon mass) × g
    const buoyancy = (atm.rho * p.BALLOON_VOLUME_M3 - totalMass) * G;

    // Drag: 0.5 × Cd × ρ × A × v²  (opposes velocity)
    const area = runtime.balloonArea_m2;
    const drag = -0.5 * p.DRAG_COEFFICIENT * atm.rho * area *
                 Math.abs(state.vv_m_s) * state.vv_m_s;

    // 3. Semi-implicit Euler: velocity first
    const accel = (buoyancy + drag) / totalMass;
    let newVV = state.vv_m_s + accel * dt_s;

    // Clamp vertical velocity
    newVV = Math.max(-2.5, Math.min(2.5, newVV));

    // 4. Update altitude using new velocity
    let newAlt = state.alt_m + newVV * dt_s;
    newAlt = Math.max(p.ALT_MIN_M, Math.min(p.ALT_MAX_M, newAlt));

    // If clamped, zero velocity
    if (newAlt === p.ALT_MIN_M || newAlt === p.ALT_MAX_M) newVV = 0;

    // 5. Horizontal advection by wind (great-circle approximation)
    const dLat = (wind.v * dt_s / 6_371_000) * (180 / Math.PI);
    const dLon = (wind.u * dt_s / (6_371_000 * Math.cos(state.lat * Math.PI / 180))) *
                 (180 / Math.PI);

    return Object.freeze({
        lat:           state.lat + dLat,
        lon:           state.lon + dLon,
        alt_m:         newAlt,
        vv_m_s:        newVV,
        ballast_kg:    acs.ballast_kg,
        energy_used_j: state.energy_used_j + acs.energy_cost,
    });
}

/**
 * Multi-step forward simulation (for look-ahead planning).
 * Returns array of states.
 */
export function rollout(state, actions, getWindFn, dt_s) {
    const trajectory = [state];
    let s = state;
    let time_s = 0;
    for (const action of actions) {
        const wind = getWindFn(s.alt_m, time_s);
        s = physicsStep(s, action, wind, dt_s);
        trajectory.push(s);
        time_s += dt_s;
    }
    return trajectory;
}

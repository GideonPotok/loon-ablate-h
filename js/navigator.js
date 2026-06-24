/**
 * navigator.js — Station-keeping altitude controller.
 *
 * Modular decision pipeline with two planning backends:
 *   - Heuristic: Fast single-altitude evaluation (default, good on tropical)
 *   - CEM-MPC: Sequence optimization via Cross-Entropy Method (good on strong-shear)
 *   - Hybrid: Runs both, picks the better projected outcome
 *
 * The wind accessor is injected via `getWindFn`, so the navigator works
 * with any wind source (presets, real forecasts, etc.).
 *
 * Decision stages:
 *   1. insideRadiusFloat    — coast when projected to stay on-station
 *   2. windDirectionOverride — emergency altitude switch when drifting away
 *   3. distanceScaledUrgency — adjust thresholds by distance ratio
 *   4. approachRateRiding    — ride favorable wind at current altitude
 *   5. altitude evaluation   — heuristic, CEM-MPC, or hybrid
 */
import { runtime } from './config.js';
import { haversine, bearingFlat, windApproachRate } from './geo.js';
import { physicsStep } from './balloon.js';
import { cemPlan } from './cem_mpc.js';

// ── Persistent state (cleared on simulation reset) ──────────────────

const _navMemory = {
    prevApproachRate: null,
    prevTime: 0,
    emaAlt: null,           // Exponential moving average of recommended altitudes
    emaVariance: 0,         // Variance of altitude recommendations (stability metric)
    recentAlts: [],         // Recent altitude recommendations for variance tracking
};

export function resetNavMemory() {
    _navMemory.prevApproachRate = null;
    _navMemory.prevTime = 0;
    _navMemory.emaAlt = null;
    _navMemory.emaVariance = 0;
    _navMemory.recentAlts = [];
}

// ── Trajectory projection ───────────────────────────────────────────

/**
 * Simulate forward from current state toward targetAlt for a given duration.
 * Returns trajectory statistics for scoring.
 *
 * Accounts for wind exposure during altitude transit — the balloon passes
 * through intermediate altitudes with potentially unfavorable wind.
 */
function projectTrajectory(state, targetAlt, getWindFn, time_s, targetLat, targetLon, durationS) {
    const nav = runtime.nav;
    const lookAheadS = durationS || nav.LOOK_AHEAD_S;
    const lookAheadSteps = Math.ceil(lookAheadS / nav.PHYSICS_DT_S);
    let s = { ...state };
    let t = time_s;
    const action = targetAlt > state.alt_m ? 1 : (targetAlt < state.alt_m ? -1 : 0);

    let minDist = haversine(s.lat, s.lon, targetLat, targetLon);
    let sumDist = 0;

    for (let i = 0; i < lookAheadSteps; i++) {
        const w = getWindFn(s.alt_m, t);
        const a = (Math.abs(s.alt_m - targetAlt) < 100) ? 0 : action;
        s = { ...physicsStep(s, a, w, nav.PHYSICS_DT_S) };
        t += nav.PHYSICS_DT_S;
        const d = haversine(s.lat, s.lon, targetLat, targetLon);
        minDist = Math.min(minDist, d);
        sumDist += d;
    }

    const finalDist = haversine(s.lat, s.lon, targetLat, targetLon);
    const avgDist = sumDist / lookAheadSteps;
    return { finalDist, minDist, avgDist };
}

/**
 * Find the altitude with the best time-averaged approach rate.
 */
function findBestApproachAlt(getWindFn, time_s, brng) {
    let bestAlt = runtime.altitudeLevels[0];
    let bestRate = -Infinity;

    const samples = [0, 600, 1200];
    for (const alt of runtime.altitudeLevels) {
        let totalRate = 0;
        for (const dt of samples) {
            const w = getWindFn(alt, time_s + dt);
            totalRate += windApproachRate(w.u, w.v, brng);
        }
        const avgRate = totalRate / samples.length;
        if (avgRate > bestRate) { bestRate = avgRate; bestAlt = alt; }
    }
    return { alt: bestAlt, rate: bestRate };
}

/**
 * Approach rate averaged over a time window for noise robustness.
 */
function avgApproachRate(getWindFn, alt_m, time_s, brng, windowS = 900) {
    let total = 0;
    const steps = 4;
    const dt = windowS / steps;
    for (let i = 0; i <= steps; i++) {
        const w = getWindFn(alt_m, time_s + i * dt);
        total += windApproachRate(w.u, w.v, brng);
    }
    return total / (steps + 1);
}

// ── Multi-horizon scoring (Q1.3) ────────────────────────────────────

/**
 * Multi-horizon scoring configuration.
 * Each horizon has a duration and weight. Shorter horizons get more weight
 * because near-term predictions are more reliable.
 */
const MULTI_HORIZON = [
    { duration_s:  900, weight: 0.40 },  // 15 min — high confidence
    { duration_s: 2700, weight: 0.35 },  // 45 min — medium confidence
    { duration_s: 7200, weight: 0.25 },  // 2 hours — lower confidence
];

/**
 * Score an altitude using multiple projection horizons.
 * Returns a blended score that captures both short-term safety
 * and long-term opportunity.
 */
function multiHorizonScore(state, alt, getWindFn, time_s, targetLat, targetLon) {
    let blendedFinal = 0;
    let blendedAvg = 0;

    for (const { duration_s, weight } of MULTI_HORIZON) {
        const proj = projectTrajectory(state, alt, getWindFn, time_s, targetLat, targetLon, duration_s);
        blendedFinal += proj.finalDist * weight;
        blendedAvg += proj.avgDist * weight;
    }

    return { finalDist: blendedFinal, avgDist: blendedAvg };
}

// ── Heuristic altitude evaluation ───────────────────────────────────

/**
 * Score all candidate altitudes via trajectory projection.
 * Returns the best altitude and its projected distance.
 *
 * When multiHorizon feature is enabled, uses blended multi-horizon scoring.
 * Otherwise uses the original single-horizon projection.
 */
function heuristicEval(state, getWindFn, time_s, targetLat, targetLon, brng, distRatio, getUncertaintyFn) {
    const nav = runtime.nav;
    const feat = runtime.features;
    const dist = haversine(state.lat, state.lon, targetLat, targetLon);
    let bestAlt = state.alt_m;
    let bestScore = Infinity;
    let bestProjDist = dist;

    for (const alt of runtime.altitudeLevels) {
        let projFinal, projAvg;

        if (feat.multiHorizon) {
            const mh = multiHorizonScore(state, alt, getWindFn, time_s, targetLat, targetLon);
            projFinal = mh.finalDist;
            projAvg = mh.avgDist;
        } else {
            const proj = projectTrajectory(state, alt, getWindFn, time_s, targetLat, targetLon);
            projFinal = proj.finalDist;
            projAvg = proj.avgDist;
        }

        const altDelta = Math.abs(alt - state.alt_m);
        const energyCost = altDelta * nav.ENERGY_WEIGHT;

        let score = projFinal * 0.6 + projAvg * 0.4 + energyCost;

        if (feat.forecastAwareScoring) {
            const altApproach = avgApproachRate(getWindFn, alt, time_s, brng);
            const urgencyMult = Math.max(0.5, Math.min(3.0, distRatio));
            score -= altApproach * 500 * urgencyMult;
        }

        // Q1.4: EKF-informed uncertainty scoring
        // Penalize altitudes with high wind uncertainty — projection is unreliable there.
        // Add exploration bonus for unvisited altitudes to encourage gathering data.
        if (feat.uncertaintyScoring && getUncertaintyFn) {
            const sigma = getUncertaintyFn(alt);

            // Risk penalty: high uncertainty means projection could be off by ±sigma × dt
            // Scale by distance — when far from target, risk-aversion matters more
            const riskWeight = 2000 * Math.min(2.0, distRatio);
            score += sigma * riskWeight;

            // Exploration bonus: small incentive to visit high-uncertainty altitudes
            // Only applies when within station radius (safe to explore)
            if (dist < runtime.platform.STATION_RADIUS_M && sigma > 2.0) {
                const explorationBonus = 500 * (sigma - 2.0);
                score -= explorationBonus;
            }
        }

        if (score < bestScore) {
            bestScore = score;
            bestAlt = alt;
            bestProjDist = projFinal;
        }
    }

    // Gradient refinement: test midpoints around the best altitude
    // On smooth wind fields, the optimal altitude may lie between grid points
    if (feat.gradientRefinement) {
        const halfStep = Math.floor(nav.ALTITUDE_STEP_M / 2);
        const refineCandidates = [bestAlt - halfStep, bestAlt + halfStep];
        for (const alt of refineCandidates) {
            if (alt < runtime.altBandLow_m || alt > runtime.altBandHigh_m) continue;
            if (Math.abs(alt - bestAlt) < 10) continue;
            const proj = projectTrajectory(state, alt, getWindFn, time_s, targetLat, targetLon);
            const altDelta = Math.abs(alt - state.alt_m);
            const energyCost = altDelta * nav.ENERGY_WEIGHT;
            const score = proj.finalDist * 0.6 + proj.avgDist * 0.4 + energyCost;
            if (score < bestScore) {
                bestScore = score;
                bestAlt = alt;
                bestProjDist = proj.finalDist;
            }
        }
    }

    return { bestAlt, bestScore, bestProjDist };
}

// ── Main decision function ──────────────────────────────────────────

export function chooseAction(state, getWindFn, time_s, targetLat, targetLon, getUncertaintyFn = null) {
    const feat = runtime.features;
    const nav = runtime.nav;
    const dist = haversine(state.lat, state.lon, targetLat, targetLon);
    const brng = bearingFlat(state.lat, state.lon, targetLat, targetLon);
    const radius = runtime.platform.STATION_RADIUS_M;
    const distRatio = dist / radius;

    // ── Stage 1: Inside-radius coast ────────────────────────────────
    if (feat.insideRadiusFloat && dist < radius) {
        const proj = projectTrajectory(state, state.alt_m, getWindFn, time_s, targetLat, targetLon);
        const coastThreshold = radius * (0.9 - 0.3 * distRatio);
        if (proj.finalDist < coastThreshold && proj.avgDist < radius * 0.8) {
            return {
                action: 0, targetAlt: state.alt_m, reason: 'On station — coasting',
                projectedDist: proj.finalDist, stage: 'insideRadiusFloat',
            };
        }
    }

    // ── Stage 2: Wind direction override ────────────────────────────
    if (feat.windDirectionOverride && dist > radius * 0.7) {
        let curAvgApproach = 0;
        const checkTimes = [0, 300, 600];
        for (const dt of checkTimes) {
            const w = getWindFn(state.alt_m, time_s + dt);
            curAvgApproach += windApproachRate(w.u, w.v, brng);
        }
        curAvgApproach /= checkTimes.length;

        const overrideThreshold = distRatio > 1.5 ? -0.5 : -2.0;

        if (curAvgApproach < overrideThreshold) {
            const best = findBestApproachAlt(getWindFn, time_s, brng);
            if (best.rate > Math.max(1.0, -curAvgApproach * 0.3)) {
                const action = best.alt > state.alt_m ? 1 : -1;
                return {
                    action, targetAlt: best.alt,
                    reason: `Wind override: ${curAvgApproach.toFixed(1)} m/s away → ${best.alt}m (${best.rate.toFixed(1)} m/s toward)`,
                    projectedDist: dist, stage: 'windDirectionOverride',
                };
            }
        }
    }

    // ── Stage 3: Distance-scaled urgency ────────────────────────────
    let approachThreshold = 0.05;
    let hysteresisScale = 1.0;

    if (feat.distanceScaledUrgency) {
        if (distRatio > 2.0) {
            approachThreshold = 0.20;
            hysteresisScale = 0.1;
        } else if (distRatio > 1.5) {
            approachThreshold = 0.15;
            hysteresisScale = 0.3;
        } else if (distRatio > 1.0) {
            const t = (distRatio - 1.0) / 0.5;
            approachThreshold = 0.05 + t * 0.10;
            hysteresisScale = 1.0 - t * 0.7;
        }
    }

    // ── Stage 4: Approach-rate riding ───────────────────────────────
    if (feat.approachRateRiding) {
        const proj = projectTrajectory(state, state.alt_m, getWindFn, time_s, targetLat, targetLon);
        if (proj.finalDist < dist * (1 - approachThreshold) && proj.avgDist < dist) {
            return {
                action: 0, targetAlt: state.alt_m,
                reason: `Riding approach wind: proj ${(proj.finalDist/1000).toFixed(0)} km (avg ${(proj.avgDist/1000).toFixed(0)} km)`,
                projectedDist: proj.finalDist, stage: 'approachRateRiding',
            };
        }
    }

    // ── Stage 5: Altitude evaluation ────────────────────────────────

    let bestAlt, bestProjDist, stage;

    if (feat.useCemMpc === 'hybrid') {
        // Hybrid mode: run both heuristic and CEM-MPC, pick the better one
        const heur = heuristicEval(state, getWindFn, time_s, targetLat, targetLon, brng, distRatio, getUncertaintyFn);
        const cem = cemPlan(state, getWindFn, time_s, targetLat, targetLon);

        if (cem.projectedDist < heur.bestProjDist * 0.95) {
            // CEM-MPC is meaningfully better (5%+ improvement)
            bestAlt = cem.targetAlt;
            bestProjDist = cem.projectedDist;
            stage = 'cem_mpc';
        } else {
            bestAlt = heur.bestAlt;
            bestProjDist = heur.bestProjDist;
            stage = 'altitudeEvaluation';
        }
    } else if (feat.useCemMpc === true) {
        // Pure CEM-MPC mode
        const cem = cemPlan(state, getWindFn, time_s, targetLat, targetLon);
        bestAlt = cem.targetAlt;
        bestProjDist = cem.projectedDist;
        stage = 'cem_mpc';
    } else {
        // Pure heuristic mode (default)
        const heur = heuristicEval(state, getWindFn, time_s, targetLat, targetLon, brng, distRatio, getUncertaintyFn);
        bestAlt = heur.bestAlt;
        bestProjDist = heur.bestProjDist;
        stage = 'altitudeEvaluation';
    }

    // Track altitude recommendation stability (EMA)
    if (feat.recommendationStability) {
        const alpha = 0.3;
        if (_navMemory.emaAlt === null) {
            _navMemory.emaAlt = bestAlt;
        } else {
            _navMemory.emaAlt = alpha * bestAlt + (1 - alpha) * _navMemory.emaAlt;
        }
        _navMemory.recentAlts.push(bestAlt);
        if (_navMemory.recentAlts.length > 6) _navMemory.recentAlts.shift();

        // Compute variance of recent recommendations
        if (_navMemory.recentAlts.length >= 3) {
            const mean = _navMemory.recentAlts.reduce((a, b) => a + b, 0) / _navMemory.recentAlts.length;
            const variance = _navMemory.recentAlts.reduce((a, b) => a + (b - mean) ** 2, 0) / _navMemory.recentAlts.length;
            _navMemory.emaVariance = variance;

            // If recommendations are unstable (high variance) and we're not in crisis,
            // hold current altitude — no altitude is clearly better
            const isUnstable = Math.sqrt(variance) > nav.ALTITUDE_STEP_M * 2;
            const notCrisis = distRatio < 2.5;
            if (isUnstable && notCrisis) {
                const stayProj = projectTrajectory(state, state.alt_m, getWindFn, time_s, targetLat, targetLon);
                // Only hold if staying isn't catastrophically worse
                if (stayProj.finalDist < bestProjDist * 1.3) {
                    return {
                        action: 0, targetAlt: state.alt_m,
                        reason: `Stability hold: variance=${Math.sqrt(variance).toFixed(0)}m, recent=[${_navMemory.recentAlts.map(a => Math.round(a)).join(',')}]`,
                        projectedDist: stayProj.finalDist, stage: 'stabilityHold',
                    };
                }
            }
        }
    }

    // Hysteresis — only switch if improvement is meaningful
    const hysteresis = nav.HYSTERESIS_M * hysteresisScale;
    if (Math.abs(bestAlt - state.alt_m) < hysteresis) {
        const stayProj = projectTrajectory(state, state.alt_m, getWindFn, time_s, targetLat, targetLon);
        const minImprovement = 3000 * hysteresisScale;
        const improvement = stayProj.finalDist - bestProjDist;
        if (improvement < minImprovement && stayProj.finalDist < bestProjDist * 1.08) {
            return {
                action: 0, targetAlt: state.alt_m,
                reason: `Hysteresis hold: best ${bestAlt}m only ${(improvement/1000).toFixed(1)} km better`,
                projectedDist: stayProj.finalDist, stage: 'hysteresis',
            };
        }
    }

    const action = bestAlt > state.alt_m ? 1 : (bestAlt < state.alt_m ? -1 : 0);
    return {
        action, targetAlt: bestAlt,
        reason: `${stage === 'cem_mpc' ? 'CEM-MPC' : 'Alt eval'}: → ${bestAlt}m (proj ${(bestProjDist/1000).toFixed(0)} km)`,
        projectedDist: bestProjDist, stage,
    };
}

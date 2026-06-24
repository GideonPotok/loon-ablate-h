/**
 * rl_trainer.js — Training harness for DQN balloon station-keeping (Q2.3).
 *
 * Runs headless training episodes using the balloon physics engine directly
 * (no Simulator dependency — avoids coupling to the UI event system).
 *
 * CRITICAL: The training pipeline uses the same sensing stack as inference.
 * The agent sees EKF-estimated wind (not truth wind) during training:
 *   1. ForecastDegrader provides a biased/stale forecast as the prior
 *   2. WindEKF is initialized from the degraded forecast
 *   3. Each physics step: GPS drift → WindObserver → EKF update
 *   4. extractState receives EKF wind + uncertainty (not truth)
 *
 * This ensures no train/test distribution mismatch — the agent learns to
 * navigate with the same partial, uncertain information it has at inference.
 *
 * Training loop:
 *   1. Reset balloon state at random offset from target
 *   2. At each NAV_INTERVAL, extract state → select action → apply
 *   3. After NAV_INTERVAL of physics steps, compute reward from new position
 *   4. Store transition (s, a, r, s', done) in replay buffer
 *   5. Train mini-batch from buffer
 *   6. After episode ends, decay epsilon
 *
 * Designed for headless Node.js execution. No DOM or browser dependencies.
 */

import { runtime } from './config.js';
import { haversine } from './geo.js';
import { getWind, getBaseWind, WIND_PRESETS } from './wind.js';
import { createState, physicsStep } from './balloon.js';
import { recalculateDerived } from './atmosphere.js';
import { DQNAgent, actionFromIndex, targetAltFromIndex } from './rl_agent.js';
import { WindObservationStore } from './wind_observer.js';
import { WindEKF } from './wind_ekf.js';
import { ForecastDegrader } from './wind_degrader.js';

/**
 * Convert a target altitude to a per-step ACS command (bang-bang chase).
 * Returns -1, 0, or +1 depending on whether we need to descend, hold, or
 * ascend to reach targetAlt_m from currentAlt_m. The 'tolerance_m' band is
 * the deadzone where we float (matches navigator's COMMITMENT_THRESHOLD_M).
 */
function chaseAction(currentAlt_m, targetAlt_m, tolerance_m = 50) {
    const delta = targetAlt_m - currentAlt_m;
    if (Math.abs(delta) < tolerance_m) return 0;
    return delta > 0 ? 1 : -1;
}

// ── Training configuration ───────────────────────────────────────────

export const TRAIN_DEFAULTS = Object.freeze({
    EPISODE_DURATION_S:   3600 * 2,    // 2 hours per episode (more diverse training)
    NAV_INTERVAL_S:       300,          // Match navigator: decide every 5 min
    PHYSICS_DT_S:         60,           // Match physics step
    EPISODES:             500,          // Total training episodes (overridden by CURRICULUM)
    TRAIN_BATCHES_PER_STEP: 2,         // Batches of training per nav step
    PRESET:               'tropical',   // Wind preset for training
    TARGET_LAT:           0,
    TARGET_LON:           170,
    SPAWN_OFFSET_KM:      30,           // Randomized initial offset from target
    SPAWN_ALT_MIN_M:      16800,        // Random initial altitude range
    SPAWN_ALT_MAX_M:      18200,
    EVAL_EVERY:           50,           // Evaluate every N episodes
    EVAL_DURATION_S:      3600 * 24,    // 24h evaluation episodes
    EVAL_RUNS:            5,            // Average N eval episodes per checkpoint (reduces noise)
    SEED:                 42,
    DEGRADER_SEED_OFFSET: 7777,        // Offset for per-episode degrader reseeding

    // Domain randomisation (Tobin 2017, Pham 2023). When DOMAIN_RAND_ENABLED
    // is true, each episode draws a random scale factor for the degrader's
    // bias and noise sigmas, plus a random forecast lag. This widens the
    // training distribution beyond a fixed σ calibration and is the cheapest
    // sim-to-real upgrade (Pham et al. 2023 NeurIPS ML4PS workshop reports
    // +6 pp TWR-50 on BLE).
    //
    // Sigma scale is sampled log-uniformly in [DOMAIN_RAND_SIGMA_MIN, MAX].
    // Lag is sampled uniformly in [0, DOMAIN_RAND_MAX_LAG_S].
    DOMAIN_RAND_ENABLED:    false,
    DOMAIN_RAND_SIGMA_MIN:  0.5,    // 0.5× calibrated value
    DOMAIN_RAND_SIGMA_MAX:  2.0,    // 2.0× calibrated value
    DOMAIN_RAND_MAX_LAG_S:  21600,  // 6 hours (cap)
    // Curriculum learning schedule.  null = fixed EPISODE_DURATION_S throughout.
    // Array of {episodes, duration_s, label?} tiers, applied in order.
    // Total episode count is derived from the sum — EPISODES is ignored when set.
    // Example:
    //   CURRICULUM: [
    //       { episodes: 200, duration_s: 3600 * 2,  label: '2h'  },
    //       { episodes: 150, duration_s: 3600 * 6,  label: '6h'  },
    //       { episodes: 100, duration_s: 3600 * 12, label: '12h' },
    //       { episodes: 100, duration_s: 3600 * 24, label: '24h' },
    //   ]
    CURRICULUM:           null,
    // ERA5 real wind archive.  null = use synthetic PRESET.
    // When set, each episode samples a random location + start time from the archive.
    // Must be a loaded WindArchive instance (see tactical/js/wind_archive.js).
    // TARGET_LAT / TARGET_LON are overridden per-episode by the sampled grid point.
    WIND_ARCHIVE:         null,

    // Approach-rate reward shaping (from Overcooked paper, §3.3).
    // Adds a bonus proportional to how much the balloon closed distance toward
    // the target in the last nav interval, normalized by the station radius.
    // The bonus anneals from APPROACH_SHAPING to 0 over the first 80% of nav
    // steps so the agent transitions from shaped to sparse-like learning.
    // Set to 0 to disable. Only applied during training (not eval).
    APPROACH_SHAPING:     0.5,
});

// ── Seeded PRNG for reproducible spawn positions ─────────────────────

function makeRng(seed) {
    let state = (seed >>> 0) || 1;
    return function rand() {
        state ^= state << 13;
        state ^= state >>> 17;
        state ^= state << 5;
        state = state >>> 0;
        return (state & 0x7FFFFFFF) / 0x80000000;
    };
}

// ── Sensing stack: creates per-episode wind observer + EKF ──────────

/**
 * Create the Q1 sensing pipeline for one episode.
 *
 * Returns functions that mirror what the Simulator provides:
 *   - truthWindFn: the actual wind (drives physics)
 *   - bestWindFn:  EKF-filtered wind (what the agent sees)
 *   - uncertaintyFn: EKF uncertainty at any altitude
 *   - stepSensing: call each physics step to feed GPS drift into EKF
 *
 * @param {{ truthWindFn: Function, baseWindFn: Function }} windFns
 *   - truthWindFn(alt_m, time_s) → {u,v}  (drives physics)
 *   - baseWindFn(alt_m)          → {u,v}  (time-0 snapshot, used for EKF prior)
 * @param {number} degraderSeed - Seed for forecast degradation (varies per episode)
 * @returns {Object} Sensing pipeline functions
 */
function createSensingStack({ truthWindFn, baseWindFn }, degraderSeed, drOpts = null) {

    // Forecast degrader: adds bias + noise + staleness to the forecast.
    // Optional domain-randomisation overrides (per-episode sampled in the trainer).
    const degraderOpts = { SEED: degraderSeed };
    if (drOpts) {
        if (drOpts.biasSigma  != null) degraderOpts.BIAS_SIGMA  = drOpts.biasSigma;
        if (drOpts.noiseSigma != null) degraderOpts.NOISE_SIGMA = drOpts.noiseSigma;
        if (drOpts.lag_s      != null) {
            degraderOpts.STALENESS_MODE = 'lagged';
            degraderOpts.LAG_S = drOpts.lag_s;
        }
    }
    const degrader = new ForecastDegrader(truthWindFn, baseWindFn, degraderOpts);

    // Degraded forecast function (what the balloon receives via uplink)
    const forecastWindFn = (alt_m, time_s) => degrader.getForecastWind(alt_m, time_s);

    // Wind observer: stores GPS drift observations
    const observer = new WindObservationStore();

    // EKF: fuses GPS observations with degraded forecast prior
    const ekf = new WindEKF();

    // Initialize EKF altitude levels from runtime
    // (recalculateDerived must have been called before this)
    ekf.initialize((alt_m) => forecastWindFn(alt_m, 0));

    /**
     * Get best wind estimate: uses EKF when confident, blends with
     * degraded forecast otherwise. Mirrors Simulator.getBestWindAt().
     */
    function bestWindFn(alt_m, time_s) {
        if (!ekf.initialized) return forecastWindFn(alt_m, time_s);

        const ekfWind = ekf.getWind(alt_m);
        const ekfSigma = ekf.getUncertainty(alt_m);

        // If EKF has reasonable confidence, use it
        if (ekfSigma < 5.0) {
            return ekfWind;
        }

        // Blend by inverse variance
        const rawWind = forecastWindFn(alt_m, time_s);
        const rawSigma = degrader.getUncertainty(alt_m);
        const wEkf = 1 / (ekfSigma * ekfSigma);
        const wRaw = 1 / (rawSigma * rawSigma);
        const wTotal = wEkf + wRaw;
        return {
            u: (ekfWind.u * wEkf + rawWind.u * wRaw) / wTotal,
            v: (ekfWind.v * wEkf + rawWind.v * wRaw) / wTotal,
        };
    }

    /**
     * Get EKF uncertainty at an altitude.
     */
    function uncertaintyFn(alt_m) {
        if (!ekf.initialized) return 10.0;
        return ekf.getUncertainty(alt_m);
    }

    /**
     * Feed a physics step into the sensing pipeline.
     * Must be called each step with current and previous balloon states.
     *
     * @param {Object} state - Current balloon state
     * @param {Object|null} prevState - Previous balloon state (null on first step)
     * @param {number} dt_s - Physics time step
     * @param {number} time_s - Current simulation time
     */
    function stepSensing(state, prevState, dt_s, time_s) {
        // EKF prediction (advance uncertainty)
        ekf.predict(dt_s);

        // GPS drift observation (requires two successive states)
        if (prevState) {
            const obs = observer.observe(state, prevState, dt_s, time_s);
            if (obs) {
                // Feed GPS-derived wind into EKF — this is the key step:
                // observation at current altitude propagates to nearby
                // altitudes via vertical correlation in the Kalman gain
                ekf.update(obs.alt_m, obs.u_obs, obs.v_obs);
            }
        }
    }

    return { truthWindFn, bestWindFn, uncertaintyFn, stepSensing, observer, ekf, degrader };
}

// ── Episode runner ───────────────────────────────────────────────────

/**
 * Run a single training episode.
 *
 * @param {DQNAgent} agent
 * @param {Object} cfg - Training config (merged TRAIN_DEFAULTS)
 * @param {Function} rng - Seeded random function
 * @param {boolean} [train=true] - Whether to train (false for eval)
 * @param {Function|null} [onNavStep=null] - Optional per-nav-step callback.
 *   Called with { stateVec, actionIdx, reward, nextStateVec, done, dist_m, time_s }
 *   after each nav decision. Used by offline diagnostics (e.g. Q-value calibration).
 * @returns {Object} Episode stats
 */
export function runEpisode(agent, cfg, rng, train = true, onNavStep = null) {
    const p = runtime.platform;

    // Resolve wind source: ERA5 archive or synthetic preset
    let windFns, targetLat, targetLon;
    if (cfg.WIND_ARCHIVE) {
        // ERA5 real wind: sample a random episode (location + start time)
        const sample = cfg.WIND_ARCHIVE.sampleEpisode(rng, {
            duration_s: cfg.EPISODE_DURATION_S,
        });
        windFns   = { truthWindFn: sample.truthWindFn, baseWindFn: sample.baseWindFn };
        targetLat = sample.targetLat;
        targetLon = sample.targetLon;
    } else {
        // Synthetic preset
        const layers = WIND_PRESETS[cfg.PRESET]?.layers;
        if (!layers) throw new Error(`Unknown preset: ${cfg.PRESET}`);
        windFns = {
            truthWindFn: (alt_m, time_s) => getWind(layers, alt_m, time_s),
            baseWindFn:  (alt_m)         => getBaseWind(layers, alt_m),
        };
        targetLat = cfg.TARGET_LAT;
        targetLon = cfg.TARGET_LON;
    }

    // Create per-episode sensing stack with unique degrader seed
    const degraderSeed = (cfg.DEGRADER_SEED_OFFSET || 7777) + Math.floor(rng() * 100000);

    // Domain randomisation: per-episode sigma scale and lag, only during training
    let drOpts = null;
    if (train && cfg.DOMAIN_RAND_ENABLED) {
        const lo = Math.log(cfg.DOMAIN_RAND_SIGMA_MIN);
        const hi = Math.log(cfg.DOMAIN_RAND_SIGMA_MAX);
        const sigmaScale = Math.exp(lo + rng() * (hi - lo));    // log-uniform
        const lag_s = rng() * cfg.DOMAIN_RAND_MAX_LAG_S;
        // Apply scale to both bias and noise sigmas (calibrated values from
        // wind_degrader.DEGRADER_DEFAULTS — the multiplier preserves their ratio)
        drOpts = {
            biasSigma:  0.71 * sigmaScale,
            noiseSigma: 2.93 * sigmaScale,
            lag_s,
        };
    }
    const sensing = createSensingStack(windFns, degraderSeed, drOpts);

    // Random spawn position: offset from target in random direction
    const offsetKm = cfg.SPAWN_OFFSET_KM;
    const angle = rng() * 2 * Math.PI;
    const offsetLat = (offsetKm / 111.32) * Math.cos(angle);
    const offsetLon = (offsetKm / (111.32 * Math.cos(targetLat * Math.PI / 180 || 1))) * Math.sin(angle);
    const spawnLat = targetLat + offsetLat;
    const spawnLon = targetLon + offsetLon;

    // Random altitude within reachable band
    const spawnAlt = cfg.SPAWN_ALT_MIN_M + rng() * (cfg.SPAWN_ALT_MAX_M - cfg.SPAWN_ALT_MIN_M);

    // Initialize balloon
    let state = createState(spawnLat, spawnLon, spawnAlt);
    let prevState = null;
    let time_s = 0;

    // Metrics
    let inRadiusSteps = 0;
    let totalNavSteps = 0;
    let maxDist = 0;
    let totalReward = 0;
    let trainLosses = [];

    const totalPhysicsSteps = Math.ceil(cfg.EPISODE_DURATION_S / cfg.PHYSICS_DT_S);
    const physicsStepsPerNav = Math.round(cfg.NAV_INTERVAL_S / cfg.PHYSICS_DT_S);

    // Current committed action
    let currentAction = 0;  // FLOAT

    // Approach-rate shaping: track distance at the previous nav decision
    let prevNavDist = null;
    const approachShaping = train ? (cfg.APPROACH_SHAPING || 0) : 0;
    const totalNavStepsEst = Math.ceil(cfg.EPISODE_DURATION_S / cfg.NAV_INTERVAL_S);

    // n-step return support: keep a rolling FIFO of pending transitions whose
    // reward is being accumulated. When the buffer reaches N_STEP entries
    // (or the episode ends) we flush the oldest entry into the agent's replay
    // buffer with the n-step accumulated reward.
    const N_STEP = (train && agent.config.N_STEP > 1) ? agent.config.N_STEP : 1;
    const GAMMA  = agent.config.GAMMA;
    const pending = [];  // { state, actionIdx, rewards: [], nextState, done }

    // Run episode
    for (let step = 0; step < totalPhysicsSteps; step++) {
        // Navigation decision at interval boundaries
        if (step % physicsStepsPerNav === 0) {
            // Extract current state using EKF-filtered wind + uncertainty
            const stateVec = agent.extractState(
                state, sensing.bestWindFn, time_s,
                targetLat, targetLon,
                sensing.uncertaintyFn
            );

            // Select action
            const actionIdx = agent.selectAction(stateVec, train);

            // Resolve action space:
            //   discrete3   — single ACS command for the whole nav interval
            //   targetAlt17 — agent picks a target alt; trainer bang-bang chases
            const space = agent.config.ACTION_SPACE || 'discrete3';
            let acsAction = 0;       // used for fall-through
            let targetAlt_m = null;
            if (space === 'targetAlt17') {
                targetAlt_m = targetAltFromIndex(
                    actionIdx,
                    runtime.altBandLow_m,
                    runtime.altBandHigh_m,
                );
            } else {
                acsAction = actionFromIndex(actionIdx);
            }

            // Apply physics for one nav interval to get next state
            let nextState = state;
            let nextPrev = prevState;
            let nextTime = time_s;
            for (let ps = 0; ps < physicsStepsPerNav && (step + ps) < totalPhysicsSteps; ps++) {
                // For targetAlt17, recompute the chase action each physics step
                // — once we cross the target we want to stop pumping.
                const stepAction = (space === 'targetAlt17')
                    ? chaseAction(nextState.alt_m, targetAlt_m)
                    : acsAction;

                // Physics uses TRUTH wind (the actual atmosphere)
                const wind = sensing.truthWindFn(nextState.alt_m, nextTime);
                const beforeStep = nextState;
                nextState = physicsStep(nextState, stepAction, wind, cfg.PHYSICS_DT_S);
                nextTime += cfg.PHYSICS_DT_S;

                // Feed each physics step into the sensing pipeline
                // (GPS drift → observer → EKF update with vertical correlation)
                sensing.stepSensing(nextState, beforeStep, cfg.PHYSICS_DT_S, nextTime);
                nextPrev = beforeStep;
            }

            // Compute reward from resulting position
            const dist = haversine(nextState.lat, nextState.lon, targetLat, targetLon);
            let reward = agent.computeReward(dist);

            // Approach-rate shaping: bonus proportional to distance closed,
            // normalized by station radius. Anneals 1→0 over 80% of nav steps.
            if (approachShaping > 0 && prevNavDist !== null) {
                const shapingFrac = Math.max(0, 1 - totalNavSteps / (0.8 * totalNavStepsEst));
                const closed = prevNavDist - dist;  // positive = closed toward target
                if (closed > 0) {
                    reward += approachShaping * shapingFrac * (closed / runtime.platform.STATION_RADIUS_M);
                }
            }
            prevNavDist = dist;

            // Extract next state vector (also using EKF-filtered wind)
            const nextStateVec = agent.extractState(
                nextState, sensing.bestWindFn, nextTime,
                targetLat, targetLon,
                sensing.uncertaintyFn
            );

            // Store transition (with n-step accumulation if enabled)
            const done = (step + physicsStepsPerNav >= totalPhysicsSteps);
            if (train) {
                if (N_STEP === 1) {
                    agent.remember(stateVec, actionIdx, reward, nextStateVec, done);
                } else {
                    // Append a new pending transition. Each pending entry tracks
                    // its own future-reward sum: we add this step's reward to
                    // ALL existing pending entries (γ-discounted by their age),
                    // then push the new one with empty future-rewards.
                    for (const t of pending) {
                        t.rewards.push(reward);
                        t.nextState = nextStateVec;
                        t.done = done;
                    }
                    pending.push({
                        state:     stateVec,
                        actionIdx,
                        rewards:   [reward],   // already includes step 0's reward
                        nextState: nextStateVec,
                        done,
                    });

                    // Flush any entry that has reached N_STEP rewards (or is done).
                    while (pending.length > 0 &&
                           (pending[0].rewards.length >= N_STEP || pending[0].done)) {
                        const t = pending.shift();
                        let G = 0;
                        for (let i = 0; i < t.rewards.length; i++) G += Math.pow(GAMMA, i) * t.rewards[i];
                        // Effective γ to apply at bootstrap step: γ^k where k = rewards.length
                        // We can't pass a per-transition γ to the agent's trainBatch (which uses
                        // a fixed config GAMMA). Workaround: encode the discount into the reward
                        // by setting the bootstrap term to γ^k * max Q(s_{t+k}). Cleanest: extend
                        // the replay format. For now we exploit that done=true zeroes the bootstrap
                        // and use a stored 'effectiveGamma' field on the transition. The agent reads
                        // it in trainBatch (see rl_agent.js).
                        agent.remember(t.state, t.actionIdx, G, t.nextState, t.done, {
                            effectiveGamma: Math.pow(GAMMA, t.rewards.length),
                        });
                    }
                }

                // Train
                for (let b = 0; b < cfg.TRAIN_BATCHES_PER_STEP; b++) {
                    const loss = agent.trainBatch();
                    if (loss !== null) trainLosses.push(loss);
                }
            }

            if (onNavStep) {
                onNavStep({
                    stateVec, actionIdx, reward, nextStateVec, done,
                    dist_m: dist, time_s: nextTime,
                });
            }

            // Metrics
            if (dist < p.STATION_RADIUS_M) inRadiusSteps++;
            totalNavSteps++;
            maxDist = Math.max(maxDist, dist);
            totalReward += reward;

            // Advance simulation state
            state = nextState;
            prevState = nextPrev;
            time_s = nextTime;
            currentAction = acsAction;

            // Skip the physics steps we already simulated
            step += physicsStepsPerNav - 1;
            continue;
        }

        // If we fall through (shouldn't happen with the skip above), just step physics
        const wind = sensing.truthWindFn(state.alt_m, time_s);
        const beforeStep = state;
        state = physicsStep(state, currentAction, wind, cfg.PHYSICS_DT_S);
        time_s += cfg.PHYSICS_DT_S;
        sensing.stepSensing(state, beforeStep, cfg.PHYSICS_DT_S, time_s);
        prevState = beforeStep;
    }

    const twr50 = totalNavSteps > 0 ? inRadiusSteps / totalNavSteps : 0;
    const meanLoss = trainLosses.length > 0
        ? trainLosses.reduce((a, b) => a + b) / trainLosses.length
        : null;

    return {
        twr50,
        maxDist_m: maxDist,
        totalReward,
        meanLoss,
        navSteps: totalNavSteps,
        epsilon: agent.epsilon,
    };
}

// ── Curriculum helpers ───────────────────────────────────────────────

/**
 * Resolve the curriculum tier for a given global episode index.
 *
 * @param {Array} curriculum - Array of {episodes, duration_s, label?} tiers
 * @param {number} ep - Global episode index (0-based)
 * @returns {{ tier: Object, tierIndex: number, tierEpStart: number, tierEpEnd: number }}
 */
export function curriculumTierAt(curriculum, ep) {
    let cumulative = 0;
    for (let i = 0; i < curriculum.length; i++) {
        const tier = curriculum[i];
        const start = cumulative;
        cumulative += tier.episodes;
        if (ep < cumulative) {
            return { tier, tierIndex: i, tierEpStart: start, tierEpEnd: cumulative - 1 };
        }
    }
    // Past the end: clamp to last tier
    const last = curriculum[curriculum.length - 1];
    return {
        tier: last,
        tierIndex: curriculum.length - 1,
        tierEpStart: cumulative - last.episodes,
        tierEpEnd: cumulative - 1,
    };
}

/**
 * Compute total episodes from a curriculum schedule.
 * @param {Array} curriculum
 * @returns {number}
 */
export function curriculumTotalEpisodes(curriculum) {
    return curriculum.reduce((sum, t) => sum + t.episodes, 0);
}

// ── Full training run ────────────────────────────────────────────────

/**
 * Train a DQN agent over multiple episodes.
 *
 * Includes best-model checkpointing: saves weights whenever evaluation
 * TWR-50 improves, and restores the best model at the end. This prevents
 * late-training instability from overwriting a good policy.
 *
 * @param {Object} [options] - Override TRAIN_DEFAULTS and RL agent options
 * @param {Function} [onEpisode] - Callback (episodeIdx, stats) after each episode
 * @returns {Object} { agent, trainHistory, evalHistory, bestEvalEpisode, bestEvalTwr50 }
 */
export function trainAgent(options = {}, onEpisode = null) {
    // Ensure atmosphere derived values are initialized
    recalculateDerived();

    const cfg = { ...TRAIN_DEFAULTS, ...options };
    const rng = makeRng(cfg.SEED);

    // Curriculum: derive total episodes and per-episode duration dynamically.
    // When CURRICULUM is set, cfg.EPISODES is ignored in favour of the tier sum.
    const curriculum = cfg.CURRICULUM || null;
    const totalEpisodes = curriculum ? curriculumTotalEpisodes(curriculum) : cfg.EPISODES;

    // Create agent (pass through any RL-specific options)
    const agent = new DQNAgent(options);

    const trainHistory = [];
    const evalHistory = [];

    // Best-model checkpoint
    let bestWeights = null;
    let bestEvalTwr50 = -Infinity;
    let bestEvalEpisode = -1;

    // Curriculum state — track the active tier so we can fire tier-start events
    let prevTierIndex = -1;

    for (let ep = 0; ep < totalEpisodes; ep++) {
        // Resolve the current tier (null when not using curriculum)
        let tierInfo = null;
        let episodeDuration_s = cfg.EPISODE_DURATION_S;
        if (curriculum) {
            tierInfo = curriculumTierAt(curriculum, ep);
            episodeDuration_s = tierInfo.tier.duration_s;
        }

        const episodeCfg = episodeDuration_s !== cfg.EPISODE_DURATION_S
            ? { ...cfg, EPISODE_DURATION_S: episodeDuration_s }
            : cfg;

        // Training episode
        const stats = runEpisode(agent, episodeCfg, rng, true);
        agent.decayEpsilon();

        // Annotate stats with curriculum info for the callback
        const annotated = { ...stats };
        if (tierInfo) {
            annotated.tierIndex    = tierInfo.tierIndex;
            annotated.tierLabel    = tierInfo.tier.label || `tier${tierInfo.tierIndex}`;
            annotated.tierDuration_s = tierInfo.tier.duration_s;
            annotated.tierEpStart  = tierInfo.tierEpStart;
            annotated.tierEpEnd    = tierInfo.tierEpEnd;
            annotated.tierChanged  = tierInfo.tierIndex !== prevTierIndex;
            prevTierIndex = tierInfo.tierIndex;
        }

        trainHistory.push({ episode: ep, ...annotated });

        // Periodic evaluation (greedy, no exploration).
        // Evaluation always uses cfg.EVAL_DURATION_S (24h by default) regardless
        // of the current curriculum tier — this gives a consistent metric across tiers.
        if (cfg.EVAL_EVERY > 0 && (ep + 1) % cfg.EVAL_EVERY === 0) {
            const evalCfg = { ...cfg, EPISODE_DURATION_S: cfg.EVAL_DURATION_S };
            const nRuns = cfg.EVAL_RUNS || 1;
            let sumTwr50 = 0, worstMax = 0, sumReward = 0, sumNavSteps = 0;

            for (let r = 0; r < nRuns; r++) {
                const evalRng = makeRng(cfg.SEED + 1000 + ep * 100 + r);
                const s = runEpisode(agent, evalCfg, evalRng, false);
                sumTwr50 += s.twr50;
                worstMax = Math.max(worstMax, s.maxDist_m);
                sumReward += s.totalReward;
                sumNavSteps += s.navSteps;
            }

            const evalStats = {
                twr50:       sumTwr50 / nRuns,
                maxDist_m:   worstMax,
                totalReward: sumReward / nRuns,
                navSteps:    Math.round(sumNavSteps / nRuns),
                epsilon:     agent.epsilon,
            };
            if (tierInfo) {
                evalStats.tierLabel = tierInfo.tier.label || `tier${tierInfo.tierIndex}`;
                evalStats.tierIndex = tierInfo.tierIndex;
            }

            evalHistory.push({ episode: ep, ...evalStats });

            // Checkpoint if this is the best eval so far
            if (evalStats.twr50 > bestEvalTwr50) {
                bestEvalTwr50    = evalStats.twr50;
                bestEvalEpisode  = ep;
                bestWeights      = agent.serialize();
            }
        }

        if (onEpisode) onEpisode(ep, annotated);
    }

    // Restore best model (if we did any evaluation)
    if (bestWeights) {
        agent.deserialize(bestWeights);
    }

    return { agent, trainHistory, evalHistory, bestEvalEpisode, bestEvalTwr50 };
}

// ── Quick benchmark (greedy evaluation only) ─────────────────────────

/**
 * Evaluate a trained agent on a specific preset for a given duration.
 *
 * @param {DQNAgent} agent
 * @param {Object} [options]
 * @returns {Object} Evaluation stats
 */
export function benchmarkAgent(agent, options = {}) {
    recalculateDerived();

    const cfg = {
        ...TRAIN_DEFAULTS,
        EPISODE_DURATION_S: 3600 * 72,  // 72 hours by default
        ...options,
    };
    const rng = makeRng(cfg.SEED + 9999);

    return runEpisode(agent, cfg, rng, false);
}

/**
 * simulator.js — Simulation engine.
 *
 * Drives the balloon physics + navigator at configurable speed.
 * Supports both synthetic wind presets and real forecast data.
 * Emits events for the UI to consume (state updates, decisions, metrics).
 */
import { runtime } from './config.js';
import { haversine } from './geo.js';
import { getWind, WIND_PRESETS } from './wind.js';
import { createState, physicsStep } from './balloon.js';
import { chooseAction, resetNavMemory } from './navigator.js';
import { WindObservationStore } from './wind_observer.js';
import { WindEKF } from './wind_ekf.js';
import { ForecastDegrader } from './wind_degrader.js';
import { getBaseWind } from './wind.js';
import { DQNAgent, actionFromIndex } from './rl_agent.js';

// ── Simulation state ────────────────────────────────────────────────

export class Simulator {
    constructor() {
        this.state = null;
        this.targetLat = 0;
        this.targetLon = 170;
        this.time_s = 0;
        this.layers = WIND_PRESETS.tropical.layers;
        this.windProfile = null;   // Set when using real forecast data
        this.presetName = 'tropical';

        // Nav state
        this.lastDecision = null;
        this.targetAlt = null;
        this.commitAction = 0;
        this.navTimer = 0;
        this.cooldownTimer = 0;
        this.stall_count = 0;
        this.prev_dist = Infinity;

        // Metrics
        this.trail = [];           // { lat, lon, alt_m, time_s, dist_m }
        this.decisions = [];       // Full decision history
        this.twr50_inRadius = 0;
        this.twr50_total = 0;
        this.maxDist = 0;
        this.totalSteps = 0;

        // Wind observation store (Q1.1: in-situ wind measurement)
        this.windObserver = new WindObservationStore();
        this._prevState = null;

        // EKF wind estimator (Q1.2: Bayesian wind state estimation)
        this.windEkf = new WindEKF();

        // Forecast degrader (Q1.5: simulate forecast errors)
        this.forecastDegrader = null;

        // RL agent (Q2: DQN controller)
        this.rlAgent = null;

        // Playback
        this.running = false;
        this.speedMultiplier = 600;
        this._animFrame = null;
        this._listeners = {};
    }

    // ── Event system ────────────────────────────────────────────────

    on(event, fn) {
        (this._listeners[event] ??= []).push(fn);
        return this;
    }

    emit(event, data) {
        for (const fn of this._listeners[event] ?? []) fn(data);
    }

    // ── Initialization ──────────────────────────────────────────────

    reset(lat = 0, lon = 170.5, alt_m = 17500) {
        this.state = createState(lat, lon, alt_m);
        this.targetLat = lat;
        this.targetLon = lon;
        this.time_s = 0;
        this.lastDecision = null;
        this.targetAlt = alt_m;
        this.commitAction = 0;
        this.navTimer = 0;
        this.cooldownTimer = 0;
        this.stall_count = 0;
        this.prev_dist = Infinity;
        this.trail = [];
        this.decisions = [];
        this.twr50_inRadius = 0;
        this.twr50_total = 0;
        this.maxDist = 0;
        this.totalSteps = 0;
        this.windObserver.reset();
        this._prevState = null;
        this.windEkf.reset();

        // Initialize forecast degrader (Q1.5)
        if (runtime.features.forecastDegradation) {
            const truthFn = (a, t) => this.getWindAt(a, t);
            // For synthetic presets: freeze the base wind (no temporal variation)
            const baseFn = this.windProfile ? null :
                (a) => getBaseWind(this.layers, a);
            this.forecastDegrader = new ForecastDegrader(truthFn, baseFn);
        } else {
            this.forecastDegrader = null;
        }

        if (runtime.features.windEkf) {
            // Initialize EKF with forecast prior (degraded if available)
            const forecastFn = this.forecastDegrader
                ? (alt_m) => this.forecastDegrader.getForecastWind(alt_m, 0)
                : (alt_m) => this.getWindAt(alt_m, 0);
            this.windEkf.initialize(forecastFn);
        }
        // Initialize RL agent (Q2) if enabled
        if (runtime.features.useRl && !this.rlAgent) {
            this.rlAgent = new DQNAgent();
        }

        resetNavMemory();
        this.emit('reset', this.getSnapshot());
    }

    setPreset(name) {
        if (WIND_PRESETS[name]) {
            this.presetName = name;
            this.layers = WIND_PRESETS[name].layers;
            this.windProfile = null;
            this.emit('presetChanged', { name, layers: this.layers });
        }
    }

    setWindProfile(profile, layers) {
        this.windProfile = profile;
        this.layers = layers;
        this.presetName = 'forecast';
        this.emit('presetChanged', { name: 'forecast', layers });
    }

    setTarget(lat, lon) {
        this.targetLat = lat;
        this.targetLon = lon;
        this.emit('targetMoved', { lat, lon });
    }

    // ── Wind accessor (auto-selects source) ─────────────────────────

    getWindAt(alt_m, time_s) {
        if (this.windProfile) {
            return this.windProfile.getWind(alt_m, time_s);
        }
        return getWind(this.layers, alt_m, time_s);
    }

    // ── Simulation step ─────────────────────────────────────────────

    step() {
        if (!this.state) return;

        const nav = runtime.nav;
        const dt = nav.PHYSICS_DT_S;

        // Navigation decision every NAV_INTERVAL_S
        this.navTimer += dt;
        if (this.navTimer >= nav.NAV_INTERVAL_S) {
            this.navTimer = 0;
            this._makeNavDecision();
        }

        // Determine action: commit to altitude transition, or use decision
        let action = this.commitAction;

        // Check if we've reached target altitude
        if (this.targetAlt != null) {
            if (Math.abs(this.state.alt_m - this.targetAlt) < nav.COMMITMENT_THRESHOLD_M) {
                this.commitAction = 0;  // Float at target
                if (this.cooldownTimer <= 0) {
                    this.cooldownTimer = nav.COOLDOWN_INTERVALS * nav.NAV_INTERVAL_S;
                }
            }
        }

        // Cooldown: float briefly after reaching target
        if (this.cooldownTimer > 0) {
            action = 0;
            this.cooldownTimer -= dt;
        }

        // Get wind at current position
        const wind = this.getWindAt(this.state.alt_m, this.time_s);

        // Physics step
        const prevState = this.state;
        this.state = physicsStep(this.state, action, wind, dt);
        this.time_s += dt;

        // Collect in-situ wind observation from GPS drift
        if (this._prevState) {
            const obs = this.windObserver.observe(this.state, this._prevState, dt, this.time_s);

            // Feed observation into EKF
            if (runtime.features.windEkf && this.windEkf.initialized) {
                this.windEkf.predict(dt);
                if (obs) {
                    this.windEkf.update(obs.alt_m, obs.u_obs, obs.v_obs);
                }
            }
        }
        this._prevState = prevState;

        // Update metrics
        const dist = haversine(this.state.lat, this.state.lon, this.targetLat, this.targetLon);
        this.totalSteps++;
        if (dist < runtime.platform.STATION_RADIUS_M) this.twr50_inRadius++;
        this.twr50_total = this.twr50_inRadius / this.totalSteps;
        this.maxDist = Math.max(this.maxDist, dist);

        // Trail
        this.trail.push({
            lat: this.state.lat, lon: this.state.lon,
            alt_m: this.state.alt_m, time_s: this.time_s,
            dist_m: dist, action,
        });

        // Emit update
        this.emit('step', this.getSnapshot());
    }

    /**
     * Get the "raw" forecast wind — either degraded or truth depending on config.
     * This is what the navigator sees before EKF correction.
     */
    getRawForecastAt(alt_m, time_s) {
        if (this.forecastDegrader) {
            return this.forecastDegrader.getForecastWind(alt_m, time_s);
        }
        return this.getWindAt(alt_m, time_s);
    }

    /**
     * Get the best available wind estimate, optionally using EKF.
     * When EKF is enabled and has low uncertainty at the queried altitude,
     * prefer the EKF estimate (informed by in-situ observations).
     *
     * Wind sources, in order of preference:
     *   1. EKF estimate (if confident, sigma < 5 m/s)
     *   2. Blend of EKF + raw forecast (weighted by inverse variance)
     *   3. Raw forecast (degraded if Q1.5 enabled, else truth)
     */
    getBestWindAt(alt_m, time_s) {
        if (runtime.features.windEkf && this.windEkf.initialized) {
            const ekfWind = this.windEkf.getWind(alt_m);
            const ekfSigma = this.windEkf.getUncertainty(alt_m);

            // If EKF has reasonable confidence (sigma < 5 m/s), use it
            if (ekfSigma < 5.0) {
                return ekfWind;
            }

            // Otherwise blend: weight by inverse uncertainty
            const rawWind = this.getRawForecastAt(alt_m, time_s);
            // When forecast is degraded, raw uncertainty is higher
            const rawSigma = this.forecastDegrader
                ? this.forecastDegrader.getUncertainty(alt_m)
                : 4.0;
            const wEkf = 1 / (ekfSigma * ekfSigma);
            const wRaw = 1 / (rawSigma * rawSigma);
            const wTotal = wEkf + wRaw;
            return {
                u: (ekfWind.u * wEkf + rawWind.u * wRaw) / wTotal,
                v: (ekfWind.v * wEkf + rawWind.v * wRaw) / wTotal,
            };
        }
        return this.getRawForecastAt(alt_m, time_s);
    }

    /**
     * Load pre-trained RL weights from serialized data.
     * Call before reset() or after enabling useRl flag.
     * @param {Object} data - Serialized DQNAgent data (from agent.serialize())
     */
    loadRlWeights(data) {
        if (!this.rlAgent) {
            this.rlAgent = new DQNAgent();
        }
        this.rlAgent.deserialize(data);
    }

    _makeNavDecision() {
        // RL controller path (Q2)
        if (runtime.features.useRl && this.rlAgent) {
            const windFn = (alt_m, t) => this.getBestWindAt(alt_m, t);
            const uncertaintyFn = (runtime.features.windEkf && this.windEkf.initialized)
                ? (alt_m) => this.windEkf.getUncertainty(alt_m)
                : null;
            const stateVec = this.rlAgent.extractState(
                this.state, windFn, this.time_s,
                this.targetLat, this.targetLon, uncertaintyFn
            );
            const actionIdx = this.rlAgent.selectAction(stateVec, false);  // greedy
            const acsAction = actionFromIndex(actionIdx);
            const qValues = this.rlAgent.getQValues(stateVec);

            const decision = {
                action: acsAction,
                targetAlt: acsAction === 1 ? this.state.alt_m + 500 :
                           acsAction === -1 ? this.state.alt_m - 500 :
                           this.state.alt_m,
                reason: `RL: Q=[${qValues.map(q => q.toFixed(2)).join(', ')}]`,
                stage: 'rl',
            };

            this.lastDecision = decision;
            this.decisions.push({ time_s: this.time_s, ...decision });

            // RL directly sets action — no stall detection or hysteresis
            if (decision.action !== 0) {
                this.targetAlt = decision.targetAlt;
                this.commitAction = decision.action;
                this.cooldownTimer = 0;
            } else {
                this.commitAction = 0;
            }

            this.emit('decision', { time_s: this.time_s, decision });
            return;
        }

        // Pass wind accessor — uses EKF-enhanced wind when available
        const windFn = (alt_m, t) => this.getBestWindAt(alt_m, t);

        // Pass uncertainty accessor for Q1.4 uncertainty scoring
        const uncertaintyFn = (runtime.features.uncertaintyScoring &&
            runtime.features.windEkf && this.windEkf.initialized)
            ? (alt_m) => this.windEkf.getUncertainty(alt_m)
            : null;

        const decision = chooseAction(
            this.state, windFn, this.time_s,
            this.targetLat, this.targetLon, uncertaintyFn
        );

        this.lastDecision = decision;
        this.decisions.push({ time_s: this.time_s, ...decision });

        // Stall detection
        const dist = haversine(this.state.lat, this.state.lon, this.targetLat, this.targetLon);
        if (Math.abs(dist - this.prev_dist) < runtime.nav.STALL_THRESHOLD_M) {
            this.stall_count++;
        } else {
            this.stall_count = 0;
        }
        this.prev_dist = dist;

        // If stalling, force re-evaluation by clearing commitment
        if (this.stall_count >= runtime.nav.STALL_INTERVALS) {
            this.commitAction = 0;
            this.targetAlt = null;
            this.cooldownTimer = 0;
            this.stall_count = 0;
        }

        // Apply decision
        if (decision.action !== 0) {
            this.targetAlt = decision.targetAlt;
            this.commitAction = decision.action;
            this.cooldownTimer = 0;
        } else if (decision.stage !== 'hysteresis') {
            this.commitAction = 0;
        }

        this.emit('decision', { time_s: this.time_s, decision });
    }

    // ── Snapshot for UI ─────────────────────────────────────────────

    getSnapshot() {
        const dist = this.state ?
            haversine(this.state.lat, this.state.lon, this.targetLat, this.targetLon) : 0;
        return {
            state: this.state,
            time_s: this.time_s,
            dist_m: dist,
            twr50: this.twr50_total,
            maxDist_m: this.maxDist,
            decision: this.lastDecision,
            targetLat: this.targetLat,
            targetLon: this.targetLon,
            presetName: this.presetName,
            trailLength: this.trail.length,
            windObsSummary: this.windObserver.summary(this.time_s),
            ekfInitialized: this.windEkf.initialized,
            forecastDegraded: !!this.forecastDegrader,
            rlLoaded: !!(this.rlAgent && runtime.features.useRl),
        };
    }

    // ── Playback controls ───────────────────────────────────────────

    start() {
        if (this.running) return;
        this.running = true;
        this._lastFrame = performance.now();
        this._loop();
        this.emit('playStateChanged', { running: true });
    }

    pause() {
        this.running = false;
        if (this._animFrame) cancelAnimationFrame(this._animFrame);
        this.emit('playStateChanged', { running: false });
    }

    toggle() {
        if (this.running) this.pause(); else this.start();
    }

    setSpeed(multiplier) {
        this.speedMultiplier = multiplier;
    }

    _loop() {
        if (!this.running) return;
        const now = performance.now();
        const wallDt = Math.min((now - this._lastFrame) / 1000, 0.1); // cap at 100ms
        this._lastFrame = now;

        // Accumulate simulated time and step when enough has accumulated
        this._simTimeAccum = (this._simTimeAccum || 0) + wallDt * this.speedMultiplier;
        const physDt = runtime.nav.PHYSICS_DT_S;
        const steps = Math.min(Math.floor(this._simTimeAccum / physDt), 200);
        this._simTimeAccum -= steps * physDt;

        for (let i = 0; i < steps; i++) {
            this.step();
        }

        // Emit interpolation fraction for smooth rendering between steps
        this._interpFrac = this._simTimeAccum / physDt;

        this._animFrame = requestAnimationFrame(() => this._loop());
    }

    // ── Batch simulation (headless, for analysis) ───────────────────

    runBatch(durationHours = 72) {
        const totalSteps = Math.ceil(durationHours * 3600 / runtime.nav.PHYSICS_DT_S);
        for (let i = 0; i < totalSteps; i++) {
            this.step();
        }
        return this.getSnapshot();
    }
}

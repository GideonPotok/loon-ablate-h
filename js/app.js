/**
 * app.js — Main application entry point.
 *
 * Initializes all modules, wires up the UI, and starts the simulation loop.
 */
import { runtime } from './config.js';
import { recalculateDerived } from './atmosphere.js';
import { Simulator } from './simulator.js';
import { MapView } from './map.js';
import { WindProfileChart, AltitudeTimelineChart } from './charts.js';
import { loadForecast, profileToLayers } from './forecast.js';

// ── Globals ─────────────────────────────────────────────────────────

let sim, mapView, windChart, altChart;
let forecastData = null;
let forecastTimeSlider = null;

// ── Initialization ──────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
    // Compute derived values
    recalculateDerived();

    // Initialize simulator
    sim = new Simulator();

    // Initialize map
    mapView = new MapView('map');

    // Initialize charts
    windChart = new WindProfileChart('wind-profile-canvas');
    altChart  = new AltitudeTimelineChart('alt-timeline-canvas');

    // Wire up simulator events
    sim.on('step', onSimStep);
    sim.on('decision', onDecision);
    sim.on('reset', onReset);

    // Wire up UI controls
    wireControls();

    // Initial reset
    sim.reset(0, 170.5, 17500);
    sim.setTarget(0, 170);
    mapView.updateTarget(0, 170);
    mapView.panTo(0, 170);

    // Open default sections
    document.querySelectorAll('.panel-section').forEach(s => s.classList.add('open'));

    updateMetrics(sim.getSnapshot());
});

// ── Simulation event handlers ───────────────────────────────────────

let frameCount = 0;

function onSimStep(snapshot) {
    frameCount++;

    // Update map every 5 frames for performance
    if (frameCount % 5 === 0) {
        mapView.updateFromSnapshot(snapshot, sim.trail);
    }

    // Update charts every 30 frames
    if (frameCount % 30 === 0) {
        updateCharts(snapshot);
    }

    // Update metrics every 10 frames
    if (frameCount % 10 === 0) {
        updateMetrics(snapshot);
    }
}

function onDecision({ time_s, decision }) {
    addDecisionLog(time_s, decision);
}

function onReset(snapshot) {
    frameCount = 0;
    updateMetrics(snapshot);
    const column = buildWindColumn(snapshot);
    windChart.draw(column, snapshot.state?.alt_m ?? 17500, runtime.altBandLow_m, runtime.altBandHigh_m);
}

// ── Chart updates ───────────────────────────────────────────────────

function buildWindColumn(snapshot) {
    const alts = [];
    for (let a = 15000; a <= 22000; a += 500) alts.push(a);
    return alts.map(alt_m => {
        const w = sim.getWindAt(alt_m, sim.time_s);
        return { alt_m, u: w.u, v: w.v };
    });
}

function updateCharts(snapshot) {
    // Wind profile
    const column = buildWindColumn(snapshot);
    windChart.draw(column, snapshot.state?.alt_m ?? 17500, runtime.altBandLow_m, runtime.altBandHigh_m);

    // Altitude timeline
    altChart.draw(sim.trail, runtime.platform.STATION_RADIUS_M);
}

// ── Metrics display ─────────────────────────────────────────────────

function updateMetrics(snapshot) {
    const s = snapshot.state;
    if (!s) return;

    const dist = snapshot.dist_m;
    const radius = runtime.platform.STATION_RADIUS_M;

    setMetric('dist', (dist / 1000).toFixed(1), 'km',
              dist < radius ? 'good' : dist < radius * 2 ? 'warn' : 'bad');
    setMetric('altitude', (s.alt_m / 1000).toFixed(2), 'km');
    setMetric('twr50', (snapshot.twr50 * 100).toFixed(1), '%',
              snapshot.twr50 > 0.6 ? 'good' : snapshot.twr50 > 0.3 ? 'warn' : 'bad');
    setMetric('max-dist', (snapshot.maxDist_m / 1000).toFixed(0), 'km');

    const hrs = (snapshot.time_s / 3600).toFixed(1);
    setMetric('sim-time', hrs, 'h');

    setMetric('ballast', s.ballast_kg.toFixed(1), 'kg');
    setMetric('vv', (s.vv_m_s * 100).toFixed(0), 'cm/s');
    setMetric('energy', s.energy_used_j.toFixed(0), 'kJ');

    // Status bar
    const statusEl = document.getElementById('status-text');
    if (statusEl) {
        const wind = sim.getWindAt(s.alt_m, snapshot.time_s);
        const wspd = Math.sqrt(wind.u * wind.u + wind.v * wind.v).toFixed(1);
        statusEl.textContent =
            `${s.lat.toFixed(4)}°, ${s.lon.toFixed(4)}° | ` +
            `Wind: ${wspd} m/s | ` +
            `Steps: ${sim.totalSteps} | ` +
            `Preset: ${snapshot.presetName}`;
    }

    // Decision info
    if (snapshot.decision) {
        const decEl = document.getElementById('current-decision');
        if (decEl) {
            decEl.textContent = snapshot.decision.reason;
        }
    }
}

function setMetric(id, value, unit, cls = '') {
    const el = document.getElementById(`metric-${id}`);
    if (!el) return;
    el.querySelector('.metric-value').textContent = value;
    el.querySelector('.metric-value').className = `metric-value ${cls}`;
    const unitEl = el.querySelector('.metric-unit');
    if (unitEl) unitEl.textContent = unit;
}

// ── Decision log ────────────────────────────────────────────────────

function addDecisionLog(time_s, decision) {
    const log = document.getElementById('decision-log');
    if (!log) return;

    const h = (time_s / 3600).toFixed(1);
    const actionName = decision.action === 1 ? 'ASC' : decision.action === -1 ? 'DES' : 'FLT';
    const actionClass = decision.action === 1 ? 'ascend' : decision.action === -1 ? 'descend' : 'float';

    const entry = document.createElement('div');
    entry.className = 'decision-entry';
    entry.innerHTML = `<span class="decision-time">${h}h</span>` +
        `<span class="decision-action ${actionClass}">${actionName}</span> ` +
        `<span>${decision.reason}</span>`;

    log.prepend(entry);

    // Limit entries
    while (log.children.length > 100) log.lastChild.remove();
}

// ── UI Controls wiring ──────────────────────────────────────────────

function wireControls() {
    // Panel section collapse/expand
    document.querySelectorAll('.panel-section-header').forEach(header => {
        header.addEventListener('click', () => {
            header.parentElement.classList.toggle('open');
        });
    });

    // Play/Pause button
    document.getElementById('btn-play')?.addEventListener('click', () => {
        sim.toggle();
        const btn = document.getElementById('btn-play');
        btn.textContent = sim.running ? '⏸' : '▶';
        document.getElementById('status-dot')?.classList.toggle('running', sim.running);
        document.getElementById('status-dot')?.classList.toggle('paused', !sim.running);
    });

    // Reset button
    document.getElementById('btn-reset')?.addEventListener('click', () => {
        const lat = parseFloat(document.getElementById('input-lat')?.value) || 0;
        const lon = parseFloat(document.getElementById('input-lon')?.value) || 170.5;
        sim.pause();
        sim.reset(lat, lon + 0.5, 17500);
        sim.setTarget(lat, lon);
        mapView.updateTarget(lat, lon);
        mapView.fitBounds(lat, lon + 0.5, lat, lon);
        document.getElementById('btn-play').textContent = '▶';
        document.getElementById('status-dot')?.classList.remove('running');
        document.getElementById('status-dot')?.classList.add('paused');
        document.getElementById('decision-log').innerHTML = '';
    });

    // Speed slider
    document.getElementById('speed-slider')?.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        const speed = [60, 300, 600, 1800, 3600, 7200, 18000][val] ?? 600;
        sim.setSpeed(speed);
        const label = speed >= 3600 ? `${speed/3600}h/s` : `${speed}×`;
        document.getElementById('speed-label').textContent = label;
    });

    // Wind preset selector
    document.getElementById('select-preset')?.addEventListener('change', (e) => {
        const preset = e.target.value;
        if (preset === 'forecast') {
            fetchForecast();
        } else {
            sim.setPreset(preset);
            forecastData = null;
        }
    });

    // Fetch forecast button
    document.getElementById('btn-fetch-forecast')?.addEventListener('click', fetchForecast);

    // Map click → reposition target
    mapView.onMapClick((lat, lon) => {
        sim.setTarget(lat, lon);
        mapView.updateTarget(lat, lon);
        document.getElementById('input-lat').value = lat.toFixed(4);
        document.getElementById('input-lon').value = lon.toFixed(4);
    });

    mapView.onTargetDrag((lat, lon) => {
        sim.setTarget(lat, lon);
        document.getElementById('input-lat').value = lat.toFixed(4);
        document.getElementById('input-lon').value = lon.toFixed(4);
    });

    // Nav feature toggles
    document.querySelectorAll('[data-feature]').forEach(toggle => {
        toggle.addEventListener('change', (e) => {
            const feature = e.target.dataset.feature;
            runtime.features[feature] = e.target.checked;
        });
    });

    // Platform parameter inputs
    document.querySelectorAll('[data-param]').forEach(input => {
        input.addEventListener('change', (e) => {
            const param = e.target.dataset.param;
            const val = parseFloat(e.target.value);
            if (isNaN(val)) return;
            if (param in runtime.platform) {
                runtime.platform[param] = val;
            } else if (param in runtime.nav) {
                runtime.nav[param] = val;
            } else if (param in runtime.wind) {
                runtime.wind[param] = val;
            }
            recalculateDerived();
            updateReachableBand();
        });
    });

    // Station radius (km → m conversion)
    document.getElementById('input-radius-km')?.addEventListener('change', (e) => {
        const km = parseFloat(e.target.value);
        if (!isNaN(km) && km > 0) {
            runtime.platform.STATION_RADIUS_M = km * 1000;
        }
    });

    // Forecast time slider
    document.getElementById('forecast-time-slider')?.addEventListener('input', (e) => {
        if (!forecastData) return;
        const hour = parseInt(e.target.value);
        const layers = profileToLayers(forecastData.profile, hour * 3600, (hour + 1) * 3600);
        sim.setWindProfile(forecastData.profile, layers);
        document.getElementById('forecast-time-label').textContent = `+${hour}h`;
    });
}

// ── Forecast fetching ───────────────────────────────────────────────

async function fetchForecast() {
    const lat = parseFloat(document.getElementById('input-lat')?.value) || 0;
    const lon = parseFloat(document.getElementById('input-lon')?.value) || 170;
    const model = document.getElementById('select-model')?.value || 'gfs';
    const days = parseInt(document.getElementById('input-forecast-days')?.value) || 7;

    const overlay = document.getElementById('loading-overlay');
    overlay?.classList.add('active');

    try {
        forecastData = await loadForecast(lat, lon, days, model);
        sim.setWindProfile(forecastData.profile, forecastData.layers);

        // Update forecast info
        const infoEl = document.getElementById('forecast-info');
        if (infoEl) {
            const snaps = forecastData.profile.snapshots.length;
            infoEl.innerHTML = `<span class="source">${model.toUpperCase()}</span> ` +
                `${snaps} hours | ${lat.toFixed(1)}°, ${lon.toFixed(1)}° ` +
                `<span class="update-time">${new Date().toLocaleTimeString()}</span>`;
        }

        // Enable time slider
        const slider = document.getElementById('forecast-time-slider');
        if (slider) {
            slider.max = forecastData.profile.snapshots.length - 1;
            slider.disabled = false;
        }

        // Update wind chart immediately
        const column = buildWindColumn(sim.getSnapshot());
        windChart.draw(column, sim.state?.alt_m ?? 17500, runtime.altBandLow_m, runtime.altBandHigh_m);

    } catch (err) {
        console.error('Forecast fetch failed:', err);
        const infoEl = document.getElementById('forecast-info');
        if (infoEl) infoEl.innerHTML = `<span style="color:var(--accent-red)">Error: ${err.message}</span>`;
    } finally {
        overlay?.classList.remove('active');
    }
}

// ── Reachable band display ──────────────────────────────────────────

function updateReachableBand() {
    const el = document.getElementById('reachable-band');
    if (el) {
        el.textContent = `${(runtime.altBandLow_m/1000).toFixed(1)} – ${(runtime.altBandHigh_m/1000).toFixed(1)} km`;
    }
}

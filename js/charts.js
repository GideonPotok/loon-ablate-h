/**
 * charts.js — Wind profile and altitude timeline visualizations.
 *
 * Uses pure Canvas rendering (no external chart libraries).
 * Two panels:
 *   1. Wind profile — u/v components by altitude (vertical axis)
 *   2. Altitude timeline — altitude + distance over time
 */

// ── Wind Profile Chart ──────────────────────────────────────────────

export class WindProfileChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this._resize();
        window.addEventListener('resize', () => this._resize());
    }

    _resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.W = this.canvas.width;
        this.H = this.canvas.height;
    }

    /**
     * Draw wind profile.
     * @param {Array<{alt_m, u, v}>} column — Wind at multiple altitudes
     * @param {number} currentAlt — Balloon's current altitude
     * @param {number} bandLow — Reachable altitude band lower bound
     * @param {number} bandHigh — Reachable altitude band upper bound
     */
    draw(column, currentAlt, bandLow = 16500, bandHigh = 18500) {
        const ctx = this.ctx;
        const W = this.W, H = this.H;
        const margin = { top: 25, bottom: 25, left: 50, right: 15 };
        const plotW = W - margin.left - margin.right;
        const plotH = H - margin.top - margin.bottom;

        ctx.clearRect(0, 0, W, H);

        // Background
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, W, H);

        // Altitude range
        const altMin = 15000, altMax = 22000;
        const yScale = alt => margin.top + plotH * (1 - (alt - altMin) / (altMax - altMin));

        // Wind range
        const maxWind = Math.max(20, ...column.map(c => Math.max(Math.abs(c.u), Math.abs(c.v))));
        const xCenter = margin.left + plotW / 2;
        const xScale = val => xCenter + (val / maxWind) * (plotW / 2 - 10);

        // Reachable band
        ctx.fillStyle = 'rgba(46, 204, 113, 0.1)';
        ctx.fillRect(margin.left, yScale(bandHigh), plotW, yScale(bandLow) - yScale(bandHigh));
        ctx.strokeStyle = 'rgba(46, 204, 113, 0.4)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(margin.left, yScale(bandLow));
        ctx.lineTo(margin.left + plotW, yScale(bandLow));
        ctx.moveTo(margin.left, yScale(bandHigh));
        ctx.lineTo(margin.left + plotW, yScale(bandHigh));
        ctx.stroke();
        ctx.setLineDash([]);

        // Zero line
        ctx.strokeStyle = '#30363d';
        ctx.beginPath();
        ctx.moveTo(xCenter, margin.top);
        ctx.lineTo(xCenter, margin.top + plotH);
        ctx.stroke();

        // Grid lines (altitude)
        ctx.fillStyle = '#8b949e';
        ctx.font = '10px monospace';
        ctx.textAlign = 'right';
        for (let alt = 15000; alt <= 22000; alt += 1000) {
            const y = yScale(alt);
            ctx.strokeStyle = '#21262d';
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(margin.left + plotW, y);
            ctx.stroke();
            ctx.fillText(`${(alt/1000).toFixed(0)} km`, margin.left - 5, y + 4);
        }

        // Wind scale labels
        ctx.textAlign = 'center';
        ctx.fillText(`−${maxWind.toFixed(0)}`, xScale(-maxWind), margin.top - 8);
        ctx.fillText('0', xCenter, margin.top - 8);
        ctx.fillText(`+${maxWind.toFixed(0)}`, xScale(maxWind), margin.top - 8);

        // U-component (eastward) — blue
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        column.forEach((c, i) => {
            const x = xScale(c.u), y = yScale(c.alt_m);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // V-component (northward) — orange
        ctx.strokeStyle = '#f0883e';
        ctx.lineWidth = 2;
        ctx.beginPath();
        column.forEach((c, i) => {
            const x = xScale(c.v), y = yScale(c.alt_m);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        });
        ctx.stroke();

        // Current altitude indicator
        ctx.strokeStyle = '#e5534b';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        const yAlt = yScale(currentAlt);
        ctx.beginPath();
        ctx.moveTo(margin.left, yAlt);
        ctx.lineTo(margin.left + plotW, yAlt);
        ctx.stroke();
        ctx.setLineDash([]);

        // Legend
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#58a6ff';
        ctx.fillText('u (east)', margin.left + 5, H - 5);
        ctx.fillStyle = '#f0883e';
        ctx.fillText('v (north)', margin.left + 70, H - 5);
        ctx.fillStyle = '#e5534b';
        ctx.fillText('▼ balloon', margin.left + 145, H - 5);

        ctx.lineWidth = 1;
    }
}

// ── Altitude Timeline Chart ─────────────────────────────────────────

export class AltitudeTimelineChart {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this._resize();
        window.addEventListener('resize', () => this._resize());
    }

    _resize() {
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.W = this.canvas.width;
        this.H = this.canvas.height;
    }

    /**
     * Draw altitude and distance over time.
     * @param {Array<{time_s, alt_m, dist_m}>} trail
     * @param {number} stationRadius
     */
    draw(trail, stationRadius = 50000) {
        if (trail.length < 2) return;

        const ctx = this.ctx;
        const W = this.W, H = this.H;
        const margin = { top: 20, bottom: 25, left: 50, right: 50 };
        const plotW = W - margin.left - margin.right;
        const plotH = H - margin.top - margin.bottom;

        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = '#0d1117';
        ctx.fillRect(0, 0, W, H);

        // Time range
        const tMax = trail[trail.length - 1].time_s;
        const tMin = trail[0].time_s;
        const xScale = t => margin.left + plotW * ((t - tMin) / (tMax - tMin || 1));

        // Altitude range
        const altMin = 15000, altMax = 22000;
        const yAltScale = alt => margin.top + plotH * (1 - (alt - altMin) / (altMax - altMin));

        // Distance range
        const maxDist = Math.max(stationRadius * 2, ...trail.map(t => t.dist_m));
        const yDistScale = d => margin.top + plotH * (1 - d / maxDist);

        // Grid
        ctx.strokeStyle = '#21262d';
        ctx.font = '10px monospace';

        // Altitude grid (left axis)
        ctx.fillStyle = '#8b949e';
        ctx.textAlign = 'right';
        for (let alt = 15000; alt <= 22000; alt += 1000) {
            const y = yAltScale(alt);
            ctx.beginPath(); ctx.moveTo(margin.left, y); ctx.lineTo(W - margin.right, y); ctx.stroke();
            ctx.fillText(`${(alt/1000).toFixed(0)}km`, margin.left - 5, y + 4);
        }

        // Time grid (bottom axis)
        ctx.textAlign = 'center';
        const hourStep = tMax > 86400 ? 12 : (tMax > 3600 ? 1 : 0.25);
        for (let h = 0; h * 3600 <= tMax; h += hourStep) {
            const x = xScale(h * 3600);
            ctx.beginPath(); ctx.moveTo(x, margin.top); ctx.lineTo(x, margin.top + plotH); ctx.stroke();
            ctx.fillText(`${h}h`, x, H - 5);
        }

        // Station radius line (distance)
        ctx.strokeStyle = 'rgba(231, 76, 60, 0.4)';
        ctx.setLineDash([6, 3]);
        const yRadius = yDistScale(stationRadius);
        ctx.beginPath(); ctx.moveTo(margin.left, yRadius); ctx.lineTo(W - margin.right, yRadius); ctx.stroke();
        ctx.setLineDash([]);

        // Subsample trail for performance
        const maxPoints = 1500;
        const step = Math.max(1, Math.floor(trail.length / maxPoints));

        // Distance line (orange, right axis)
        ctx.strokeStyle = '#f0883e';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i += step) {
            const x = xScale(trail[i].time_s);
            const y = yDistScale(trail[i].dist_m);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Altitude line (cyan)
        ctx.strokeStyle = '#58a6ff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let i = 0; i < trail.length; i += step) {
            const x = xScale(trail[i].time_s);
            const y = yAltScale(trail[i].alt_m);
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Distance scale (right axis)
        ctx.fillStyle = '#f0883e';
        ctx.textAlign = 'left';
        for (let d = 0; d <= maxDist; d += stationRadius) {
            const y = yDistScale(d);
            ctx.fillText(`${(d/1000).toFixed(0)}km`, W - margin.right + 5, y + 4);
        }

        // Legend
        ctx.font = '10px monospace';
        ctx.textAlign = 'left';
        ctx.fillStyle = '#58a6ff';
        ctx.fillText('altitude', margin.left + 5, margin.top - 5);
        ctx.fillStyle = '#f0883e';
        ctx.fillText('distance', margin.left + 70, margin.top - 5);

        ctx.lineWidth = 1;
    }
}

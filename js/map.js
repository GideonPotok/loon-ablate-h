/**
 * map.js — Leaflet map integration with trajectory rendering.
 *
 * Features:
 * - OpenStreetMap base layer
 * - Balloon position marker with altitude coloring
 * - Trail polyline colored by altitude
 * - Target marker (draggable)
 * - Station-keeping radius circle
 * - Wind barbs overlay at current position
 * - Click to reposition target
 */
import { DEFAULT_MAP, runtime } from './config.js';

const ALT_MIN = 15000, ALT_MAX = 22000;

function altitudeColor(alt_m) {
    const t = Math.max(0, Math.min(1, (alt_m - ALT_MIN) / (ALT_MAX - ALT_MIN)));
    // Blue → Cyan → Green → Yellow → Red
    if (t < 0.25) {
        const s = t / 0.25;
        return `rgb(${Math.round(50 * (1-s))}, ${Math.round(100 + 155 * s)}, 255)`;
    } else if (t < 0.5) {
        const s = (t - 0.25) / 0.25;
        return `rgb(0, 255, ${Math.round(255 * (1-s))})`;
    } else if (t < 0.75) {
        const s = (t - 0.5) / 0.25;
        return `rgb(${Math.round(255 * s)}, 255, 0)`;
    } else {
        const s = (t - 0.75) / 0.25;
        return `rgb(255, ${Math.round(255 * (1-s))}, 0)`;
    }
}

export class MapView {
    constructor(containerId) {
        this.map = L.map(containerId, {
            center: [DEFAULT_MAP.CENTER_LAT, DEFAULT_MAP.CENTER_LON],
            zoom: DEFAULT_MAP.ZOOM,
            zoomControl: true,
        });

        L.tileLayer(DEFAULT_MAP.TILE_URL, {
            attribution: DEFAULT_MAP.TILE_ATTR,
            maxZoom: 18,
        }).addTo(this.map);

        // Balloon marker
        this.balloonMarker = L.circleMarker([0, 0], {
            radius: 8, color: '#fff', weight: 2, fillColor: '#3388ff',
            fillOpacity: 1, className: 'balloon-marker',
        }).addTo(this.map);

        // Target marker (draggable)
        this.targetMarker = L.marker([0, 170], {
            draggable: true,
            icon: L.divIcon({
                className: 'target-icon',
                html: '<div style="width:20px;height:20px;border:3px solid #e74c3c;border-radius:50%;position:relative"><div style="position:absolute;top:50%;left:0;right:0;height:1px;background:#e74c3c"></div><div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#e74c3c"></div></div>',
                iconSize: [20, 20],
                iconAnchor: [10, 10],
            }),
        }).addTo(this.map);

        // Station radius circle
        this.radiusCircle = L.circle([0, 170], {
            radius: runtime.platform.STATION_RADIUS_M,
            color: '#e74c3c', weight: 1, fillColor: '#e74c3c',
            fillOpacity: 0.05, dashArray: '5,5',
        }).addTo(this.map);

        // Trail segments (colored polylines)
        this.trailSegments = [];
        this.trailLayer = L.layerGroup().addTo(this.map);

        // Wind barbs layer
        this.windLayer = L.layerGroup().addTo(this.map);

        // Predicted trajectory
        this.predictionLine = L.polyline([], {
            color: '#9b59b6', weight: 2, dashArray: '4,8', opacity: 0.7,
        }).addTo(this.map);

        // Event handlers
        this._onTargetDrag = null;
        this._onMapClick = null;

        this.targetMarker.on('dragend', () => {
            const pos = this.targetMarker.getLatLng();
            if (this._onTargetDrag) this._onTargetDrag(pos.lat, pos.lng);
        });

        this.map.on('click', (e) => {
            if (this._onMapClick) this._onMapClick(e.latlng.lat, e.latlng.lng);
        });
    }

    onTargetDrag(fn) { this._onTargetDrag = fn; }
    onMapClick(fn) { this._onMapClick = fn; }

    /**
     * Update balloon position and trail from simulator snapshot.
     */
    updateFromSnapshot(snapshot, trail) {
        if (!snapshot.state) return;
        const { lat, lon, alt_m } = snapshot.state;
        const color = altitudeColor(alt_m);

        // Update balloon marker
        this.balloonMarker.setLatLng([lat, lon]);
        this.balloonMarker.setStyle({ fillColor: color });

        // Update trail (render last N segments for performance)
        this._updateTrail(trail);

        // Update radius circle position
        this.radiusCircle.setLatLng([snapshot.targetLat, snapshot.targetLon]);
        this.radiusCircle.setRadius(runtime.platform.STATION_RADIUS_M);
    }

    _updateTrail(trail) {
        this.trailLayer.clearLayers();
        if (trail.length < 2) return;

        // Subsample for performance: keep every Nth point
        const maxPoints = 2000;
        const step = Math.max(1, Math.floor(trail.length / maxPoints));
        const sampled = [];
        for (let i = 0; i < trail.length; i += step) sampled.push(trail[i]);
        if (sampled[sampled.length - 1] !== trail[trail.length - 1]) {
            sampled.push(trail[trail.length - 1]);
        }

        // Draw colored segments
        for (let i = 1; i < sampled.length; i++) {
            const p0 = sampled[i - 1], p1 = sampled[i];
            const color = altitudeColor((p0.alt_m + p1.alt_m) / 2);
            L.polyline([[p0.lat, p0.lon], [p1.lat, p1.lon]], {
                color, weight: 2, opacity: 0.8,
            }).addTo(this.trailLayer);
        }
    }

    /**
     * Update target marker and radius.
     */
    updateTarget(lat, lon) {
        this.targetMarker.setLatLng([lat, lon]);
        this.radiusCircle.setLatLng([lat, lon]);
    }

    /**
     * Draw wind barbs at the balloon's position.
     */
    updateWindBarbs(lat, lon, windColumn) {
        this.windLayer.clearLayers();
        // Show a single wind arrow at balloon position
        if (windColumn.length === 0) return;

        // Use the wind at the balloon's current altitude
        const w = windColumn[0]; // Caller passes current-altitude wind
        if (!w) return;

        const speed = Math.sqrt(w.u * w.u + w.v * w.v);
        const angle = Math.atan2(w.u, w.v) * 180 / Math.PI;
        const len = Math.min(speed * 3, 40);

        // Draw as a rotated arrow
        const arrowHtml = `<div style="transform:rotate(${angle}deg);width:2px;height:${len}px;background:#333;position:relative;margin:auto">
            <div style="position:absolute;top:0;left:-4px;border:5px solid transparent;border-bottom:8px solid #333"></div>
        </div>`;

        L.marker([lat, lon], {
            icon: L.divIcon({
                className: 'wind-arrow',
                html: arrowHtml,
                iconSize: [10, len + 8],
                iconAnchor: [5, len / 2 + 4],
            }),
            interactive: false,
        }).addTo(this.windLayer);
    }

    /**
     * Show predicted trajectory as a dashed line.
     */
    showPrediction(points) {
        this.predictionLine.setLatLngs(points.map(p => [p.lat, p.lon]));
    }

    /**
     * Center map on a position.
     */
    panTo(lat, lon) {
        this.map.panTo([lat, lon]);
    }

    /**
     * Fit map to show both balloon and target.
     */
    fitBounds(balloonLat, balloonLon, targetLat, targetLon) {
        const bounds = L.latLngBounds(
            [balloonLat, balloonLon],
            [targetLat, targetLon]
        ).pad(0.3);
        this.map.fitBounds(bounds);
    }

    resize() {
        this.map.invalidateSize();
    }
}

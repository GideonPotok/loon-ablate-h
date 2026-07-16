"""
replay_gassand.py — Roll out a policy in the helium/sand gas-balloon env and
plot its behaviour, including the finite-reserve DEPLETION over time.

Unlike replay.py / make_gif.py (which drive the v1/v2 ballast-pump env with a
trained QR-DQN), this targets `server_version='gassand'`, whose altitude control
is one-way and metered: drop finite SAND → rise, vent finite HELIUM → sink. The
headline addition here is a depletion panel — how the two reserves drain as the
policy flies — in both the static PNG and the animated GIF.

Two drivers, same panels (so heuristic vs. learned render identically):
  * default            — the env's built-in wind-follower heuristic
                         (`env.heuristic_step()`; needs no weights, no torch).
  * --weight <path.pt> — a QR-DQN trained on the gassand env
                         (state_dim=21, action_count=11).

Usage:
    python replay_gassand.py                         # heuristic, all 3 presets
    python replay_gassand.py --preset calm           # one preset
    python replay_gassand.py --duration 86400        # 24 h episode
    python replay_gassand.py --weight weights/dqn_gassand_w00.pt --tag learned
    python replay_gassand.py --no-gif                # PNG only (faster)

Outputs:  replay_gassand_<tag>_<preset>.png  (+ .gif unless --no-gif)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from balloon_env import BalloonEnv

# ── Constants (mirror servers/balloon_env_server_gassand.mjs + DEFAULT_GASSAND) ─

STATION_LAT      = 0.0
STATION_LON      = 170.0
STATION_RADIUS_M = 50_000

HELIUM_CAP_KG    = 19.24165      # HELIUM_CAPACITY_KG
SAND_CAP_KG      = 20.0          # SAND_CAPACITY_KG
ALT_MIN_M        = 15_000
ALT_MAX_M        = 22_000
CEILING_M        = 17_100        # balanced-launch zero-pressure ceiling (design doc)
MAX_VV_M_S       = 2.5

PRESETS = ['tropical', 'strong-shear', 'calm']
PRESET_COLORS = {
    'tropical':     '#e67e22',
    'strong-shear': '#c0392b',
    'calm':         '#27ae60',
}
# Resource colours, used consistently across PNG + GIF.
HE_COLOR   = '#5dade2'      # helium = lift gas (blue)
SAND_COLOR = '#d4a24e'      # sand   = ballast  (tan)

# 11-action release ladder (idx → label + signed "altitude-rate rung").
# 0..4 drop sand (fast→slow up), 5 float, 6..10 vent helium (slow→fast down).
ACTION_LABELS = ['+sand 1.0', '+sand 0.30', '+sand 0.08', '+sand 0.02', '+sand 0.005',
                 'FLOAT',
                 '-He 0.0008', '-He 0.0032', '-He 0.0128', '-He 0.0481', '-He 0.1604']
FLOAT_INDEX = 5


def action_rung(idx: int) -> int:
    """Signed ladder rung: +5 fastest-up … 0 float … -5 fastest-down."""
    if idx < FLOAT_INDEX:
        return FLOAT_INDEX - idx           # 0→+5 (fastest up) … 4→+1
    if idx == FLOAT_INDEX:
        return 0
    return -(idx - FLOAT_INDEX)            # 6→-1 (slow down) … 10→-5


# ── Helpers ────────────────────────────────────────────────────────────────────

def m_to_deg_lat(m): return m / 111_320
def m_to_deg_lon(m, lat=STATION_LAT):
    return m / (111_320 * math.cos(math.radians(lat)))


def load_agent(weight_path: Path):
    """Load a gassand-trained QR-DQN (only imported when --weight is given)."""
    import torch
    from qr_agent import QRAgent, QRConfig
    ckpt  = torch.load(str(weight_path), map_location='cpu', weights_only=False)
    cfg_d = ckpt.get('config', {})
    config = QRConfig(
        state_dim    = cfg_d.get('state_dim',    21),
        hidden_sizes = cfg_d.get('hidden_sizes', [128, 64]),
        action_count = cfg_d.get('action_count', 11),
        n_quantiles  = cfg_d.get('n_quantiles',  1),
        cvar_alpha   = cfg_d.get('cvar_alpha',   1.0),
        epsilon_end  = 0.0,
        device       = 'cpu',
    )
    agent = QRAgent(config)
    agent.policy_net.load_state_dict(ckpt['policy_net'])
    agent.epsilon = 0.0
    return agent


def run_episode(preset: str, duration_s: float, seed: int, agent=None, flags=None) -> dict:
    """Roll out one episode; return a dict of per-decision arrays."""
    env   = BalloonEnv(preset=preset, duration_s=duration_s, seed=seed,
                       server_version='gassand', flags=flags)
    state = env.reset()
    if agent is not None:
        agent.reset_hidden()      # no-op unless recurrent

    keys = ('time_h', 'lat', 'lon', 'alt_m', 'dist_km', 'in_radius', 'vv',
            'action', 'rung', 'helium_kg', 'sand_kg', 'he_vented', 'sand_dropped',
            'reward')
    traj = {k: [] for k in keys}
    done, step = False, 0
    while not done:
        if agent is None:
            action, next_state, reward, done, info = env.heuristic_step()
        else:
            action = agent.select_action(state, greedy=True)
            next_state, reward, done, info = env.step(action)

        dist_m = info.get('dist_m', 0.0)
        traj['time_h'].append(info.get('time_s', step * 300) / 3600)
        traj['lat'].append(info.get('lat', STATION_LAT))
        traj['lon'].append(info.get('lon', STATION_LON))
        traj['alt_m'].append(info.get('alt_m', CEILING_M))
        traj['dist_km'].append(dist_m / 1000)
        traj['in_radius'].append(dist_m < STATION_RADIUS_M)
        traj['vv'].append(info.get('vv_peak_m_s', info.get('vv_m_s', 0.0)))
        traj['action'].append(int(action))
        traj['rung'].append(action_rung(int(action)))
        traj['helium_kg'].append(info.get('helium_kg', float('nan')))
        traj['sand_kg'].append(info.get('sand_kg', float('nan')))
        traj['he_vented'].append(info.get('helium_vented_kg', float('nan')))
        traj['sand_dropped'].append(info.get('sand_dropped_kg', float('nan')))
        traj['reward'].append(float(reward))
        state = next_state
        step += 1
    env.close()

    for k in keys:
        traj[k] = np.array(traj[k])
    traj['twr50']    = float(traj['in_radius'].mean()) if step else 0.0
    traj['n_steps']  = step
    # Summary reserve figures for annotations.
    traj['he_left']       = float(traj['helium_kg'][-1])  if step else HELIUM_CAP_KG
    traj['sand_left']     = float(traj['sand_kg'][-1])    if step else SAND_CAP_KG
    traj['he_used']       = float(traj['he_vented'][-1])  if step else 0.0
    traj['sand_used']     = float(traj['sand_dropped'][-1]) if step else 0.0
    return traj


# ── Static PNG ─────────────────────────────────────────────────────────────────

def plot_episode(traj: dict, preset: str, out_path: Path, label: str):
    color = PRESET_COLORS.get(preset, '#3498db')
    lats, lons = traj['lat'], traj['lon']
    alts       = traj['alt_m']
    dists      = traj['dist_km']
    times      = traj['time_h']
    in_r       = traj['in_radius']
    he, sand   = traj['helium_kg'], traj['sand_kg']
    rungs      = traj['rung']
    twr50      = traj['twr50']

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'{label} — {preset}  |  TWR50 = {twr50*100:.1f}%  |  {traj["n_steps"]} steps '
        f'({times[-1]:.0f} h)   ·   He left {traj["he_left"]:.2f}/{HELIUM_CAP_KG:.1f} kg '
        f'· sand left {traj["sand_left"]:.2f}/{SAND_CAP_KG:.0f} kg',
        fontsize=12.5, fontweight='bold',
    )
    gs = fig.add_gridspec(2, 3, hspace=0.40, wspace=0.34)

    # ── Panels 1 & 4: trajectory maps (zoom out / zoom in) ────────────────────
    r_lat, r_lon = m_to_deg_lat(STATION_RADIUS_M), m_to_deg_lon(STATION_RADIUS_M)
    theta = np.linspace(0, 2 * math.pi, 200)
    circ_lat = STATION_LAT + r_lat * np.sin(theta)
    circ_lon = STATION_LON + r_lon * np.cos(theta)

    def _draw_map(ax, xlim, ylim, title):
        ax.fill(circ_lon, circ_lat, alpha=0.12, color=color, zorder=0)
        ax.plot(circ_lon, circ_lat, color=color, lw=1.2, ls='--', zorder=1)
        ax.plot(STATION_LON, STATION_LAT, '*', color=color, ms=12, zorder=3)
        points = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segs   = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = ['#2ecc71' if i else '#e74c3c' for i in in_r[1:]]
        ax.add_collection(LineCollection(segs, colors=seg_colors, linewidths=1.2, zorder=2))
        ax.plot(lons[0], lats[0], 'o', color='#2c3e50', ms=6, zorder=4, label='start')
        ax.plot(lons[-1], lats[-1], 's', color='#2c3e50', ms=6, zorder=4, label='end')
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)', fontsize=9)
        ax.set_title(title, fontsize=9)

    lon_dev = max(np.abs(lons - STATION_LON).max(), r_lon * 1.2) * 1.12
    lat_dev = max(np.abs(lats - STATION_LAT).max(), r_lat * 1.2) * 1.12
    ax_map_out = fig.add_subplot(gs[0, 0])
    _draw_map(ax_map_out, (STATION_LON - lon_dev, STATION_LON + lon_dev),
              (STATION_LAT - lat_dev, STATION_LAT + lat_dev),
              'Full trajectory (zoomed out)\n(green = in radius, red = out)')
    ax_map_out.legend(fontsize=7, loc='upper right')

    zi = 1.7
    ax_map_in = fig.add_subplot(gs[1, 0])
    _draw_map(ax_map_in, (STATION_LON - r_lon * zi, STATION_LON + r_lon * zi),
              (STATION_LAT - r_lat * zi, STATION_LAT + r_lat * zi),
              'Near station (zoomed in)')

    # ── Panel 2: altitude over time ───────────────────────────────────────────
    ax_alt = fig.add_subplot(gs[0, 1])
    alt_colors = ['#2ecc71' if i else '#e74c3c' for i in in_r]
    ax_alt.scatter(times, alts / 1000, c=alt_colors, s=2, zorder=2)
    ax_alt.axhline(CEILING_M / 1000, color='gray', lw=0.9, ls='--')
    ax_alt.text(times[-1], CEILING_M / 1000, ' launch ceiling', color='gray',
                fontsize=6.5, va='bottom', ha='right')
    ax_alt.set_ylim(ALT_MIN_M / 1000 - 0.2, ALT_MAX_M / 1000 + 0.2)
    ax_alt.set_xlabel('Time (h)', fontsize=9)
    ax_alt.set_ylabel('Altitude (km)', fontsize=9)
    ax_alt.set_title('Altitude over time', fontsize=9)

    # ── Panel 5 (star): reserve DEPLETION over time ───────────────────────────
    ax_dep = fig.add_subplot(gs[1, 1])
    ax_dep.plot(times, 100 * he / HELIUM_CAP_KG, color=HE_COLOR, lw=1.8,
                label=f'helium (lift, {HELIUM_CAP_KG:.1f} kg full)')
    ax_dep.plot(times, 100 * sand / SAND_CAP_KG, color=SAND_COLOR, lw=1.8,
                label=f'sand (ballast, {SAND_CAP_KG:.0f} kg full)')
    ax_dep.fill_between(times, 0, 100 * he / HELIUM_CAP_KG,   color=HE_COLOR,   alpha=0.10)
    ax_dep.fill_between(times, 0, 100 * sand / SAND_CAP_KG,   color=SAND_COLOR, alpha=0.10)
    ax_dep.set_ylim(0, 105)
    ax_dep.set_xlabel('Time (h)', fontsize=9)
    ax_dep.set_ylabel('Reserve remaining (% of launch)', fontsize=9)
    ax_dep.set_title('Finite-reserve depletion  (one-way: vent He / drop sand)', fontsize=9)
    ax_dep.legend(fontsize=7, loc='lower left')

    # ── Panel 3: distance over time ───────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_dist.plot(times, dists, lw=0.8, color=color, zorder=2)
    ax_dist.axhline(STATION_RADIUS_M / 1000, color='gray', lw=1.0, ls='--', label='50 km radius')
    ax_dist.fill_between(times, 0, STATION_RADIUS_M / 1000, alpha=0.08, color='#2ecc71')
    ax_dist.set_xlabel('Time (h)', fontsize=9)
    ax_dist.set_ylabel('Distance from station (km)', fontsize=9)
    ax_dist.set_title('Distance over time', fontsize=9)
    ax_dist.legend(fontsize=7)

    # ── Panel 6: action ladder over time ──────────────────────────────────────
    ax_act = fig.add_subplot(gs[1, 2])
    rung_colors = [SAND_COLOR if r > 0 else (HE_COLOR if r < 0 else '#888')
                   for r in rungs]
    ax_act.scatter(times, rungs, c=rung_colors, s=5, zorder=2)
    ax_act.axhline(0, color='gray', lw=0.8, ls=':')
    ax_act.set_ylim(-5.5, 5.5)
    ax_act.set_yticks([-5, -3, -1, 0, 1, 3, 5])
    ax_act.set_yticklabels(['He fast', 'He', 'He slow', 'float', 'sand slow', 'sand', 'sand fast'],
                           fontsize=6.5)
    ax_act.set_xlabel('Time (h)', fontsize=9)
    ax_act.set_title('Release command over time\n(sand = up, helium = down)', fontsize=9)

    # shade out-of-radius spans on the time-series panels
    for ax in (ax_alt, ax_dep, ax_dist, ax_act):
        seg = None
        for i, ir in enumerate(in_r):
            if not ir and seg is None:
                seg = times[i]
            elif ir and seg is not None:
                ax.axvspan(seg, times[i], alpha=0.06, color='#e74c3c', zorder=0); seg = None
        if seg is not None:
            ax.axvspan(seg, times[-1], alpha=0.06, color='#e74c3c', zorder=0)

    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── Animated GIF ───────────────────────────────────────────────────────────────

def make_gif(traj: dict, preset: str, out_path: Path, label: str, stride=4, fps=12):
    from matplotlib.animation import FuncAnimation, PillowWriter

    color = PRESET_COLORS.get(preset, '#3498db')
    lats, lons = traj['lat'], traj['lon']
    alts       = traj['alt_m']
    dists      = traj['dist_km']
    times      = traj['time_h']
    in_r       = traj['in_radius']
    he, sand   = traj['helium_kg'], traj['sand_kg']
    he_v, sd_d = traj['he_vented'], traj['sand_dropped']
    twr50      = traj['twr50']
    n          = len(times)

    r_lat, r_lon = m_to_deg_lat(STATION_RADIUS_M), m_to_deg_lon(STATION_RADIUS_M)
    theta = np.linspace(0, 2 * math.pi, 300)
    circ_lat = STATION_LAT + r_lat * np.sin(theta)
    circ_lon = STATION_LON + r_lon * np.cos(theta)
    lon_dev = max(abs(lons - STATION_LON).max(), r_lon * 1.5) + r_lon * 0.4
    lat_dev = max(abs(lats - STATION_LAT).max(), r_lat * 1.5) + r_lat * 0.4

    fig = plt.figure(figsize=(16, 8), facecolor='#12121f')
    fig.suptitle(f'{label} — {preset}  |  TWR50 = {twr50*100:.1f}%',
                 color='white', fontsize=13, fontweight='bold', y=0.97)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.62, wspace=0.30,
                          left=0.07, right=0.95, top=0.91, bottom=0.08)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_alt  = fig.add_subplot(gs[1, 1])
    ax_dep  = fig.add_subplot(gs[2, 1])
    for ax in (ax_map, ax_dist, ax_alt, ax_dep):
        ax.set_facecolor('#0a0a18')
        ax.tick_params(colors='#888899', labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')

    # static backdrops
    ax_map.plot(lons, lats, lw=0.5, color='#2a2a4a', zorder=1)
    ax_map.fill(circ_lon, circ_lat, alpha=0.10, color=color, zorder=0)
    ax_map.plot(circ_lon, circ_lat, color=color, lw=1.2, ls='--', alpha=0.5, zorder=1)
    for mult in (3, 5, 10):
        rm = STATION_RADIUS_M * mult
        ax_map.plot(STATION_LON + m_to_deg_lon(rm) * np.cos(theta),
                    STATION_LAT + m_to_deg_lat(rm) * np.sin(theta),
                    color='#1e1e33', lw=0.5, ls=':', zorder=0)
    ax_map.plot(STATION_LON, STATION_LAT, '*', color='white', ms=9, zorder=4, alpha=0.9)
    ax_map.set_xlim(STATION_LON - lon_dev, STATION_LON + lon_dev)
    ax_map.set_ylim(STATION_LAT - lat_dev, STATION_LAT + lat_dev)
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlabel('Longitude (°)', color='#888899', fontsize=8)
    ax_map.set_ylabel('Latitude (°)', color='#888899', fontsize=8)
    ax_map.set_title('Balloon position  (★=station, dashed=50 km radius)',
                     color='white', fontsize=9)

    ax_dist.set_xlim(0, times[-1])
    ax_dist.set_ylim(0, max(dists.max() * 1.1, STATION_RADIUS_M / 1000 * 1.5))
    ax_dist.axhline(STATION_RADIUS_M / 1000, color='#556688', lw=0.9, ls='--')
    ax_dist.fill_between([0, times[-1]], 0, STATION_RADIUS_M / 1000, alpha=0.07, color='#2ecc71')
    ax_dist.set_ylabel('km', color='#888899', fontsize=7)
    ax_dist.set_title('Distance from station', color='white', fontsize=8)

    ax_alt.set_xlim(0, times[-1])
    ax_alt.set_ylim(ALT_MIN_M / 1000 - 0.2, ALT_MAX_M / 1000 + 0.2)
    ax_alt.axhline(CEILING_M / 1000, color='#444455', lw=0.7, ls='--')
    ax_alt.set_ylabel('km', color='#888899', fontsize=7)
    ax_alt.set_title('Altitude', color='white', fontsize=8)

    ax_dep.set_xlim(0, times[-1])
    ax_dep.set_ylim(0, 105)
    ax_dep.set_ylabel('% left', color='#888899', fontsize=7)
    ax_dep.set_xlabel('Time (h)', color='#888899', fontsize=7)
    ax_dep.set_title('Reserve depletion (blue=He / tan=sand)', color='white', fontsize=8)

    # side fuel-gauge bars (drain per frame), overlaid on the map axis
    gax = fig.add_axes([0.015, 0.10, 0.035, 0.34]); gax.set_facecolor('#0a0a18')
    gax.set_xlim(0, 2); gax.set_ylim(0, 100); gax.set_xticks([])
    gax.tick_params(colors='#888899', labelsize=6)
    for sp in gax.spines.values(): sp.set_edgecolor('#333355')
    gax.set_title('fuel', color='white', fontsize=7)
    he_bar   = gax.bar(0.5, 100, width=0.7, color=HE_COLOR,   align='center')[0]
    sand_bar = gax.bar(1.5, 100, width=0.7, color=SAND_COLOR, align='center')[0]
    gax.set_xticks([0.5, 1.5]); gax.set_xticklabels(['He', 'sd'], color='#888899', fontsize=6)

    # animated artists
    trail,   = ax_map.plot([], [], lw=2.2, zorder=2, solid_capstyle='round')
    balloon, = ax_map.plot([], [], 'o', ms=11, zorder=5,
                           markeredgecolor='white', markeredgewidth=1.0)
    time_txt = ax_map.text(0.03, 0.97, '', transform=ax_map.transAxes, color='white',
                           fontsize=8, va='top', family='monospace',
                           bbox=dict(facecolor='#12121f', alpha=0.65, pad=3, edgecolor='none'))
    dist_line, = ax_dist.plot([], [], lw=1.2, color=color)
    dist_dot,  = ax_dist.plot([], [], 'o', color='white', ms=4)
    alt_line,  = ax_alt.plot([], [], lw=1.2, color='#5dade2')
    alt_dot,   = ax_alt.plot([], [], 'o', color='white', ms=4)
    he_line,   = ax_dep.plot([], [], lw=1.6, color=HE_COLOR)
    sand_line, = ax_dep.plot([], [], lw=1.6, color=SAND_COLOR)
    he_dot,    = ax_dep.plot([], [], 'o', color='white', ms=3)

    tail = 60
    frames = list(range(0, n, stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    def init():
        for a in (trail, balloon, dist_line, dist_dot, alt_line, alt_dot,
                  he_line, sand_line, he_dot):
            a.set_data([], [])
        time_txt.set_text('')
        return (trail, balloon, dist_line, dist_dot, alt_line, alt_dot,
                he_line, sand_line, he_dot, he_bar, sand_bar)

    def update(f):
        lo = max(0, f - tail)
        trail.set_data(lons[lo:f + 1], lats[lo:f + 1])
        trail.set_color('#2ecc71' if in_r[f] else '#e74c3c')
        balloon.set_data([lons[f]], [lats[f]])
        balloon.set_color('#2ecc71' if in_r[f] else '#e74c3c')
        dist_line.set_data(times[:f + 1], dists[:f + 1])
        dist_dot.set_data([times[f]], [dists[f]])
        alt_line.set_data(times[:f + 1], alts[:f + 1] / 1000)
        alt_dot.set_data([times[f]], [alts[f] / 1000])
        he_pct   = 100 * he[:f + 1] / HELIUM_CAP_KG
        sand_pct = 100 * sand[:f + 1] / SAND_CAP_KG
        he_line.set_data(times[:f + 1], he_pct)
        sand_line.set_data(times[:f + 1], sand_pct)
        he_dot.set_data([times[f]], [he_pct[-1]])
        he_bar.set_height(100 * he[f] / HELIUM_CAP_KG)
        sand_bar.set_height(100 * sand[f] / SAND_CAP_KG)
        time_txt.set_text(
            f't={times[f]:5.1f}h  alt={alts[f]/1000:5.2f}km\n'
            f'He {he[f]:5.2f}kg (vent {he_v[f]:4.2f})\n'
            f'sd {sand[f]:5.2f}kg (drop {sd_d[f]:4.2f})')
        return (trail, balloon, dist_line, dist_dot, alt_line, alt_dot,
                he_line, sand_line, he_dot, he_bar, sand_bar, time_txt)

    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                         blit=False, interval=1000 / fps)
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weight', default=None,
                   help='Path to a gassand-trained .pt checkpoint (default: built-in heuristic)')
    p.add_argument('--tag', default=None,
                   help="Filename tag (default: 'heuristic' or 'learned')")
    p.add_argument('--preset', default=None, help='tropical | strong-shear | calm (default: all)')
    p.add_argument('--duration', type=float, default=3600 * 72, help='episode length (s), default 72h')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--no-gif', action='store_true', help='PNG only (skip the animated GIF)')
    p.add_argument('--realism', action='store_true',
                   help="Roll out under R's 4 realism flags (match an R_Gassand-trained policy)")
    args = p.parse_args()

    agent = None
    if args.weight:
        agent = load_agent(Path(args.weight))
    tag   = args.tag or ('learned' if args.weight else 'heuristic')
    flags = ({'wind_phase_jitter': True, 'wind_episode_noise': True,
              'wind_param_jitter': True, 'domain_rand': True} if args.realism else None)
    label = f'Gassand ({tag}{", realism" if flags else ""})'

    presets = [args.preset] if args.preset else PRESETS
    for preset in presets:
        print(f'[{preset}] rolling out ({tag}{", realism" if flags else ""}) …')
        traj = run_episode(preset, args.duration, args.seed, agent=agent, flags=flags)
        print(f'  TWR50 {traj["twr50"]*100:.1f}%  ·  He {traj["he_left"]:.2f}kg left '
              f'(vented {traj["he_used"]:.2f})  ·  sand {traj["sand_left"]:.2f}kg left '
              f'(dropped {traj["sand_used"]:.2f})')
        fslug = preset.replace('-', '_')
        plot_episode(traj, preset, Path(f'replay_gassand_{tag}_{fslug}.png'), label)
        if not args.no_gif:
            make_gif(traj, preset, Path(f'replay_gassand_{tag}_{fslug}.gif'), label)


if __name__ == '__main__':
    main()

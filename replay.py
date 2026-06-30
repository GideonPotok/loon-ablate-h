"""
replay.py — Run the trained agent and plot its behaviour.

Usage:
    python replay.py                        # uses best weight (w00)
    python replay.py --weight weights/final-ablate-h/dqn_ablate_h.pt
    python replay.py --ablation k2          # v2 server + that ablation's env flags
    python replay.py --preset tropical      # single preset
    python replay.py --duration 43200       # 12-hour episode (default 72h)
    python replay.py --seed 7

Outputs:  replay_<preset>.png  (one per preset, or one if --preset given)
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

from qr_agent import QRAgent, QRConfig
from balloon_env import BalloonEnv

# ── Constants ─────────────────────────────────────────────────────────────────

STATION_LAT      = 0.0
STATION_LON      = 170.0
STATION_RADIUS_M = 50_000
ALT_BAND_LOW_M   = 15_500       # approx from config
ALT_BAND_HIGH_M  = 19_500

PRESETS = ['tropical', 'strong-shear', 'calm']
PRESET_COLORS = {
    'tropical':     '#e67e22',
    'strong-shear': '#c0392b',
    'calm':         '#27ae60',
}

# Env flags as actually used at training time (mirrors make_gif.py).
# server_version='v2' for both; K2 = 20-dim state, L = 24-dim w/ Fourier time features.
ABLATION_ENV_FLAGS = {
    'k2': {
        'use_reward_fix':     True,
        'use_shaping':        True,
        'use_expanded_state': False,
        'shaping_beta':       0.5,
        'shaping_gamma':      0.97,
        'terminal_twr_bonus': 50.0,
        'shaping_linear':     False,            # exponential shaping
        'shaping_D_max':      500_000.0,        # tau = 500 km
    },
    'l': {
        'use_reward_fix':     True,
        'use_shaping':        True,
        'use_expanded_state': False,
        'use_time_features':  True,             # 20 -> 24 dim
        'shaping_beta':       0.5,
        'shaping_gamma':      0.97,
        'terminal_twr_bonus': 50.0,
        'shaping_linear':     False,
        'shaping_D_max':      500_000.0,
    },
}
ABLATION_LABELS = {
    'k2': 'Ablation K2 (exp shaping, tau=500km)',
    'l':  'Ablation L (+ Fourier time features)',
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def m_to_deg_lat(m): return m / 111_320

def m_to_deg_lon(m, lat_deg=STATION_LAT):
    return m / (111_320 * math.cos(math.radians(lat_deg))) if lat_deg != 90 else 0


def load_agent(weight_path: Path) -> QRAgent:
    ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=False)
    cfg_d = ckpt.get('config', {})
    config = QRConfig(
        state_dim    = cfg_d.get('state_dim',    20),
        hidden_sizes = cfg_d.get('hidden_sizes', [128, 64]),
        action_count = cfg_d.get('action_count', 17),
        n_quantiles  = cfg_d.get('n_quantiles',  1),
        cvar_alpha   = cfg_d.get('cvar_alpha',   1.0),
        epsilon_end  = 0.0,   # greedy at eval
        device       = 'cpu',
    )
    agent = QRAgent(config)
    agent.policy_net.load_state_dict(ckpt['policy_net'])
    agent.epsilon = 0.0       # force greedy
    return agent


def run_episode(agent: QRAgent, preset: str, duration_s: float, seed: int,
                server_version: str = 'v1', flags: dict | None = None) -> dict:
    """Return a dict of lists: time_s, lat, lon, alt_m, dist_m, action, reward, in_radius."""
    env = BalloonEnv(preset=preset, duration_s=duration_s, seed=seed,
                     server_version=server_version, flags=flags)
    state = env.reset()

    traj = {k: [] for k in ('time_s', 'lat', 'lon', 'alt_m', 'dist_m', 'action', 'reward', 'in_radius')}
    done = False
    step = 0

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)

        time_s  = info.get('time_s',  step * 300)
        lat     = info.get('lat',     STATION_LAT)
        lon     = info.get('lon',     STATION_LON)
        alt_m   = info.get('alt_m',   17_000)
        dist_m  = info.get('dist_m',  0)

        traj['time_s'].append(time_s / 3600)   # convert to hours
        traj['lat'].append(lat)
        traj['lon'].append(lon)
        traj['alt_m'].append(alt_m)
        traj['dist_m'].append(dist_m / 1000)   # convert to km
        traj['action'].append(action)
        traj['reward'].append(reward)
        traj['in_radius'].append(dist_m < STATION_RADIUS_M)

        state = next_state
        step += 1

    env.close()

    twr50 = sum(traj['in_radius']) / max(len(traj['in_radius']), 1)
    traj['twr50'] = twr50
    traj['n_steps'] = step
    return traj


def plot_episode(traj: dict, preset: str, out_path: Path, label: str = 'Ablation H (w00)'):
    color = PRESET_COLORS.get(preset, '#3498db')
    lats  = np.array(traj['lat'])
    lons  = np.array(traj['lon'])
    alts  = np.array(traj['alt_m'])
    dists = np.array(traj['dist_m'])
    times = np.array(traj['time_s'])
    acts  = np.array(traj['action'])
    in_r  = np.array(traj['in_radius'])
    twr50 = traj['twr50']

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        f'{label} — {preset}  |  TWR50 = {twr50*100:.1f}%  |  '
        f'{traj["n_steps"]} steps  ({times[-1]:.0f} h)',
        fontsize=13, fontweight='bold',
    )
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # ── Panel 1: lat/lon map ──────────────────────────────────────────────────
    ax_map = fig.add_subplot(gs[:, 0])   # spans both rows

    # station circle (convert radius to degrees for rough scale)
    r_lat = m_to_deg_lat(STATION_RADIUS_M)
    r_lon = m_to_deg_lon(STATION_RADIUS_M)
    theta = np.linspace(0, 2 * math.pi, 200)
    circ_lat = STATION_LAT + r_lat * np.sin(theta)
    circ_lon = STATION_LON + r_lon * np.cos(theta)
    ax_map.fill(circ_lon, circ_lat, alpha=0.12, color=color, zorder=0)
    ax_map.plot(circ_lon, circ_lat, color=color, lw=1.2, ls='--', zorder=1)
    ax_map.plot(STATION_LON, STATION_LAT, '*', color=color, ms=12, zorder=3)

    # trajectory coloured by in/out radius
    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segs   = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = ['#2ecc71' if i else '#e74c3c' for i in in_r[1:]]
    lc = LineCollection(segs, colors=seg_colors, linewidths=1.2, zorder=2)
    ax_map.add_collection(lc)
    ax_map.autoscale_view()

    ax_map.set_xlabel('Longitude (°)', fontsize=9)
    ax_map.set_ylabel('Latitude (°)', fontsize=9)
    ax_map.set_title('Balloon trajectory\n(green = in radius, red = out)', fontsize=9)
    ax_map.set_aspect('equal', adjustable='datalim')
    # start/end markers
    ax_map.plot(lons[0], lats[0], 'o', color='#2c3e50', ms=6, zorder=4, label='start')
    ax_map.plot(lons[-1], lats[-1], 's', color='#2c3e50', ms=6, zorder=4, label='end')
    ax_map.legend(fontsize=7, loc='upper right')

    # ── Panel 2: altitude vs time ─────────────────────────────────────────────
    ax_alt = fig.add_subplot(gs[0, 1])
    alt_colors = ['#2ecc71' if i else '#e74c3c' for i in in_r]
    ax_alt.scatter(times, alts / 1000, c=alt_colors, s=2, zorder=2)
    ax_alt.set_xlabel('Time (h)', fontsize=9)
    ax_alt.set_ylabel('Altitude (km)', fontsize=9)
    ax_alt.set_title('Altitude over time', fontsize=9)
    ax_alt.axhline(ALT_BAND_LOW_M / 1000,  color='gray', lw=0.8, ls=':')
    ax_alt.axhline(ALT_BAND_HIGH_M / 1000, color='gray', lw=0.8, ls=':')

    # ── Panel 3: distance vs time ─────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[0, 2])
    ax_dist.plot(times, dists, lw=0.8, color=color, zorder=2)
    ax_dist.axhline(STATION_RADIUS_M / 1000, color='gray', lw=1.0, ls='--', label='50 km radius')
    ax_dist.fill_between(times, 0, STATION_RADIUS_M / 1000, alpha=0.08, color='#2ecc71')
    ax_dist.set_xlabel('Time (h)', fontsize=9)
    ax_dist.set_ylabel('Distance from station (km)', fontsize=9)
    ax_dist.set_title('Distance over time', fontsize=9)
    ax_dist.legend(fontsize=7)

    # ── Panel 4: action histogram ─────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 1])
    alt_bins = np.linspace(ALT_BAND_LOW_M, ALT_BAND_HIGH_M, 18) / 1000
    act_alts = ALT_BAND_LOW_M + (acts / 16) * (ALT_BAND_HIGH_M - ALT_BAND_LOW_M)
    ax_hist.hist(act_alts / 1000, bins=alt_bins, color=color, edgecolor='white', lw=0.5)
    ax_hist.set_xlabel('Target altitude (km)', fontsize=9)
    ax_hist.set_ylabel('# decisions', fontsize=9)
    ax_hist.set_title('Action distribution\n(target alt histogram)', fontsize=9)

    # ── Panel 5: action vs time ───────────────────────────────────────────────
    ax_act = fig.add_subplot(gs[1, 2])
    # map action 0–16 to altitude km
    act_alt_km = (ALT_BAND_LOW_M + (acts / 16) * (ALT_BAND_HIGH_M - ALT_BAND_LOW_M)) / 1000
    ax_act.plot(times, act_alt_km, lw=0.7, color=color)
    ax_act.set_xlabel('Time (h)', fontsize=9)
    ax_act.set_ylabel('Target altitude (km)', fontsize=9)
    ax_act.set_title('Action sequence over time', fontsize=9)
    ax_act.axhline(ALT_BAND_LOW_M / 1000,  color='gray', lw=0.8, ls=':')
    ax_act.axhline(ALT_BAND_HIGH_M / 1000, color='gray', lw=0.8, ls=':')

    # ── In-radius band on time-series panels ─────────────────────────────────
    for ax in (ax_alt, ax_dist, ax_act):
        # shade periods outside radius
        out_spans = []
        in_seg = None
        for i, ir in enumerate(in_r):
            t = times[i]
            if not ir and in_seg is None:
                in_seg = t
            elif ir and in_seg is not None:
                out_spans.append((in_seg, t))
                in_seg = None
        if in_seg is not None:
            out_spans.append((in_seg, times[-1]))
        for t0, t1 in out_spans:
            ax.axvspan(t0, t1, alpha=0.07, color='#e74c3c', zorder=0)

    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved → {out_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='weights/worker-0/dqn_ablate_h_w00.pt',
                        help='Path to .pt checkpoint')
    parser.add_argument('--ablation', default=None, choices=sorted(ABLATION_ENV_FLAGS),
                        help='If set, replays with that ablation\'s training-time env '
                             'flags + v2 server (e.g. k2, l). Omit for plain v1 (e.g. H).')
    parser.add_argument('--preset', default=None,
                        help='One of: tropical, strong-shear, calm  (default: all three)')
    parser.add_argument('--duration', type=float, default=3600 * 72,
                        help='Episode length in seconds (default: 72h)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--label', default=None,
                        help='Title label for the plot (default: derived from --ablation)')
    parser.add_argument('--out-prefix', default=None,
                        help='Output filename prefix (default: derived from --ablation, else "replay")')
    args = parser.parse_args()

    weight_path = Path(args.weight)
    if not weight_path.exists():
        raise FileNotFoundError(f'Weight file not found: {weight_path}')

    server_version = 'v2' if args.ablation else 'v1'
    flags          = ABLATION_ENV_FLAGS.get(args.ablation)
    label          = args.label or ABLATION_LABELS.get(args.ablation, 'Ablation H (w00)')
    out_prefix     = args.out_prefix or (f'replay_ablate_{args.ablation}' if args.ablation else 'replay')

    print(f'Loading agent from {weight_path}')
    agent = load_agent(weight_path)
    agent.epsilon = 0.0

    presets = [args.preset] if args.preset else PRESETS

    for preset in presets:
        print(f'\nRunning {preset} ({args.duration/3600:.0f} h)...')
        traj = run_episode(agent, preset, args.duration, args.seed,
                           server_version=server_version, flags=flags)
        print(f'  TWR50 = {traj["twr50"]*100:.1f}%  steps = {traj["n_steps"]}')
        out = Path(f'{out_prefix}_{preset.replace("-","_")}.png')
        plot_episode(traj, preset, out, label=label)

    print('\nDone.')


if __name__ == '__main__':
    main()

"""Generate animated GIFs of evaluation episodes for a given ablation's weights."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from qr_agent import QRAgent, QRConfig
from balloon_env import BalloonEnv

STATION_LAT      = 0.0
STATION_LON      = 170.0
STATION_RADIUS_M = 50_000
ALT_BAND_LOW_M   = 15_500
ALT_BAND_HIGH_M  = 19_500

PRESET_COLORS = {
    'tropical':     '#e67e22',
    'strong-shear': '#c0392b',
    'calm':         '#27ae60',
}

# Env flags as actually used at training time for each ablation
# (server_version='v2' for both; K2 = 20-dim state, L = 24-dim w/ Fourier time features)
ENV_FLAGS = {
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
    'm': {
        'use_reward_fix':     True,
        'use_shaping':        True,
        'use_expanded_state': False,
        'use_time_features':  True,             # 20 -> 24 dim (same as L)
        'shaping_beta':       0.5,
        'shaping_gamma':      0.97,
        'terminal_twr_bonus': 50.0,
        'shaping_linear':     False,
        'shaping_D_max':      500_000.0,
    },
}

LABELS = {
    'k2': 'Ablation K2 (exp shaping, tau=500km)',
    'l':  'Ablation L (+ Fourier time features)',
    'm':  'Ablation M (option-critic + GRU-64)',
}
# Architecture overrides not recoverable from the checkpoint's saved config
# (QRAgent.state_dict() only persists the feedforward-relevant fields).
AGENT_KWARGS = {
    'm': {
        'use_recurrent': True, 'use_options': True,
        'n_options': 4, 'gru_hidden': 64,
    },
}


def m_to_deg_lat(m): return m / 111_320
def m_to_deg_lon(m, lat=0.0):
    return m / (111_320 * math.cos(math.radians(lat)))


def load_agent(weight_path: Path, agent_kwargs: dict | None = None) -> QRAgent:
    ckpt = torch.load(str(weight_path), map_location='cpu', weights_only=False)
    cfg_d = ckpt.get('config', {})
    config = QRConfig(
        state_dim    = cfg_d.get('state_dim',    20),
        hidden_sizes = cfg_d.get('hidden_sizes', [128, 64]),
        action_count = cfg_d.get('action_count', 17),
        n_quantiles  = cfg_d.get('n_quantiles',  1),
        cvar_alpha   = cfg_d.get('cvar_alpha',   1.0),
        epsilon_end  = 0.0,
        device       = 'cpu',
        **(agent_kwargs or {}),
    )
    agent = QRAgent(config)
    agent.policy_net.load_state_dict(ckpt['policy_net'])
    agent.epsilon = 0.0
    return agent


def run_episode(agent, preset, duration_s, seed, flags):
    env = BalloonEnv(preset=preset, duration_s=duration_s, seed=seed,
                     server_version='v2', flags=flags)
    agent.reset_hidden()      # no-op unless agent.config.use_recurrent
    state = env.reset()
    traj = {k: [] for k in ('time_h', 'lat', 'lon', 'alt_m', 'dist_km', 'in_radius', 'reward')}
    done = False
    step = 0
    while not done:
        action = agent.select_action(state, greedy=True)
        next_state, reward, done, info = env.step(action)
        traj['time_h'].append(info.get('time_s', step * 300) / 3600)
        traj['lat'].append(info.get('lat', STATION_LAT))
        traj['lon'].append(info.get('lon', STATION_LON))
        traj['alt_m'].append(info.get('alt_m', 17_000))
        dist_m = info.get('dist_m', 0)
        traj['dist_km'].append(dist_m / 1000)
        traj['in_radius'].append(dist_m < STATION_RADIUS_M)
        traj['reward'].append(float(reward))
        state = next_state
        step += 1
    env.close()
    traj['twr50'] = sum(traj['in_radius']) / max(len(traj['in_radius']), 1)
    return {k: (np.array(v) if k != 'twr50' else v) for k, v in traj.items()}


def make_gif(traj, preset, out_path, label, stride=4, fps=12):
    color   = PRESET_COLORS.get(preset, '#3498db')
    lats    = traj['lat']
    lons    = traj['lon']
    alts    = traj['alt_m']
    dists   = traj['dist_km']
    times   = traj['time_h']
    in_r    = traj['in_radius']
    rewards = traj['reward']
    twr50   = traj['twr50']
    n       = len(times)

    r_lat = m_to_deg_lat(STATION_RADIUS_M)
    r_lon = m_to_deg_lon(STATION_RADIUS_M)
    theta = np.linspace(0, 2 * math.pi, 300)
    circ_lat = STATION_LAT + r_lat * np.sin(theta)
    circ_lon = STATION_LON + r_lon * np.cos(theta)

    traj_lon_dev = max(abs(lons - STATION_LON).max(), r_lon * 1.5) + r_lon * 0.4
    traj_lat_dev = max(abs(lats - STATION_LAT).max(), r_lat * 1.5) + r_lat * 0.4
    lon_lo = STATION_LON - traj_lon_dev
    lon_hi = STATION_LON + traj_lon_dev
    lat_lo = STATION_LAT - traj_lat_dev
    lat_hi = STATION_LAT + traj_lat_dev

    reward_roll = np.convolve(rewards, np.ones(12) / 12, mode='same')
    tail_len = 60

    fig = plt.figure(figsize=(16, 8), facecolor='#12121f')
    fig.suptitle(
        f'{label} — {preset}  |  TWR50 = {twr50*100:.1f}%',
        color='white', fontsize=13, fontweight='bold', y=0.97,
    )

    gs = fig.add_gridspec(3, 2, width_ratios=[1.2, 1], hspace=0.58, wspace=0.32,
                          left=0.07, right=0.97, top=0.91, bottom=0.07)
    ax_map  = fig.add_subplot(gs[:, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_alt  = fig.add_subplot(gs[1, 1])
    ax_rew  = fig.add_subplot(gs[2, 1])

    for ax in (ax_map, ax_dist, ax_alt, ax_rew):
        ax.set_facecolor('#0a0a18')
        ax.tick_params(colors='#888899', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')

    ax_map.plot(lons, lats, lw=0.5, color='#2a2a4a', zorder=1)
    ax_map.fill(circ_lon, circ_lat, alpha=0.10, color=color, zorder=0)
    ax_map.plot(circ_lon, circ_lat, color=color, lw=1.2, ls='--', alpha=0.5, zorder=1)
    for mult in [3, 5, 10]:
        rm = STATION_RADIUS_M * mult
        rl, rn = m_to_deg_lat(rm), m_to_deg_lon(rm)
        ax_map.plot(STATION_LON + rn * np.cos(theta), STATION_LAT + rl * np.sin(theta),
                    color='#1e1e33', lw=0.5, ls=':', zorder=0)
    ax_map.plot(STATION_LON, STATION_LAT, '*', color='white', ms=9, zorder=4, alpha=0.9)
    ax_map.set_xlim(lon_lo, lon_hi)
    ax_map.set_ylim(lat_lo, lat_hi)
    ax_map.set_aspect('equal', adjustable='box')
    ax_map.set_xlabel('Longitude (°)', color='#888899', fontsize=8)
    ax_map.set_ylabel('Latitude (°)', color='#888899', fontsize=8)
    ax_map.set_title('Balloon position  (★=station, dashed=50 km radius)', color='white', fontsize=9)

    ax_dist.set_xlim(0, times[-1])
    ax_dist.set_ylim(0, max(dists.max() * 1.1, STATION_RADIUS_M / 1000 * 1.5))
    ax_dist.axhline(STATION_RADIUS_M / 1000, color='#556688', lw=0.9, ls='--')
    ax_dist.fill_between([0, times[-1]], 0, STATION_RADIUS_M / 1000,
                         alpha=0.07, color='#2ecc71', zorder=0)
    ax_dist.set_xlabel('Time (h)', color='#888899', fontsize=7)
    ax_dist.set_ylabel('km', color='#888899', fontsize=7)
    ax_dist.set_title('Distance from station', color='white', fontsize=8)

    ax_alt.set_xlim(0, times[-1])
    ax_alt.set_ylim(ALT_BAND_LOW_M / 1000 - 0.2, ALT_BAND_HIGH_M / 1000 + 0.2)
    ax_alt.axhline(ALT_BAND_LOW_M  / 1000, color='#444455', lw=0.7, ls=':')
    ax_alt.axhline(ALT_BAND_HIGH_M / 1000, color='#444455', lw=0.7, ls=':')
    ax_alt.set_xlabel('Time (h)', color='#888899', fontsize=7)
    ax_alt.set_ylabel('km', color='#888899', fontsize=7)
    ax_alt.set_title('Altitude', color='white', fontsize=8)

    ax_rew.set_xlim(0, times[-1])
    r_min = min(rewards.min() - 0.05, -0.15)
    r_max = rewards.max() + 0.15
    ax_rew.set_ylim(r_min, r_max)
    ax_rew.axhline(0, color='#444455', lw=0.7, ls=':')
    ax_rew.axhline(1, color='#225533', lw=0.7, ls=':')
    ax_rew.set_xlabel('Time (h)', color='#888899', fontsize=7)
    ax_rew.set_ylabel('reward', color='#888899', fontsize=7)
    ax_rew.set_title('Step reward  (grey = 1-h rolling mean)', color='white', fontsize=8)

    trail_line,  = ax_map.plot([], [], lw=2.2, zorder=2, solid_capstyle='round')
    balloon_dot, = ax_map.plot([], [], 'o', ms=11, zorder=5,
                                markeredgecolor='white', markeredgewidth=1.0)
    time_txt = ax_map.text(0.03, 0.97, '', transform=ax_map.transAxes,
                           color='white', fontsize=8, va='top', family='monospace',
                           bbox=dict(facecolor='#12121f', alpha=0.65, pad=3, edgecolor='none'))

    dist_line, = ax_dist.plot([], [], lw=1.2, color=color)
    dist_dot,  = ax_dist.plot([], [], 'o', color='white', ms=4)

    alt_line,  = ax_alt.plot([], [], lw=1.2, color='#5dade2')
    alt_dot,   = ax_alt.plot([], [], 'o', color='white', ms=4)

    rew_step,  = ax_rew.plot([], [], lw=0.9, color=color, alpha=0.5)
    rew_roll,  = ax_rew.plot([], [], lw=1.6, color='#aaaaaa')
    rew_dot,   = ax_rew.plot([], [], 'o', color='white', ms=4)

    frames = list(range(0, n, stride))
    if frames[-1] != n - 1:
        frames.append(n - 1)

    def init():
        for a in (trail_line, balloon_dot, dist_line, dist_dot,
                  alt_line, alt_dot, rew_step, rew_roll, rew_dot):
            a.set_data([], [])
        time_txt.set_text('')
        return (trail_line, balloon_dot, dist_line, dist_dot,
                alt_line, alt_dot, rew_step, rew_roll, rew_dot, time_txt)

    def update(fi):
        i = frames[fi]
        t0 = max(0, i - tail_len)

        c_trail = '#2ecc71' if in_r[i] else '#e74c3c'
        trail_line.set_data(lons[t0:i+1], lats[t0:i+1])
        trail_line.set_color(c_trail)
        balloon_dot.set_data([lons[i]], [lats[i]])
        balloon_dot.set_color('#2ecc71' if in_r[i] else '#e74c3c')
        time_txt.set_text(
            f't = {times[i]:.1f} h  [{"IN " if in_r[i] else "OUT"}]\n'
            f'dist = {dists[i]:.0f} km\n'
            f'alt  = {alts[i]/1000:.2f} km\n'
            f'rew  = {rewards[i]:.3f}'
        )

        dist_line.set_data(times[:i+1], dists[:i+1])
        dist_dot.set_data([times[i]], [dists[i]])

        alt_line.set_data(times[:i+1], alts[:i+1] / 1000)
        alt_dot.set_data([times[i]], [alts[i] / 1000])

        rew_step.set_data(times[:i+1], rewards[:i+1])
        rew_roll.set_data(times[:i+1], reward_roll[:i+1])
        rew_dot.set_data([times[i]], [rewards[i]])

        return (trail_line, balloon_dot, dist_line, dist_dot,
                alt_line, alt_dot, rew_step, rew_roll, rew_dot, time_txt)

    ani = FuncAnimation(fig, update, frames=len(frames), init_func=init,
                        interval=1000/fps, blit=True)
    ani.save(str(out_path), writer=PillowWriter(fps=fps), dpi=110)
    plt.close(fig)
    print(f'  Saved → {out_path}  ({len(frames)} frames @ {fps}fps)')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation', required=True, choices=sorted(ENV_FLAGS))
    parser.add_argument('--weight', required=True)
    parser.add_argument('--out-dir', default='.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--duration', type=float, default=3600 * 72)
    args = parser.parse_args()

    flags = ENV_FLAGS[args.ablation]
    label = LABELS[args.ablation]
    out_dir = Path(args.out_dir)

    print(f'Loading agent from {args.weight}')
    agent = load_agent(Path(args.weight), AGENT_KWARGS.get(args.ablation))

    for preset in ['calm', 'tropical', 'strong-shear']:
        print(f'\nRunning {preset}...')
        traj = run_episode(agent, preset, args.duration, args.seed, flags)
        print(f'  TWR50={traj["twr50"]*100:.1f}%  steps={len(traj["time_h"])}')
        out = out_dir / f'replay_ablate_{args.ablation}_{preset.replace("-","_")}.gif'
        make_gif(traj, preset, out, label=label, stride=4, fps=12)

    print('\nDone.')


if __name__ == '__main__':
    main()

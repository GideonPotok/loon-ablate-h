#!/usr/bin/env python3
"""
render_era5.py — replay media for the ERA5 evaluation.

The ERA5 finding is a paired one: R and the navigator heuristic fly the *same*
wind, and one holds station while the other does not. A per-policy plot alone
loses that, so alongside the repo-standard 6-panel replays this emits a
side-by-side figure and an animation of both policies on one episode.

Note both plotters in this repo hardcode the station at (0°N, 170°E). Under
wind_source='era5' the station is wherever the archive sample landed, so
plot_episode now takes the episode's real target; passing the constant would
draw the 50 km radius circle a few thousand km from the balloon.

Episodes are chosen from the eval JSON rather than picked by hand: the one
where the heuristic most outscores R (the failure, at its clearest) and the
one nearest the heuristic's median (the typical case).

Usage:
    python scratch/render_era5.py <era5_json_dir> [--results weights/era5_eval_r_s_100ep.json]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from replay import (ABLATION_ENV_FLAGS, STATION_RADIUS_M, load_agent,  # noqa: E402
                    m_to_deg_lat, m_to_deg_lon, plot_episode, run_episode)

DURATION_S = 72 * 3600
R_COLOR, H_COLOR = '#8e44ad', '#e67e22'
FLOAT_ACTION = 8


class FloatAgent:
    def reset_hidden(self): pass
    def select_action(self, state, greedy=True): return FLOAT_ACTION


def fly(seed: str, era5_dir: str, heuristic: bool) -> dict:
    agent = (FloatAgent() if heuristic
             else load_agent(REPO / 'weights' / 'dqn_ablate_r.pt'))
    return run_episode(agent, 'tropical', DURATION_S, seed,
                       server_version='v2', flags=dict(ABLATION_ENV_FLAGS['r']),
                       wind_source='era5', era5_dir=era5_dir, heuristic=heuristic)


def compare_png(r: dict, h: dict, station: tuple[float, float], out: Path, title: str) -> None:
    st_lat, st_lon = station
    r_lat_deg = m_to_deg_lat(STATION_RADIUS_M)
    r_lon_deg = m_to_deg_lon(STATION_RADIUS_M, st_lat)
    theta = np.linspace(0, 2 * math.pi, 200)

    fig = plt.figure(figsize=(15, 9))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.30, wspace=0.22,
                          height_ratios=[1.5, 1])

    for col, (traj, name, color) in enumerate(
            [(r, 'Ablation R', R_COLOR), (h, 'navigator heuristic', H_COLOR)]):
        ax = fig.add_subplot(gs[0, col])
        lons, lats = np.array(traj['lon']), np.array(traj['lat'])
        in_r = traj['in_radius']

        ax.fill(st_lon + r_lon_deg * np.cos(theta), st_lat + r_lat_deg * np.sin(theta),
                alpha=0.14, color=color, zorder=0)
        ax.plot(st_lon + r_lon_deg * np.cos(theta), st_lat + r_lat_deg * np.sin(theta),
                color=color, lw=1.2, ls='--', zorder=1)
        ax.plot(st_lon, st_lat, '*', color=color, ms=14, zorder=4)

        pts = np.array([lons, lats]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        ax.add_collection(LineCollection(
            segs, colors=['#2ecc71' if i else '#e74c3c' for i in in_r[1:]],
            linewidths=1.2, zorder=2))
        ax.plot(lons[0], lats[0], 'o', color='#2c3e50', ms=6, zorder=5, label='start')
        ax.plot(lons[-1], lats[-1], 's', color='#2c3e50', ms=6, zorder=5, label='end')

        # Shared extent so the two panels are visually comparable. The circle's
        # own bounds go in: without them the radius gets clipped by the axis and
        # the station star reads as sitting off-centre inside its own circle.
        all_lon = np.concatenate([r['lon'], h['lon'],
                                  [st_lon - r_lon_deg, st_lon + r_lon_deg]])
        all_lat = np.concatenate([r['lat'], h['lat'],
                                  [st_lat - r_lat_deg, st_lat + r_lat_deg]])
        pad_x = max(np.ptp(all_lon), r_lon_deg * 3) * 0.08
        pad_y = max(np.ptp(all_lat), r_lat_deg * 3) * 0.08
        ax.set_xlim(all_lon.min() - pad_x, all_lon.max() + pad_x)
        ax.set_ylim(all_lat.min() - pad_y, all_lat.max() + pad_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)', fontsize=9)
        ax.set_title(f'{name} — TWR50 {traj["twr50"]*100:.1f}%', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

    ax_d = fig.add_subplot(gs[1, 0])
    for traj, name, color in [(r, 'R', R_COLOR), (h, 'heuristic', H_COLOR)]:
        ax_d.plot(traj['time_s'], traj['dist_m'], lw=1.0, color=color, label=name)
    ax_d.axhline(STATION_RADIUS_M / 1000, color='gray', lw=1.0, ls='--', label='50 km radius')
    ax_d.fill_between([0, max(r['time_s'])], 0, STATION_RADIUS_M / 1000,
                      alpha=0.10, color='#2ecc71')
    ax_d.set_yscale('log')
    ax_d.set_xlabel('Time (h)', fontsize=9)
    ax_d.set_ylabel('Distance (km, log)', fontsize=9)
    ax_d.set_title('Distance from station', fontsize=10)
    ax_d.legend(fontsize=8)

    ax_a = fig.add_subplot(gs[1, 1])
    for traj, name, color in [(r, 'R', R_COLOR), (h, 'heuristic', H_COLOR)]:
        ax_a.plot(traj['time_s'], np.array(traj['alt_m']) / 1000, lw=0.9,
                  color=color, label=name)
    ax_a.set_xlabel('Time (h)', fontsize=9)
    ax_a.set_ylabel('Altitude (km)', fontsize=9)
    ax_a.set_title('Altitude commanded — the only control either policy has', fontsize=10)
    ax_a.legend(fontsize=8)

    fig.savefig(str(out), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def compare_gif(r: dict, h: dict, station: tuple[float, float], out: Path,
                title: str, stride: int = 6, fps: int = 12) -> None:
    st_lat, st_lon = station
    r_lat_deg = m_to_deg_lat(STATION_RADIUS_M)
    r_lon_deg = m_to_deg_lon(STATION_RADIUS_M, st_lat)
    theta = np.linspace(0, 2 * math.pi, 200)
    frames = list(range(0, len(r['lat']), stride))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    all_lon = np.concatenate([r['lon'], h['lon'],
                              [st_lon - r_lon_deg, st_lon + r_lon_deg]])
    all_lat = np.concatenate([r['lat'], h['lat'],
                              [st_lat - r_lat_deg, st_lat + r_lat_deg]])
    pad_x, pad_y = np.ptp(all_lon) * 0.08 + 0.3, np.ptp(all_lat) * 0.08 + 0.3

    artists = []
    for ax, (traj, name, color) in zip(
            axes, [(r, 'Ablation R', R_COLOR), (h, 'navigator heuristic', H_COLOR)]):
        ax.fill(st_lon + r_lon_deg * np.cos(theta), st_lat + r_lat_deg * np.sin(theta),
                alpha=0.14, color=color, zorder=0)
        ax.plot(st_lon + r_lon_deg * np.cos(theta), st_lat + r_lat_deg * np.sin(theta),
                color=color, lw=1.2, ls='--', zorder=1)
        ax.plot(st_lon, st_lat, '*', color=color, ms=14, zorder=4)
        ax.set_xlim(all_lon.min() - pad_x, all_lon.max() + pad_x)
        ax.set_ylim(all_lat.min() - pad_y, all_lat.max() + pad_y)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)', fontsize=9)
        trail, = ax.plot([], [], lw=1.3, color=color, zorder=2)
        dot,   = ax.plot([], [], 'o', color='#2c3e50', ms=7, zorder=5)
        txt = ax.set_title(f'{name}', fontsize=10)
        artists.append((traj, trail, dot, ax, name))

    def update(fi):
        out_artists = []
        for traj, trail, dot, ax, name in artists:
            trail.set_data(traj['lon'][:fi + 1], traj['lat'][:fi + 1])
            dot.set_data([traj['lon'][fi]], [traj['lat'][fi]])
            twr = sum(traj['in_radius'][:fi + 1]) / (fi + 1)
            ax.set_title(f'{name} — t={traj["time_s"][fi]:.0f} h, '
                         f'{traj["dist_m"][fi]:.0f} km out, TWR50 {twr*100:.0f}%',
                         fontsize=10)
            out_artists += [trail, dot]
        return out_artists

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  wrote {out.name} ({len(frames)} frames)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('era5_dir')
    ap.add_argument('--results', default=str(REPO / 'weights' / 'era5_eval_r_s_100ep.json'))
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())
    seeds = res['seeds']
    r_scores = res['policies']['R']['twr50']
    h_scores = res['policies']['heuristic']['twr50']

    gaps = [h - rr for h, rr in zip(h_scores, r_scores)]
    showcase = seeds[gaps.index(max(gaps))]
    med = statistics.median(h_scores)
    typical = seeds[min(range(len(h_scores)), key=lambda i: abs(h_scores[i] - med))]

    picks = [('gap', showcase, 'largest heuristic-over-R gap'),
             ('typical', typical, 'median heuristic episode')]

    for tag, seed, why in picks:
        print(f'\n{tag}: seed {seed} ({why})')
        r = fly(seed, args.era5_dir, heuristic=False)
        h = fly(seed, args.era5_dir, heuristic=True)
        cell = r['wind']
        station = (cell['lat'], cell['lon360'] - 360 if cell['lon360'] > 180 else cell['lon360'])
        head = (f'ERA5 {cell["startISO"][:10]} @ {cell["lat"]}°N, {cell["lon360"]}°E  '
                f'— same wind, both policies  |  R {r["twr50"]*100:.1f}%  '
                f'vs heuristic {h["twr50"]*100:.1f}%')
        print(f'  R {r["twr50"]*100:.1f}%   heuristic {h["twr50"]*100:.1f}%')

        # Repo-standard 6-panel replays, with the station where it actually is.
        plot_episode(r, 'tropical', REPO / f'replay_era5_r_{tag}.png',
                     label=f'Ablation R on ERA5 (seed {seed})',
                     station_lat=station[0], station_lon=station[1])
        plot_episode(h, 'tropical', REPO / f'replay_era5_heuristic_{tag}.png',
                     label=f'Navigator heuristic on ERA5 (seed {seed})',
                     station_lat=station[0], station_lon=station[1])
        print(f'  wrote replay_era5_r_{tag}.png, replay_era5_heuristic_{tag}.png')

        compare_png(r, h, station, REPO / f'replay_era5_compare_{tag}.png', head)
        compare_gif(r, h, station, REPO / f'replay_era5_compare_{tag}.gif', head)

    print('\nRENDER_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())

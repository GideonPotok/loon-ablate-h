#!/usr/bin/env python3
"""
render_era5_gassand.py — replay media for the gassand ERA5 evaluation.

The gassand twin of scratch/render_era5.py. The finding is a paired one: two
policies fly the *same* ERA5 wind and get different results, which no
single-policy plot can show, so alongside the repo-standard 6-panel replays this
emits a side-by-side figure and an animation of both policies on one episode.

Two gassand-specific differences from the v2 renderer:

  * A third row for reserve depletion. Helium and sand are the point of this
    env, and "who held station" is only half the story when one policy got
    there by dumping all 20 kg of ballast in the first six hours.
  * replay_gassand.plot_episode hardcoded the station at (0°N, 170°E), same as
    replay.py did before the v2 port. Under wind_source='era5' the station is
    the sampled grid cell, so it now takes the episode's real target; passing
    the constant drew the 50 km radius circle a few thousand km from the balloon.

Episodes come from the eval JSON rather than being picked by hand: the one where
the comparison policy most outscores the subject (the failure at its clearest)
and the one nearest the comparison's median (the typical case).

Usage:
    python scratch/render_era5_gassand.py "/Volumes/Gideon SDD/data/era5_json" \
        [--results weights/era5_eval_gassand_100ep.json] [--pair r_gassand float]
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

from replay_gassand import (HELIUM_CAP_KG, SAND_CAP_KG, STATION_RADIUS_M,  # noqa: E402
                            HE_COLOR, SAND_COLOR, load_agent, m_to_deg_lat,
                            m_to_deg_lon, plot_episode, run_episode)
from scratch.eval_era5_gassand import (FLOAT_ACTION, POLICY_WEIGHTS,  # noqa: E402
                                       REALISM_FLAGS, FloatAgent)

DURATION_S = 72 * 3600
PALETTE = {
    'r_gassand':   '#8e44ad',
    'res_gassand': '#16a085',
    'gassand':     '#2980b9',
    'heuristic':   '#e67e22',
    'float':       '#7f8c8d',
}
NICE = {
    'r_gassand':   'R_Gassand',
    'res_gassand': 'Res_Gassand',
    'gassand':     'Gassand',
    'heuristic':   'navigator heuristic',
    'float':       'FLOAT (do nothing)',
}


def fly(policy: str, seed: int, era5_dir: str) -> dict:
    """Roll one episode of `policy` on the ERA5 episode that `seed` selects."""
    w = POLICY_WEIGHTS[policy]
    if w == 'FLOAT':
        agent = FloatAgent()
    elif w is None:
        agent = None                       # built-in JS navigator heuristic
    else:
        agent = load_agent(REPO / w)
    return run_episode('tropical', DURATION_S, seed, agent=agent,
                       flags=REALISM_FLAGS, wind_source='era5', era5_dir=era5_dir)


def _extent(trajs, st_lat, st_lon, r_lat_deg, r_lon_deg):
    """
    Shared axis extent for the paired maps. The circle's own bounds go in:
    without them the radius gets clipped by the axis and the station star reads
    as sitting off-centre inside its own circle.
    """
    all_lon = np.concatenate([t['lon'] for t in trajs]
                             + [[st_lon - r_lon_deg, st_lon + r_lon_deg]])
    all_lat = np.concatenate([t['lat'] for t in trajs]
                             + [[st_lat - r_lat_deg, st_lat + r_lat_deg]])
    pad_x = max(np.ptp(all_lon), r_lon_deg * 3) * 0.08
    pad_y = max(np.ptp(all_lat), r_lat_deg * 3) * 0.08
    return (all_lon.min() - pad_x, all_lon.max() + pad_x,
            all_lat.min() - pad_y, all_lat.max() + pad_y)


def _draw_track(ax, traj, name, st_lat, st_lon, r_lat_deg, r_lon_deg, theta):
    """Station, 50 km radius and the in/out-coloured track on one map axes."""
    color = PALETTE[name]
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
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Longitude (°)', fontsize=9)
    ax.set_ylabel('Latitude (°)', fontsize=9)


def compare_png(a: dict, b: dict, names: tuple[str, str],
                station: tuple[float, float], out: Path, title: str) -> None:
    st_lat, st_lon = station
    r_lat_deg = m_to_deg_lat(STATION_RADIUS_M)
    r_lon_deg = m_to_deg_lon(STATION_RADIUS_M, st_lat)
    theta = np.linspace(0, 2 * math.pi, 200)
    pair = [(a, names[0]), (b, names[1])]

    # Four rows, because the two spatial scales cannot share one panel: the
    # balloons drift thousands of km while the thing they are supposed to hold
    # is 50 km across. At equal aspect on the drift extent the station circle is
    # a dot, so the drift map and the near-station map get a row each.
    fig = plt.figure(figsize=(15, 17))
    fig.suptitle(title, fontsize=13, fontweight='bold')
    gs = fig.add_gridspec(4, 2, hspace=0.30, wspace=0.22,
                          height_ratios=[0.62, 1.55, 0.95, 0.95])
    x0, x1, y0, y1 = _extent([a, b], st_lat, st_lon, r_lat_deg, r_lon_deg)

    for col, (traj, name) in enumerate(pair):
        ax = fig.add_subplot(gs[0, col])
        _draw_track(ax, traj, name, st_lat, st_lon, r_lat_deg, r_lon_deg, theta)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_title(f'{NICE[name]} — TWR50 {traj["twr50"]*100:.1f}%  '
                     f'(full drift, {traj["dist_km"][-1]:.0f} km out at 72 h)', fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

    # Near-station zoom: ±3 radii, where the 50 km circle is actually readable
    # and you can see whether the policy ever held anything at all.
    zi = 3.0
    for col, (traj, name) in enumerate(pair):
        ax = fig.add_subplot(gs[1, col])
        _draw_track(ax, traj, name, st_lat, st_lon, r_lat_deg, r_lon_deg, theta)
        ax.set_xlim(st_lon - r_lon_deg * zi, st_lon + r_lon_deg * zi)
        ax.set_ylim(st_lat - r_lat_deg * zi, st_lat + r_lat_deg * zi)
        hrs = float(np.sum(traj['in_radius'])) * 300 / 3600
        ax.set_title(f'{NICE[name]} near the station (±150 km)\n'
                     f'{hrs:.1f} h of 72 h inside the 50 km radius', fontsize=10)

    ax_d = fig.add_subplot(gs[2, 0])
    for traj, name in pair:
        ax_d.plot(traj['time_h'], traj['dist_km'], lw=1.0,
                  color=PALETTE[name], label=NICE[name])
    ax_d.axhline(STATION_RADIUS_M / 1000, color='gray', lw=1.0, ls='--',
                 label='50 km radius')
    ax_d.fill_between([0, max(a['time_h'])], 0, STATION_RADIUS_M / 1000,
                      alpha=0.10, color='#2ecc71')
    ax_d.set_yscale('log')
    ax_d.set_xlabel('Time (h)', fontsize=9)
    ax_d.set_ylabel('Distance (km, log)', fontsize=9)
    ax_d.set_title('Distance from station', fontsize=10)
    ax_d.legend(fontsize=8)

    ax_alt = fig.add_subplot(gs[2, 1])
    for traj, name in pair:
        ax_alt.plot(traj['time_h'], np.array(traj['alt_m']) / 1000, lw=0.9,
                    color=PALETTE[name], label=NICE[name])
    ax_alt.set_xlabel('Time (h)', fontsize=9)
    ax_alt.set_ylabel('Altitude (km)', fontsize=9)
    ax_alt.set_title('Altitude flown — the only control either policy has', fontsize=10)
    ax_alt.legend(fontsize=8)

    # Reserve depletion: the gassand-specific half of the story.
    for col, (traj, name) in enumerate(pair):
        ax_r = fig.add_subplot(gs[3, col])
        he_pct   = 100 * np.array(traj['helium_kg']) / HELIUM_CAP_KG
        sand_pct = 100 * np.array(traj['sand_kg']) / SAND_CAP_KG
        ax_r.plot(traj['time_h'], he_pct, color=HE_COLOR, lw=1.8, label='helium (lift)')
        ax_r.plot(traj['time_h'], sand_pct, color=SAND_COLOR, lw=1.8, label='sand (ballast)')
        ax_r.fill_between(traj['time_h'], 0, he_pct,   color=HE_COLOR,   alpha=0.10)
        ax_r.fill_between(traj['time_h'], 0, sand_pct, color=SAND_COLOR, alpha=0.10)
        ax_r.set_ylim(0, 105)
        ax_r.set_xlabel('Time (h)', fontsize=9)
        ax_r.set_ylabel('Reserve remaining (%)', fontsize=9)
        ax_r.set_title(f'{NICE[name]} reserves — He {traj["he_left"]:.2f} kg, '
                       f'sand {traj["sand_left"]:.2f} kg left', fontsize=10)
        ax_r.legend(fontsize=7, loc='lower left')

    fig.savefig(str(out), dpi=140, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {out.name}')


def compare_gif(a: dict, b: dict, names: tuple[str, str],
                station: tuple[float, float], out: Path, title: str,
                stride: int = 6, fps: int = 12) -> None:
    st_lat, st_lon = station
    r_lat_deg = m_to_deg_lat(STATION_RADIUS_M)
    r_lon_deg = m_to_deg_lon(STATION_RADIUS_M, st_lat)
    theta = np.linspace(0, 2 * math.pi, 200)
    frames = list(range(0, len(a['lat']), stride))
    x0, x1, y0, y1 = _extent([a, b], st_lat, st_lon, r_lat_deg, r_lon_deg)

    # Size the canvas from the data aspect. The drift extent is typically several
    # times wider than it is tall, and equal-aspect map panels in a fixed square
    # figure letterbox into a thin strip surrounded by whitespace.
    panel_w_in = 6.0
    panel_h_in = min(6.0, max(2.0, panel_w_in * (y1 - y0) / (x1 - x0)))
    fig = plt.figure(figsize=(13, panel_h_in + 4.0))
    fig.suptitle(title, fontsize=12, fontweight='bold')
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.20,
                          height_ratios=[panel_h_in, 2.1])

    # The 50 km radius is invisible at drift scale — it genuinely is a dot next
    # to 2,000 km of travel — so the shared distance track below carries the
    # in/out question instead, with the radius as a line you can actually see.
    ax_d = fig.add_subplot(gs[1, :])
    ax_d.set_xlim(0, a['time_h'][-1])
    ax_d.set_yscale('log')
    ax_d.set_ylim(max(1.0, min(np.min(a['dist_km']), np.min(b['dist_km'])) * 0.7),
                  max(np.max(a['dist_km']), np.max(b['dist_km'])) * 1.4)
    ax_d.axhline(STATION_RADIUS_M / 1000, color='gray', lw=1.1, ls='--')
    ax_d.text(a['time_h'][-1] * 0.55, STATION_RADIUS_M / 1000, '50 km radius',
              color='gray', fontsize=8, va='bottom', ha='left')
    ax_d.set_xlabel('Time (h)', fontsize=9)
    ax_d.set_ylabel('Distance from station (km, log)', fontsize=9)

    artists = []
    for col, (traj, name) in enumerate([(a, names[0]), (b, names[1])]):
        ax = fig.add_subplot(gs[0, col])
        color = PALETTE[name]
        ax.fill(st_lon + r_lon_deg * np.cos(theta), st_lat + r_lat_deg * np.sin(theta),
                alpha=0.14, color=color, zorder=0)
        ax.plot(st_lon + r_lon_deg * np.cos(theta), st_lat + r_lat_deg * np.sin(theta),
                color=color, lw=1.2, ls='--', zorder=1)
        ax.plot(st_lon, st_lat, '*', color=color, ms=14, zorder=4)
        ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Longitude (°)', fontsize=9)
        ax.set_ylabel('Latitude (°)', fontsize=9)
        trail, = ax.plot([], [], lw=1.3, color=color, zorder=2)
        dot,   = ax.plot([], [], 'o', color='#2c3e50', ms=7, zorder=5)
        # Top-left, so it cannot sit on the station star at bottom-centre.
        txt = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=7.5,
                      family='monospace', va='top',
                      bbox=dict(facecolor='white', alpha=0.75, pad=2, edgecolor='none'))
        dline, = ax_d.plot([], [], lw=1.4, color=color, label=NICE[name])
        ddot,  = ax_d.plot([], [], 'o', color=color, ms=5)
        artists.append((traj, trail, dot, ax, name, txt, dline, ddot))
    ax_d.legend(fontsize=8, loc='lower right')

    def update(fi):
        out_artists = []
        for traj, trail, dot, ax, name, txt, dline, ddot in artists:
            trail.set_data(traj['lon'][:fi + 1], traj['lat'][:fi + 1])
            dot.set_data([traj['lon'][fi]], [traj['lat'][fi]])
            twr = float(np.mean(traj['in_radius'][:fi + 1]))
            ax.set_title(f'{NICE[name]} — t={traj["time_h"][fi]:.0f} h, '
                         f'{traj["dist_km"][fi]:.0f} km out, TWR50 {twr*100:.0f}%',
                         fontsize=10)
            txt.set_text(f'alt {traj["alt_m"][fi]/1000:5.2f} km\n'
                         f'He  {traj["helium_kg"][fi]:5.2f} kg\n'
                         f'sand{traj["sand_kg"][fi]:6.2f} kg')
            dline.set_data(traj['time_h'][:fi + 1], traj['dist_km'][:fi + 1])
            ddot.set_data([traj['time_h'][fi]], [traj['dist_km'][fi]])
            out_artists += [trail, dot, txt, dline, ddot]
        return out_artists

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    anim.save(str(out), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'  wrote {out.name} ({len(frames)} frames)')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('era5_dir')
    ap.add_argument('--results',
                    default=str(REPO / 'weights' / 'era5_eval_gassand_100ep.json'))
    ap.add_argument('--pair', nargs=2, default=['r_gassand', 'heuristic'],
                    choices=list(POLICY_WEIGHTS),
                    help='subject then comparison (episodes are picked by the '
                         'comparison policy beating the subject)')
    args = ap.parse_args()

    subj, comp = args.pair
    res = json.loads(Path(args.results).read_text())
    seeds = res['seeds']
    s_scores = res['policies'][subj]['twr50']
    c_scores = res['policies'][comp]['twr50']

    gaps = [c - s for c, s in zip(c_scores, s_scores)]
    showcase = seeds[gaps.index(max(gaps))]
    med = statistics.median(c_scores)
    typical = seeds[min(range(len(c_scores)), key=lambda i: abs(c_scores[i] - med))]

    picks = [('gap', showcase, f'largest {comp}-over-{subj} gap'),
             ('typical', typical, f'median {comp} episode')]
    # Namespace by pair: the two pairs pick different episodes, so unsuffixed
    # names would silently overwrite each other's figures.
    slug = f'{subj}_vs_{comp}'

    for tag, seed, why in picks:
        print(f'\n{tag}: seed {seed} ({why})')
        ta = fly(subj, seed, args.era5_dir)
        tb = fly(comp, seed, args.era5_dir)
        cell = ta['wind']
        st_lat = ta['station_lat']
        st_lon = ta['station_lon']
        assert (tb['station_lat'], tb['station_lon']) == (st_lat, st_lon), \
            'paired episodes disagree on the station — pairing is broken'
        head = (f'ERA5 {cell["startISO"][:10]} @ {cell["lat"]}°N, {cell["lon360"]}°E'
                f'  — same wind, both policies  |  {NICE[subj]} {ta["twr50"]*100:.1f}%'
                f'  vs {NICE[comp]} {tb["twr50"]*100:.1f}%')
        print(f'  {subj} {ta["twr50"]*100:.1f}%   {comp} {tb["twr50"]*100:.1f}%')

        # Repo-standard 6-panel replays, with the station where it actually is.
        for traj, nm in ((ta, subj), (tb, comp)):
            plot_episode(traj, 'tropical',
                         REPO / f'replay_era5_gassand_{slug}_{nm}_{tag}.png',
                         label=f'{NICE[nm]} on ERA5 (seed {seed})',
                         station_lat=st_lat, station_lon=st_lon)

        compare_png(ta, tb, (subj, comp), (st_lat, st_lon),
                    REPO / f'replay_era5_gassand_{slug}_{tag}.png', head)
        compare_gif(ta, tb, (subj, comp), (st_lat, st_lon),
                    REPO / f'replay_era5_gassand_{slug}_{tag}.gif', head)

    print('\nRENDER_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())

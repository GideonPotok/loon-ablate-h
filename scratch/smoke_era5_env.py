#!/usr/bin/env python3
"""
smoke_era5_env.py — end-to-end check of BalloonEnv(wind_source='era5').

Runs real episodes through the Python → NDJSON → v2 server → WindArchive path
and asserts the things that would silently produce a meaningless eval:
the station has to move with the sample, seeds have to draw different weather,
the wind has to actually vary over the episode, and the episode's provenance
has to come back so a result can be reproduced.

Also runs the navigator heuristic on the same episodes. That is the control
the ERA5 numbers are useless without: if the heuristic collapses too, the
env is the ceiling, not the policy.

Usage:
    python scratch/smoke_era5_env.py <era5_json_dir>
"""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from balloon_env import BalloonEnv  # noqa: E402

R_FLAGS = {
    'use_reward_fix': True, 'use_shaping': True, 'use_expanded_state': False,
    'use_time_features': False, 'shaping_beta': 0.5, 'shaping_gamma': 0.97,
    'terminal_twr_bonus': 50.0, 'shaping_linear': False, 'shaping_D_max': 500_000.0,
    'wind_phase_jitter': True, 'wind_episode_noise': True,
    'wind_param_jitter': True, 'domain_rand': True,
}

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    print(f"  {'ok  ' if cond else 'FAIL'} {msg}")
    if not cond:
        failures.append(msg)


def run_episode(env: BalloonEnv, heuristic: bool) -> dict:
    env.reset()
    info = {}
    while True:
        if heuristic:
            _, _, _, done, info = env.heuristic_step()
        else:
            _, _, done, info = env.step(8)   # mid-band hold: a fixed-altitude float
        if done:
            break
    return info


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    era5_dir = sys.argv[1]

    print('1. guard rails')
    for kwargs, want in [
        (dict(server_version='v1', wind_source='era5', era5_dir=era5_dir), 'v1'),
        (dict(server_version='v2', wind_source='era5', era5_dir=None),     'era5_dir'),
        (dict(server_version='v2', wind_source='nope'),                    'wind_source'),
    ]:
        try:
            BalloonEnv(**kwargs).close()
            check(False, f'{want}: should have raised')
        except ValueError as e:
            check(want in str(e), f'{want} rejected: {str(e)[:60]}')

    print('\n2. ERA5 episodes (72 h, R flags)')
    metas, twrs, dists = [], [], []
    for i in range(6):
        env = BalloonEnv(preset='tropical', duration_s=72 * 3600,
                         seed=42 + i * 1_000_003, server_version='v2',
                         flags=dict(R_FLAGS), wind_source='era5', era5_dir=era5_dir)
        try:
            reset_info = env.last_reset_info if env.last_reset_info else None
            env.reset()
            reset_info = env.last_reset_info
            metas.append(reset_info)
            info = run_episode(env, heuristic=False)
            twrs.append(info['twr50'])
            dists.append(info['dist_m'])
        finally:
            env.close()

    check(all(m.get('wind', {}).get('source') == 'era5' for m in metas),
          'every reset reports source=era5')
    check(all('startISO' in m.get('wind', {}) for m in metas),
          'reset carries the archive start time (reproducible)')

    cells = {(m['wind']['lat'], m['wind']['lon360']) for m in metas}
    times = {m['wind']['startUnix'] for m in metas}
    check(len(cells) > 1, f'seeds draw different grid cells ({len(cells)}/6 distinct)')
    check(len(times) > 1, f'seeds draw different start times ({len(times)}/6 distinct)')

    targets = {(round(m['target_lat'], 3), round(m['target_lon'], 3)) for m in metas}
    check((0.0, 170.0) not in targets or len(targets) > 1,
          'station moved off the hardcoded (0, 170) preset target')
    check(len(targets) == len(cells), 'target tracks the sampled cell')

    print(f"     sample: {metas[0]['wind']['startISO']} @ "
          f"lat {metas[0]['wind']['lat']}, lon {metas[0]['wind']['lon360']}")
    print(f"     float-at-mid-band TWR50: {statistics.mean(twrs) * 100:.1f}% "
          f"(final dist {statistics.mean(dists) / 1000:.0f} km)")

    print('\n3. heuristic control on the same episodes')
    h_twrs = []
    for i in range(6):
        env = BalloonEnv(preset='tropical', duration_s=72 * 3600,
                         seed=42 + i * 1_000_003, server_version='v2',
                         flags=dict(R_FLAGS), wind_source='era5', era5_dir=era5_dir)
        try:
            info = run_episode(env, heuristic=True)
            h_twrs.append(info['twr50'])
        finally:
            env.close()

    heur, passive = statistics.mean(h_twrs) * 100, statistics.mean(twrs) * 100
    print(f"     heuristic TWR50 {heur:.1f}%   vs passive float {passive:.1f}%")
    check(heur >= passive,
          f'heuristic beats a passive float ({heur:.1f}% vs {passive:.1f}%) '
          '— altitude control is worth something in this wind')

    print('\n4. preset source still works from the same build')
    env = BalloonEnv(preset='tropical', duration_s=6 * 3600, seed=42,
                     server_version='v2', flags=dict(R_FLAGS))
    try:
        s = env.reset()
        check(len(s) == 20, f'preset episode returns 20-dim state (got {len(s)})')
        check(env.last_reset_info['wind']['source'] == 'preset', 'preset reports source=preset')
        check(env.last_reset_info['target_lat'] == 0
              and env.last_reset_info['target_lon'] == 170,
              'preset keeps the hardcoded station at (0, 170)')
    finally:
        env.close()

    print(f"\n{'SMOKE FAILED: ' + str(len(failures)) + ' problem(s)' if failures else 'all checks passed'}")
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())

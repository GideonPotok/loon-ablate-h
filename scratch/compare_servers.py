#!/usr/bin/env python3
"""
compare_servers.py — prove two env-server builds produce identical episodes.

Adding an ERA5 wind source touches the code path every existing ablation runs
through, so the preset path has to be shown byte-identical rather than assumed
to be. This drives two server binaries over the same reset/step script and
diffs every state vector and reward.

Usage:
    # Regression: current v2 vs the version on main
    git show main:servers/balloon_env_server_v2.mjs > /tmp/old_v2.mjs
    python scratch/compare_servers.py /tmp/old_v2.mjs servers/balloon_env_server_v2.mjs

Exit code is 0 only if every response matches.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# R's exact realism bundle, plus a plain-v1-shaped case and an expanded-state
# case, so the comparison covers the flag combinations that actually get run.
FLAG_SETS = {
    'bare': {},
    'ablation_R': {
        'use_reward_fix': True, 'use_shaping': True, 'use_expanded_state': False,
        'use_time_features': False, 'shaping_beta': 0.5, 'shaping_gamma': 0.97,
        'terminal_twr_bonus': 50.0, 'shaping_linear': False, 'shaping_D_max': 500_000.0,
        'wind_phase_jitter': True, 'wind_episode_noise': True,
        'wind_param_jitter': True, 'domain_rand': True,
    },
    'ablation_S': {
        'use_reward_fix': True, 'use_shaping': True, 'use_expanded_state': False,
        'use_estimated_phase_features': True, 'shaping_beta': 0.5, 'shaping_gamma': 0.97,
        'terminal_twr_bonus': 50.0, 'shaping_linear': False, 'shaping_D_max': 500_000.0,
        'wind_phase_jitter': True, 'wind_episode_noise': True,
        'wind_param_jitter': True, 'domain_rand': True,
    },
    'expanded_state': {
        'use_reward_fix': True, 'use_shaping': True, 'use_expanded_state': True,
        'use_time_features': True, 'shaping_beta': 0.5, 'shaping_gamma': 0.97,
        'terminal_twr_bonus': 50.0, 'shaping_linear': True, 'shaping_D_max': 500_000.0,
    },
}

PRESETS = ['tropical', 'strong-shear', 'calm']
SEEDS = [42, 1_000_045, 7]
N_STEPS = 40

# Keys added by the new build; absent on the old one, so excluded from the diff.
NEW_INFO_KEYS = {'wind', 'target_lat', 'target_lon'}


class Server:
    def __init__(self, path: Path):
        self.path = path
        self.proc = subprocess.Popen(
            ['node', str(path)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            bufsize=0, cwd=str(REPO / 'servers'),
        )

    def send(self, obj: dict) -> dict:
        self.proc.stdin.write((json.dumps(obj, separators=(',', ':')) + '\n').encode())
        self.proc.stdin.flush()
        raw = self.proc.stdout.readline()
        if not raw:
            raise RuntimeError(f'{self.path} closed stdout')
        return json.loads(raw.decode())

    def close(self):
        try:
            self.send({'cmd': 'close'})
        except Exception:
            pass
        try:
            self.proc.kill()
        except Exception:
            pass


def script() -> list[dict]:
    """The identical command sequence both servers get driven through."""
    out = []
    for name, flags in FLAG_SETS.items():
        for preset in PRESETS:
            for seed in SEEDS:
                out.append({'cmd': 'reset', 'preset': preset,
                            'duration_s': 6 * 3600, 'seed': seed, **flags})
                # Deterministic action pattern — covers up, down and hold.
                for i in range(N_STEPS):
                    out.append({'cmd': 'step', 'action': (i * 5 + 3) % 17})
    return out


def compare(a: dict, b: dict, ctx: str, diffs: list[str]) -> None:
    if a.get('ok') != b.get('ok'):
        diffs.append(f'{ctx}: ok differs {a.get("ok")} vs {b.get("ok")}')
        return
    for key in ('reward', 'done', 'action'):
        if key in a or key in b:
            if a.get(key) != b.get(key):
                diffs.append(f'{ctx}: {key} {a.get(key)!r} != {b.get(key)!r}')
    sa, sb = a.get('state'), b.get('state')
    if (sa is None) != (sb is None):
        diffs.append(f'{ctx}: one response has no state')
    elif sa is not None:
        if len(sa) != len(sb):
            diffs.append(f'{ctx}: state dim {len(sa)} != {len(sb)}')
        else:
            for i, (x, y) in enumerate(zip(sa, sb)):
                if x != y:
                    diffs.append(f'{ctx}: state[{i}] {x!r} != {y!r}')
                    break
    ia, ib = a.get('info') or {}, b.get('info') or {}
    for k in (set(ia) | set(ib)) - NEW_INFO_KEYS:
        if ia.get(k) != ib.get(k):
            diffs.append(f'{ctx}: info[{k}] {ia.get(k)!r} != {ib.get(k)!r}')


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    # Absolute: the servers run with cwd=servers/, so a relative argv path
    # would be resolved against the wrong directory.
    old = Server(Path(sys.argv[1]).resolve())
    new = Server(Path(sys.argv[2]).resolve())
    cmds = script()
    diffs: list[str] = []
    ctx = ''
    try:
        for n, cmd in enumerate(cmds):
            if cmd['cmd'] == 'reset':
                ctx = f"preset={cmd['preset']} seed={cmd['seed']}"
            ra, rb = old.send(cmd), new.send(cmd)
            compare(ra, rb, f'[{n}] {ctx} {cmd["cmd"]}', diffs)
            if len(diffs) > 20:
                break
    finally:
        old.close()
        new.close()

    print(f'compared {len(cmds)} commands '
          f'({len(FLAG_SETS)} flag sets × {len(PRESETS)} presets × {len(SEEDS)} seeds)')
    if diffs:
        print(f'\n{len(diffs)} DIFFERENCE(S):')
        for d in diffs[:20]:
            print(f'  {d}')
        return 1
    print('identical — preset path unchanged')
    return 0


if __name__ == '__main__':
    sys.exit(main())

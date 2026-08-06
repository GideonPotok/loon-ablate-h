#!/usr/bin/env python3
"""
compare_servers_gassand.py — prove two gassand env-server builds produce
identical episodes.

The gassand twin of compare_servers.py. Adding an ERA5 wind source rewrites the
lines every existing gassand ablation runs through (the wind functions, and the
station constants that spawn/state/reward all read), so the preset path has to be
shown identical rather than assumed to be. Every gassand checkpoint on main —
dqn_gassand_w00, dqn_r_gassand_w00, dqn_res_gassand_w00 — was trained against
the current behaviour, so a silent change invalidates all three.

Differences from the v2 harness: gassand has 11 actions (not 17), a 21-dim state,
and its own flag vocabulary (the 4 realism flags plus the resource-aware reward
block), so the flag sets and the action pattern differ. It also returns the
resource fields (helium_kg, sand_kg, helium_vented_kg, sand_dropped_kg) in info,
which the info diff picks up for free.

The old build has to be written into servers/ , not /tmp: it imports '../js/…',
and ESM resolves that against the module's own path rather than the cwd.

Usage:
    git show main:servers/balloon_env_server_gassand.mjs > servers/_old_gassand_tmp.mjs
    python scratch/compare_servers_gassand.py servers/_old_gassand_tmp.mjs \
        servers/balloon_env_server_gassand.mjs
    rm servers/_old_gassand_tmp.mjs

Exit code is 0 only if every response matches.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

REALISM_FLAGS = {
    'wind_phase_jitter':  True,
    'wind_episode_noise': True,
    'wind_param_jitter':  True,
    'domain_rand':        True,
}

# Res_Gassand's calibrated coefficients (ablate_res_gassand_train.py).
RESOURCE_FLAGS = {
    'use_resource_reward':    True,
    'sand_cost_per_kg':       2.0,
    'helium_cost_per_kg':     25.0,
    'terminal_reserve_bonus': 25.0,
    'depletion_penalty':      25.0,
    'floor_penalty':          0.1,
}

# The flag combinations that actually get run: the plain demonstrator, the
# R_Gassand bundle, the Res_Gassand bundle, both together, and a non-default
# spawn offset (the recovery-spawn curriculum uses these).
FLAG_SETS = {
    'bare':               {},
    'realism':            REALISM_FLAGS,
    'resource_reward':    RESOURCE_FLAGS,
    'resource_realism':   {**REALISM_FLAGS, **RESOURCE_FLAGS},
    'recovery_spawn':     {**REALISM_FLAGS, 'spawn_offset_km': 220.0},
}

PRESETS = ['tropical', 'strong-shear', 'calm']
SEEDS = [42, 1_000_045, 7]
N_STEPS = 40
N_ACTIONS = 11          # gassand release ladder: 5 sand · float · 5 helium

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
                # Deterministic action pattern. Stride 3 against an 11-action
                # ladder walks the whole table (gcd(3,11)=1), so sand drops,
                # float and helium vents all get exercised in every episode.
                for i in range(N_STEPS):
                    out.append({'cmd': 'step', 'action': (i * 3 + 2) % N_ACTIONS})
    # The unknown-preset error path, which the port moved behind a source check.
    out.append({'cmd': 'reset', 'preset': 'no-such-preset',
                'duration_s': 3600, 'seed': 1})
    return out


def compare(a: dict, b: dict, ctx: str, diffs: list[str]) -> None:
    if a.get('ok') != b.get('ok'):
        diffs.append(f'{ctx}: ok differs {a.get("ok")} vs {b.get("ok")}')
        return
    for key in ('reward', 'done', 'action', 'error', 'n_actions'):
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
    n_state_cmp = 0
    try:
        for n, cmd in enumerate(cmds):
            if cmd['cmd'] == 'reset':
                ctx = f"preset={cmd['preset']} seed={cmd['seed']}"
            ra, rb = old.send(cmd), new.send(cmd)
            if ra.get('state') is not None:
                n_state_cmp += 1
            compare(ra, rb, f'[{n}] {ctx} {cmd["cmd"]}', diffs)
            if len(diffs) > 20:
                break
    finally:
        old.close()
        new.close()

    print(f'compared {len(cmds)} commands '
          f'({len(FLAG_SETS)} flag sets × {len(PRESETS)} presets × {len(SEEDS)} seeds, '
          f'{n_state_cmp} state vectors)')
    if diffs:
        print(f'\n{len(diffs)} DIFFERENCE(S):')
        for d in diffs[:20]:
            print(f'  {d}')
        return 1
    print('identical — gassand preset path unchanged')
    return 0


if __name__ == '__main__':
    sys.exit(main())

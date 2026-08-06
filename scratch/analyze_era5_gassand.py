#!/usr/bin/env python3
"""
analyze_era5_gassand.py — read the paired ERA5 gassand results and answer the
one question the raw means cannot: is the ceiling the policy or the environment?

The discriminator is heuristic − float. The hand-coded navigator is not a good
policy, but it is a *deliberate* one, so if real wind contains altitude
structure worth exploiting the heuristic should find some of it and beat the
do-nothing baseline. On v2/ERA5 it did, by +3.80 pp at t = 4.05, which is what
made R's failure there readable as a failure. If it does not clear the floor
here, no statement about R_Gassand's skill is available from this data — the
environment is the binding constraint.

Adds to what eval_era5_gassand.py already printed:
  * two-sided p-values (t distribution, n−1 df) via a normal-tail approximation
    good to ~1e-4 at these sample sizes, plus a bootstrap CI that assumes nothing
  * a re-entry diagnostic: of the episodes that leave the 50 km radius, how many
    ever come back? TWR50 is indistinguishable from time-to-escape if none do.

Usage:
    python scratch/analyze_era5_gassand.py [--results weights/era5_eval_gassand_100ep.json]
                                           [--reentry <era5_dir> --reentry-episodes 30]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def t_sf_two_sided(t: float, df: int) -> float:
    """
    Two-sided tail probability for Student's t. Uses the incomplete beta
    relation I_x(df/2, 1/2) with x = df/(df+t²), evaluated by a continued
    fraction — exact enough to quote at three decimals.
    """
    t = abs(t)
    if t == 0:
        return 1.0
    x = df / (df + t * t)
    return _betainc(df / 2.0, 0.5, x)


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a,b), Lentz continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _betacf(a: float, b: float, x: float, itmax: int = 300, eps: float = 3e-16) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def bootstrap_ci(d: list[float], n_boot: int = 20000, seed: int = 12345) -> tuple[float, float]:
    """Percentile bootstrap CI on the paired mean. TWR50 is badly skewed, so the
    t interval is worth a second opinion."""
    rng = random.Random(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        means.append(sum(d[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def report_pair(res: dict, a: str, b: str) -> dict:
    sa = res['policies'][a]['twr50']
    sb = res['policies'][b]['twr50']
    d = [x - y for x, y in zip(sa, sb)]
    n = len(d)
    m = statistics.mean(d)
    sd = statistics.stdev(d)
    t = m / (sd / n ** 0.5)
    p = t_sf_two_sided(t, n - 1)
    lo, hi = bootstrap_ci(d)
    verdict = 'SIGNIFICANT' if p < 0.05 else 'not distinguishable'
    print(f'  {a:12s} − {b:12s}  {m*100:+6.2f} pp   t = {t:+5.2f}   p = {p:.3f}   '
          f'95% CI [{lo*100:+.2f}, {hi*100:+.2f}] pp   {verdict}')
    return {'delta_pp': m * 100, 't': t, 'p': p,
            'ci95_pp': [lo * 100, hi * 100], 'n': n}


def reentry_diagnostic(era5_dir: str, episodes: int, results: dict) -> dict:
    """
    Of the episodes that leave the radius, how many ever return? If essentially
    none do, TWR50 on this env is measuring time-to-escape from the 30 km spawn,
    not station-keeping, and no policy ranking off it means much.
    """
    from replay_gassand import load_agent, run_episode
    from scratch.eval_era5_gassand import (POLICY_WEIGHTS, REALISM_FLAGS, FloatAgent)

    seeds = results['seeds'][:episodes]
    out = {}
    print(f'\nre-entry diagnostic ({episodes} episodes, same seeds):')
    for name in ('r_gassand', 'heuristic', 'float'):
        w = POLICY_WEIGHTS[name]
        agent = (FloatAgent() if w == 'FLOAT'
                 else None if w is None else load_agent(REPO / w))
        left = returned = 0
        escape_h = []
        for seed in seeds:
            traj = run_episode('tropical', 72 * 3600, seed, agent=agent,
                               flags=REALISM_FLAGS, wind_source='era5',
                               era5_dir=era5_dir)
            inr = list(traj['in_radius'])
            if not any(inr) or all(inr):
                continue
            first_out = inr.index(False)
            left += 1
            escape_h.append(first_out * 300 / 3600)
            if any(inr[first_out:]):
                returned += 1
        out[name] = {'left': left, 'returned': returned,
                     'median_escape_h': statistics.median(escape_h) if escape_h else None}
        pct = 100 * returned / left if left else 0.0
        print(f'  {name:12s} left the radius in {left}/{len(seeds)} episodes; '
              f'{returned} ever came back ({pct:.0f}%)   '
              f'median time to first exit {statistics.median(escape_h):.1f} h')
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results',
                    default=str(REPO / 'weights' / 'era5_eval_gassand_100ep.json'))
    ap.add_argument('--reentry', default=None, metavar='ERA5_DIR',
                    help='also run the re-entry diagnostic against this archive')
    ap.add_argument('--reentry-episodes', type=int, default=30)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    res = json.loads(Path(args.results).read_text())
    n = res['episodes']
    print(f'ERA5 gassand — {n} paired episodes x 72 h\n')

    print('TWR50 by policy (mean / median / max):')
    for name, p in res['policies'].items():
        print(f'  {name:12s} {p["mean"]*100:5.2f}%  {p["median"]*100:5.2f}%  '
              f'{p["max"]*100:5.1f}%     '
              f'He kept {p["mean_he_left_kg"]:6.3f}/19.24 kg   '
              f'sand kept {p["mean_sand_left_kg"]:5.2f}/20 kg')

    print('\npaired deltas with p-values:')
    pairs = [('heuristic', 'float'), ('r_gassand', 'float'),
             ('r_gassand', 'heuristic'), ('res_gassand', 'r_gassand'),
             ('gassand', 'float'), ('res_gassand', 'float')]
    stats = {f'{a}_vs_{b}': report_pair(res, a, b) for a, b in pairs}

    ctrl = stats['heuristic_vs_float']
    print('\nCONTROL READ — heuristic vs the do-nothing floor:')
    if ctrl['p'] < 0.05:
        print('  The navigator beats FLOAT, so exploitable altitude structure is')
        print('  present and any policy near the floor is failing to use it.')
    else:
        print(f'  The navigator does NOT beat FLOAT ({ctrl["delta_pp"]:+.2f} pp, '
              f'p = {ctrl["p"]:.3f}).')
        print('  Deliberate altitude control buys nothing measurable here, so this')
        print('  data cannot separate a bad policy from a hostile environment —')
        print('  the environment is the binding constraint.')

    report = {'source': args.results, 'episodes': n, 'paired_stats': stats}
    if args.reentry:
        report['reentry'] = reentry_diagnostic(args.reentry, args.reentry_episodes, res)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f'\nwrote {args.out}')
    print('ANALYZE_DONE')
    return 0


if __name__ == '__main__':
    sys.exit(main())

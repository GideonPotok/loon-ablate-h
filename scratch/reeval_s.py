"""Clean 10-seed re-eval of Ablation S's winner checkpoint.

Same protocol as reeval_r.py (job c0a2d544): seeds 42+i*1_000_003, 72h,
composite 0.5*mean + 0.5*worst. Realism = S's training env; legacy = the
4 realism flags off but the estimator stays ON (state must remain 24-dim;
the estimator is computable in the deterministic env too).
"""
import json
import os
import statistics
import sys
from pathlib import Path

WORKTREE = '/Users/gideonpotok/repos/loon-ablate-h/.claude/worktrees/ablate-s'
sys.path.insert(0, WORKTREE)
os.chdir(WORKTREE)

from replay import ABLATION_ENV_FLAGS, load_agent, run_episode

WEIGHTS = Path('/Users/gideonpotok/repos/loon-ablate-h/weights/dqn_ablate_s.pt')
OUT = Path('/Users/gideonpotok/.claude/jobs/8bdc873d/tmp/s_reeval.json')
PRESETS = ['tropical', 'strong-shear', 'calm']
SEEDS = 10
DURATION_S = 72 * 3600

agent = load_agent(WEIGHTS, None)   # feedforward; state_dim 24 from ckpt

realism = dict(ABLATION_ENV_FLAGS['s'])
legacy = {**realism, 'wind_phase_jitter': False, 'wind_episode_noise': False,
          'wind_param_jitter': False, 'domain_rand': False}

results = {}
for mode, flags in (('realism', realism), ('legacy', legacy)):
    results[mode] = {}
    for preset in PRESETS:
        scores = []
        for i in range(SEEDS):
            seed = 42 + i * 1_000_003
            traj = run_episode(agent, preset, DURATION_S, seed,
                               server_version='v2', flags=flags)
            scores.append(traj['twr50'])
        m, sd = statistics.mean(scores), statistics.stdev(scores)
        print(f'{mode:8s} {preset:13s} TWR50 {m*100:5.1f}% ± {sd*100:4.1f}pp '
              f'({SEEDS} seeds)', flush=True)
        results[mode][preset] = scores

for mode in list(results):
    means = {p: statistics.mean(results[mode][p]) for p in PRESETS}
    score = 0.5 * statistics.mean(list(means.values())) + 0.5 * min(means.values())
    results[f'{mode}_composite'] = score
    print(f'{mode:8s} composite (0.5*mean + 0.5*worst) = {score*100:.1f}%', flush=True)

OUT.write_text(json.dumps(results, indent=2))
print('REEVAL_DONE')

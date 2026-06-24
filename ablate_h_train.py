"""
Ablation H — Extended ablate_a (plain DQN, v1 server, 3600 episodes):
  Same config as Ablation A but with a longer curriculum (800 extra eps).

Question this answers:
  "Was ablate_a still improving at ep 2799 because it hadn't converged,
   or was it close to its ceiling? Does more training close the 4pp gap
   with the heuristic?"

Changes vs Ablation A:
  - 48h tier: 400 → 800 eps
  - New 72h tier: 400 eps  (eval also uses 72h, so train/eval horizons align)
  - Total: 3600 eps/worker (vs 2800 in A)
  - Everything else identical (n_q=1, v1 server, PER, n-step=3)

Usage:
    python ablate_h_train.py
"""
from __future__ import annotations

import json
import time
import multiprocessing as mp
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

from qr_agent import QRAgent, QRConfig
from replay_buffer import PrioritizedReplayBuffer, NStepAccumulator
from balloon_env import BalloonEnv

# ── Hyperparameters ───────────────────────────────────────────────────────────

CURRICULUM = [
    {'episodes':  200, 'duration_s': 3600 *  2, 'label':  '2h'},
    {'episodes': 1000, 'duration_s': 3600 *  6, 'label':  '6h'},
    {'episodes':  600, 'duration_s': 3600 * 12, 'label': '12h'},
    {'episodes':  600, 'duration_s': 3600 * 24, 'label': '24h'},
    {'episodes':  800, 'duration_s': 3600 * 48, 'label': '48h'},  # doubled vs A
    {'episodes':  400, 'duration_s': 3600 * 72, 'label': '72h'},  # new tier
]
TOTAL_EPS       = sum(t['episodes'] for t in CURRICULUM)   # 3600
PRESETS         = ['tropical', 'strong-shear', 'calm']
N_WORKERS       = 10
EVAL_EVERY      = 300
EVAL_RUNS       = 3
EVAL_DURATION_S = 3600 * 72

BASE_CONFIG = QRConfig(
    state_dim         = 20,            # v1 state
    hidden_sizes      = [128, 64],
    action_count      = 17,
    n_quantiles       = 1,             # plain DQN, same as A
    huber_kappa       = 1.0,
    learning_rate     = 1e-4,
    optimizer         = 'adam',
    gamma             = 0.97,
    epsilon_start     = 1.0,
    epsilon_end       = 0.03,
    epsilon_decay     = 0.9988,
    target_update_freq = 15,
    replay_capacity   = 100_000,
    batch_size        = 64,
    n_step            = 3,
    per_alpha         = 0.6,
    per_beta0         = 0.4,
    per_beta_anneal   = 1e-4,
    cvar_alpha        = 1.0,           # no-op when n_q=1
    train_batches_per_step = 2,
    device            = 'cpu',
    # v1 server — all v2 flags OFF
    use_reward_fix     = False,
    use_shaping        = False,
    use_expanded_state = False,
    use_recurrent      = False,
    use_options        = False,
)

WEIGHTS_DIR    = Path(__file__).parent / 'weights'
LOG_PATH       = Path('/tmp/train_ablate_h.log')
WEIGHTS_PREFIX = 'dqn_ablate_h'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tier_at(ep: int) -> dict:
    cum = 0
    for tier in CURRICULUM:
        cum += tier['episodes']
        if ep < cum:
            return tier
    return CURRICULUM[-1]


def _eval_multi_preset(agent: QRAgent, ep: int, seed: int,
                       n_runs: int, duration_s: float) -> dict:
    per_preset: dict[str, float] = {}
    all_scores: list[float] = []
    worst_preset = None
    worst_twr = float('inf')

    for pi, preset in enumerate(PRESETS):
        scores = []
        for r in range(n_runs):
            eval_seed = seed + 1_000_000 + ep * 1000 + pi * 17 + r
            env = BalloonEnv(preset=preset, duration_s=duration_s, seed=eval_seed,
                             server_version='v1')
            state = env.reset()
            done = False
            twr50 = 0.0
            while not done:
                action = agent.select_action(state)
                state, _, done, info = env.step(action)
                twr50 = info.get('twr50', twr50)
            scores.append(twr50)
            env.close()

        mean_p = float(np.mean(scores))
        per_preset[preset] = mean_p
        all_scores.extend(scores)
        if mean_p < worst_twr:
            worst_twr = mean_p
            worst_preset = preset

    mean_twr50 = float(np.mean(all_scores))
    score      = 0.5 * mean_twr50 + 0.5 * worst_twr
    return {
        'score':        score,
        'mean':         mean_twr50,
        'worst':        worst_twr,
        'worst_preset': worst_preset,
        'per_preset':   per_preset,
    }


# ── Worker ────────────────────────────────────────────────────────────────────

def worker_fn(worker_id: int, result_queue: mp.Queue):
    seed = 42 + worker_id * 1_000_003
    config = replace(BASE_CONFIG, seed=seed)

    agent  = QRAgent(config)
    per_buf = PrioritizedReplayBuffer(
        config.replay_capacity, config.per_alpha, config.per_beta0, seed=seed + 1,
    )
    n_acc = NStepAccumulator(config.n_step, config.gamma, per_buf)
    rng   = np.random.default_rng(seed * 31 + 7919)

    best_score      = -float('inf')
    best_weights    = None
    best_per_preset = None
    best_episode    = -1
    start_ts        = time.time()

    result_queue.put({'type': 'start', 'worker_id': worker_id, 'seed': seed,
                      'total_episodes': TOTAL_EPS})

    for ep in range(TOTAL_EPS):
        tier   = _tier_at(ep)
        preset = PRESETS[ep % len(PRESETS)]
        ep_seed = int(rng.integers(1_000_000_000))

        env = BalloonEnv(preset=preset, duration_s=tier['duration_s'], seed=ep_seed,
                         server_version='v1')
        state = env.reset()
        n_acc.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            n_acc.push(state, action, reward, next_state, done)
            n_acc.flush_to_buffer(next_state, episode_done=done)
            for _ in range(config.train_batches_per_step):
                agent.train_batch(per_buf)
            state = next_state

        env.close()
        agent.decay_epsilon()

        if (ep + 1) % EVAL_EVERY == 0 or ep == TOTAL_EPS - 1:
            ev = _eval_multi_preset(agent, ep, seed, EVAL_RUNS, EVAL_DURATION_S)
            new_best = ev['score'] > best_score
            if new_best:
                best_score      = ev['score']
                best_per_preset = ev['per_preset']
                best_episode    = ep
                best_weights    = agent.state_dict()
                ckpt_path = WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_w{worker_id:02d}.pt'
                torch.save(best_weights, ckpt_path)
                (WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_w{worker_id:02d}.json').write_text(
                    json.dumps({'best_score': best_score, 'best_episode': best_episode,
                                'best_per_preset': best_per_preset, 'worker_id': worker_id})
                )

            result_queue.put({
                'type':       'eval',
                'worker_id':  worker_id,
                'ep':         ep,
                'elapsed_s':  time.time() - start_ts,
                'tier':       tier['label'],
                'epsilon':    agent.epsilon,
                'is_best':    new_best,
                **ev,
            })

    result_queue.put({
        'type':            'done',
        'worker_id':       worker_id,
        'elapsed_s':       time.time() - start_ts,
        'best_episode':    best_episode,
        'best_score':      best_score,
        'best_per_preset': best_per_preset,
        'best_weights':    best_weights,
    })


# ── Launcher ──────────────────────────────────────────────────────────────────

def _fmt_pct(x: float) -> str: return f'{x * 100:5.1f}%'
def _fmt_time(s: float) -> str:
    m, sec = divmod(int(s), 60); return f'{m}m{sec:02d}s'


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, 'w', buffering=1)
    def tee(line: str):
        print(line); log.write(line + '\n')

    n_params = sum(p.numel() for p in QRAgent(BASE_CONFIG).policy_net.parameters())
    tee('═' * 78)
    tee('ABLATION H: extended ablate_a — plain DQN, v1 server, 3600 eps')
    tee('═' * 78)
    tee(f'Workers:     {N_WORKERS}')
    tee(f'Network:     {BASE_CONFIG.state_dim} → {" → ".join(str(h) for h in BASE_CONFIG.hidden_sizes)} '
        f'→ {BASE_CONFIG.action_count}   ({n_params:,} params)')
    tee('Curriculum:  ' + '  →  '.join(
        f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM) + f'   total {TOTAL_EPS} eps/worker')
    tee(f'n_quantiles: 1  PER+n-step=3  server: v1 (all v2 flags OFF)')
    tee(f'Extension:   +800 eps vs ablate_a (3600 total vs 2800)')
    tee('─' * 78)

    result_queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=worker_fn, args=(wid, result_queue), daemon=True)
        for wid in range(N_WORKERS)
    ]
    for p in processes:
        p.start()

    worker_results: list[dict] = []
    done_count = 0
    launch_ts = time.time()

    while done_count < N_WORKERS:
        msg = result_queue.get()
        wid = msg['worker_id']
        tag = f'[w{wid:02d}]'

        if msg['type'] == 'start':
            tee(f'  {tag} started  seed={msg["seed"]}  total={msg["total_episodes"]} eps')
        elif msg['type'] == 'eval':
            best_mark = ' ★' if msg['is_best'] else '  '
            pp = msg['per_preset']
            tee(
                f'  {tag} {_fmt_time(msg["elapsed_s"]):>7}  ep {msg["ep"]:4d} [{msg["tier"]:3s}]'
                f'  score {_fmt_pct(msg["score"])}'
                f'  mean {_fmt_pct(msg["mean"])}'
                f'  worst({msg["worst_preset"]:<13}) {_fmt_pct(msg["worst"])}'
                f'  trop {_fmt_pct(pp["tropical"])}'
                f'  shear {_fmt_pct(pp["strong-shear"])}'
                f'  calm {_fmt_pct(pp["calm"])}'
                f'  ε {msg["epsilon"]:.3f}'
                + best_mark
            )
        elif msg['type'] == 'done':
            done_count += 1
            worker_results.append(msg)
            w = msg['worker_id']
            bp = msg.get('best_per_preset') or {}
            tee(
                f'  [w{w:02d}] DONE  {_fmt_time(msg["elapsed_s"])}'
                f'  best ep {msg["best_episode"]}'
                f'  score {_fmt_pct(msg["best_score"])}'
                + (f'  trop {_fmt_pct(bp.get("tropical", 0))}'
                   f'  shear {_fmt_pct(bp.get("strong-shear", 0))}'
                   f'  calm {_fmt_pct(bp.get("calm", 0))}' if bp else '')
            )

    for p in processes:
        p.join()

    winner = max(worker_results, key=lambda r: r['best_score'])
    wid    = winner['worker_id']
    tee('')
    tee('─' * 78)
    tee(f'Winner: w{wid:02d}  score {_fmt_pct(winner["best_score"])}  ep {winner["best_episode"]}')

    out_path = WEIGHTS_DIR / f'{WEIGHTS_PREFIX}.pt'
    torch.save(winner['best_weights'], out_path)

    summary = {
        'ablation':          'H_extended_ablate_a',
        'winner_worker':     wid,
        'best_score':        winner['best_score'],
        'best_episode':      winner['best_episode'],
        'best_per_preset':   winner['best_per_preset'],
        'wall_time_s':       time.time() - launch_ts,
        'ablate_a_best':     0.491,
        'ablate_a_best_ep':  2799,
        'workers': [
            {'worker_id': r['worker_id'], 'best_score': r['best_score'],
             'best_episode': r['best_episode']}
            for r in worker_results
        ],
    }
    (WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_summary.json').write_text(json.dumps(summary, indent=2))
    tee(f'Summary: {WEIGHTS_DIR / f"{WEIGHTS_PREFIX}_summary.json"}')
    tee(f'Total wall time: {_fmt_time(time.time() - launch_ts)}')
    log.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()

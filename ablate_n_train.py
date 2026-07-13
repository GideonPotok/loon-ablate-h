"""
Ablation N — Horizon + Curriculum (v2 server, 3600 episodes):

Question this answers:
  "L gave the agent exact knowledge of where it is in the 8h IGW cycle —
   does it actually help if the discount horizon is too short to act on it,
   and the curriculum barely visits episodes long enough for reversals to occur?"

Problem with Ablation L:
  L adds lossless sin/cos(2π·t/T_IGW) features (confirmed against js/wind.js:
  IGW phase is a deterministic function of elapsed episode time, no per-episode
  random offset), so the agent can always tell what phase it's in. But L kept
  K's gamma=0.97, which gives an effective horizon (γⁿ=0.1 ⇒ n≈75 steps × 5 min)
  of ~6.25h — *shorter* than the 8h IGW period itself. The value function cannot
  back-propagate "wind reverses in 6 more hours" into today's altitude choice;
  that event sits past where TD credit has decayed to noise. Separately, L's
  curriculum is unchanged from K: 1800/3600 episodes (50%) are ≤12h, so even if
  the agent could see further, it rarely trains on episodes long enough to
  experience and be corrected for getting an IGW reversal wrong.
  (Both of these were confirmed never touched across K → K2 → L → M by diffing
  BASE_CONFIG / CURRICULUM across all four scripts.)

Change vs Ablation L:
  1. gamma: 0.97 → 0.99
     Effective horizon ⇒ n≈229 steps × 5 min ≈ 19h — comfortably covers a full
     8h IGW cycle (and then some) instead of falling 1.75h short of just one.
  2. target_update_freq: 15 → 25
     Stabilization paired with the gamma raise: longer effective horizons mean
     bootstrapped TD targets compound over more steps, so a stale target network
     has more opportunity to amplify oscillation. Slowing target refresh damps
     this without touching any other learning dynamics (n_step, batch_size,
     learning_rate all unchanged from L to keep this a 2-variable test).
  3. Curriculum rebalanced toward long episodes, same total budget (3600 eps):
       2h:  200 → 100   6h: 1000 → 500   12h: 600 → 500   (≤12h: 50% → 31%)
       24h:  600 → 900  48h:  800 → 1000  72h: 400 → 600   (≥24h: 50% → 69%)
     Short tiers are kept (not eliminated) — they still matter for early policy
     bootstrapping — but exposure shifts toward the durations where IGW
     reversals and planetary-wave drift actually have to be handled and
     corrected for.

  Everything else identical to L: state_dim=24 (Fourier time features),
  exponential shaping (τ=500 km), recovery spawn, v2 server, binary reward,
  [128,64] arch, n_quantiles=1, n_step=3, lr=1e-4, batch_size=64, 3600 eps/worker.
  No capacity or sample-count change (hidden_sizes unchanged, same total
  episodes) — both K2 and L plateaued at the same [128,64] ceiling on hard
  presets, which is not the signature of a capacity- or data-starved run, so
  this ablation isolates horizon + curriculum exposure rather than conflating
  them with "bigger net" / "longer training".

Also fixes an eval-contamination bug present in every prior ablate_*_train.py:
  _eval_multi_preset called agent.select_action(state) without suppressing
  ε-greedy exploration. epsilon is logged at eval time but was never actually
  zeroed for the eval rollout itself — it only floors at 0.03 and starts at
  1.0, so every previous ablation's reported TWR50 had between 3% and 100%
  literal random actions mixed into the metric depending on where eval landed
  in the decay schedule. qr_agent.py's select_action()/_select_action_options()
  now take a `greedy` flag; this script's eval loop passes greedy=True so the
  scores being compared across h/i/j/k/l/n are the actual learned policy, not
  policy-plus-noise. (Training rollouts are untouched — they still need
  ε-exploration.)

Usage:
    python ablate_n_train.py
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
    {'episodes':  100, 'duration_s': 3600 *  2, 'label':  '2h'},
    {'episodes':  500, 'duration_s': 3600 *  6, 'label':  '6h'},
    {'episodes':  500, 'duration_s': 3600 * 12, 'label': '12h'},
    {'episodes':  900, 'duration_s': 3600 * 24, 'label': '24h'},
    {'episodes': 1000, 'duration_s': 3600 * 48, 'label': '48h'},
    {'episodes':  600, 'duration_s': 3600 * 72, 'label': '72h'},
]
TOTAL_EPS       = sum(t['episodes'] for t in CURRICULUM)   # 3600 (same budget as L)
PRESETS         = ['tropical', 'strong-shear', 'calm']
N_WORKERS       = 10
EVAL_EVERY      = 300
EVAL_RUNS       = 3
EVAL_DURATION_S = 3600 * 72

# ── Recovery spawn parameters (carried from ablation J) ───────────────────────

RECOVERY_SPAWN_PROB    = 0.30
RECOVERY_SPAWN_MIN_KM  = 150.0
RECOVERY_SPAWN_MAX_KM  = 500.0
RECOVERY_SPAWN_MIN_DUR = 3600 * 24

SHAPING_TAU_KM = 500.0  # exponential shaping length scale (unchanged from K/L)

BASE_CONFIG = QRConfig(
    state_dim         = 24,            # 20-dim base + 4 Fourier time features (from L)
    hidden_sizes      = [128, 64],
    action_count      = 17,
    n_quantiles       = 1,
    huber_kappa       = 1.0,
    learning_rate     = 1e-4,
    optimizer         = 'adam',
    gamma             = 0.99,          # was 0.97 in K/K2/L/M — effective horizon 6.25h → ~19h
    epsilon_start     = 1.0,
    epsilon_end       = 0.03,
    epsilon_decay     = 0.9988,
    target_update_freq = 25,           # was 15 — stabilizes longer bootstrap chains at higher gamma
    replay_capacity   = 100_000,
    batch_size        = 64,
    n_step            = 3,
    per_alpha         = 0.6,
    per_beta0         = 0.4,
    per_beta_anneal   = 1e-4,
    cvar_alpha        = 1.0,
    train_batches_per_step = 2,
    device            = 'cpu',
    use_reward_fix     = False,
    use_shaping        = False,
    use_expanded_state = False,
    use_recurrent      = False,
    use_options        = False,
)

WEIGHTS_DIR    = Path(__file__).parent / 'weights'
LOG_PATH       = Path('/tmp/train_ablate_n.log')
WEIGHTS_PREFIX = 'dqn_ablate_n'


def _env_flags() -> dict:
    return {
        'use_reward_fix':     True,
        'use_shaping':        True,
        'use_expanded_state': False,
        'use_time_features':  True,              # 20 → 24 dim (from L)
        'shaping_beta':       0.5,
        'shaping_gamma':      0.97,               # shaping potential's own gamma, unrelated to agent's TD gamma
        'terminal_twr_bonus': 50.0,
        'shaping_linear':     False,             # exponential (from K)
        'shaping_D_max':      SHAPING_TAU_KM * 1000.0,
    }


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
                             server_version='v2', flags=_env_flags())
            state = env.reset()
            done = False
            twr50 = 0.0
            while not done:
                action = agent.select_action(state, greedy=True)  # no ε-noise in eval scores
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

def worker_fn(worker_id: int, result_queue: mp.Queue, max_episodes: int = 0):
    seed = 42 + worker_id * 1_000_003
    config = replace(BASE_CONFIG, seed=seed)

    n_eps = min(TOTAL_EPS, max_episodes) if max_episodes > 0 else TOTAL_EPS

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
                      'total_episodes': n_eps})

    for ep in range(n_eps):
        tier   = _tier_at(ep)
        preset = PRESETS[ep % len(PRESETS)]
        ep_seed = int(rng.integers(1_000_000_000))

        spawn_km = None
        if tier['duration_s'] >= RECOVERY_SPAWN_MIN_DUR and rng.random() < RECOVERY_SPAWN_PROB:
            spawn_km = float(rng.uniform(RECOVERY_SPAWN_MIN_KM, RECOVERY_SPAWN_MAX_KM))

        env = BalloonEnv(preset=preset, duration_s=tier['duration_s'], seed=ep_seed,
                         server_version='v2', flags=_env_flags())
        state = env.reset(spawn_offset_km=spawn_km)
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

        if (ep + 1) % EVAL_EVERY == 0 or ep == n_eps - 1:
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
    tee('ABLATION N: horizon + curriculum — v2 server, 24-dim state, 3600 eps')
    tee('═' * 78)
    tee(f'Workers:     {N_WORKERS}')
    tee(f'Network:     {BASE_CONFIG.state_dim} → {" → ".join(str(h) for h in BASE_CONFIG.hidden_sizes)} '
        f'→ {BASE_CONFIG.action_count}   ({n_params:,} params)')
    tee('Curriculum:  ' + '  →  '.join(
        f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM) + f'   total {TOTAL_EPS} eps/worker')
    tee(f'n_quantiles: 1  PER+n-step=3  server: v2')
    tee(f'Gamma:       {BASE_CONFIG.gamma}  (was 0.97 in K/L)  target_update_freq={BASE_CONFIG.target_update_freq} (was 15)')
    tee(f'Shaping:     exponential  β=0.5  γ_shape=0.97  τ={SHAPING_TAU_KM:.0f} km (10R)')
    tee(f'Time features: sin/cos(2π·t/8h) + sin/cos(2π·t/5d)  →  state 20→24 (from L)')
    tee(f'Recovery spawn: {RECOVERY_SPAWN_PROB*100:.0f}% of ≥24h episodes  '
        f'range [{RECOVERY_SPAWN_MIN_KM:.0f}, {RECOVERY_SPAWN_MAX_KM:.0f}] km')
    tee(f'Eval:        greedy (ε suppressed) — fixes noise contamination present in h/i/j/k/l eval scores')
    tee(f'Change vs L: gamma 0.97→0.99, target_update_freq 15→25, curriculum rebalanced '
        f'≤12h 50%→31% / ≥24h 50%→69% (same 3600-ep budget)')
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
        'ablation':              'N_horizon_curriculum',
        'winner_worker':         wid,
        'best_score':            winner['best_score'],
        'best_episode':          winner['best_episode'],
        'best_per_preset':       winner['best_per_preset'],
        'wall_time_s':           time.time() - launch_ts,
        'gamma':                 BASE_CONFIG.gamma,
        'target_update_freq':    BASE_CONFIG.target_update_freq,
        'shaping':               'exponential',
        'shaping_tau_km':        SHAPING_TAU_KM,
        'state_dim':             BASE_CONFIG.state_dim,
        'time_features':         ['sin_igw', 'cos_igw', 'sin_pw', 'cos_pw'],
        'curriculum':            [{'label': t['label'], 'episodes': t['episodes']} for t in CURRICULUM],
        'recovery_spawn_prob':   RECOVERY_SPAWN_PROB,
        'recovery_spawn_min_km': RECOVERY_SPAWN_MIN_KM,
        'recovery_spawn_max_km': RECOVERY_SPAWN_MAX_KM,
        'eval_greedy':           True,
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

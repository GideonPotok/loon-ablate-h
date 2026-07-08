"""
Ablation T — Memory arm: plain GRU (no options) in the realism env
(v2 server, 3600 episodes):

Question this answers:
  "Can memory *learn* phase inference? In the realism env the IGW/PW phases
   are per-episode random and can only be recovered by watching the wind
   evolve — exactly the partially-observable setting recurrence was designed
   for. T gives R's recipe a GRU hidden state and nothing else: no options
   (M/O/P/Q showed the option-critic machinery underperforms here — Q's
   clean re-eval matched M at ~25% despite 8× the gradient steps), no
   engineered estimator. T ≈ S means the GRU learns what the hand-built
   demodulator computes (the recurrent rationale finally vindicated in the
   setting it was built for); T ≈ R means memory fails to learn it and
   explicit estimation wins."

Background:
  R/S/T form the realism-era comparison — same stochastic environment (all
  four realism flags on, train AND eval), varying only the agent's
  information structure: R = no phase information (20-dim feedforward),
  S = engineered estimator features (24-dim feedforward), T = GRU memory
  (20-dim + hidden state). Scores are NOT comparable to H–Q, which lived in
  the deterministic wind; the cross-env bridge is probe_realism_transfer.py.

Environment (identical to R — 20-dim state, no phase features of any kind):
  wind_phase_jitter   — per-episode φ_igw, φ_pw ~ U[0,2π)
  wind_episode_noise  — per-episode seed mixed into the background-noise hash
  wind_param_jitter   — per-episode IGW/PW amplitude × logU[0.7, 1.4]
  domain_rand         — per-episode forecast-degrader σ-scale logU[0.5, 2.0]
                        + forecast lag U[0, 6h]

Changes vs Ablation R (information structure only):
  1. use_recurrent=True, gru_hidden=64 (use_options stays False — plain
     R2D2-style recurrent DQN, not option-critic).
  2. Q's sequence-replay machinery, unchanged: EpisodeSequenceBuffer
     (500 episodes), burn-in 16 + train 16 windows, batch 32 windows,
     per-step training every TRAIN_EVERY_STEPS=2 env steps via
     train_batch_seq (zero-init + burn-in hidden warm-up).
  Everything else identical to R: γ=0.99, target_update 25, lr 1e-4,
  n_step 3, [128,64] encoder, n_quantiles=1, N's rebalanced curriculum
  (3600 eps), exponential shaping (τ=500 km), recovery spawn, greedy eval.

Wall-clock: Q ran this exact cadence + sequence machinery WITH option-critic
heads in ~3h10m/worker on CI (6h cap); T's single Q-head GRU is lighter, so
the budget is comfortable.

Usage:
    python ablate_t_train.py
"""
from __future__ import annotations

import json
import time
import multiprocessing as mp
from collections import deque
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

from qr_agent import QRAgent, QRConfig
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

# One gradient step per this many env steps (Q's cadence — a 32-window GRU
# sequence batch trains 512 transitions/gradient step vs R's 128/env step).
TRAIN_EVERY_STEPS = 2

# ── Recovery spawn parameters (carried from ablation J) ───────────────────────

RECOVERY_SPAWN_PROB    = 0.30
RECOVERY_SPAWN_MIN_KM  = 150.0
RECOVERY_SPAWN_MAX_KM  = 500.0
RECOVERY_SPAWN_MIN_DUR = 3600 * 24

SHAPING_TAU_KM = 500.0  # exponential shaping length scale (unchanged from K/L)

BASE_CONFIG = QRConfig(
    state_dim         = 20,            # same as R — memory, not features, is the variable
    hidden_sizes      = [128, 64],
    action_count      = 17,
    n_quantiles       = 1,
    huber_kappa       = 1.0,
    learning_rate     = 1e-4,
    optimizer         = 'adam',
    gamma             = 0.99,          # R's value — effective horizon ~19h
    epsilon_start     = 1.0,
    epsilon_end       = 0.03,
    epsilon_decay     = 0.9988,
    target_update_freq = 25,           # R's value
    replay_capacity   = 100_000,       # unused for seq replay but kept for compat
    batch_size        = 32,            # sequence windows (32 × 16 trained steps)
    n_step            = 3,
    per_alpha         = 0.6,
    per_beta0         = 0.4,
    per_beta_anneal   = 1e-4,
    cvar_alpha        = 1.0,
    train_batches_per_step = 0,        # cadence handled by TRAIN_EVERY_STEPS above
    device            = 'cpu',
    use_reward_fix     = False,
    use_shaping        = False,
    use_expanded_state = False,
    use_recurrent      = True,         # GRU hidden state — the one new ingredient
    use_options        = False,        # plain recurrent DQN, no option-critic
    gru_hidden         = 64,
    seq_burn_in        = 16,
    seq_train          = 16,
    grad_clip_norm     = None,         # L/R have no clipping; P showed it never engages
)

WEIGHTS_DIR    = Path(__file__).parent / 'weights'
LOG_PATH       = Path('/tmp/train_ablate_t.log')
WEIGHTS_PREFIX = 'dqn_ablate_t'

SEQ_LEN           = BASE_CONFIG.seq_burn_in + BASE_CONFIG.seq_train   # 32
SEQ_BUF_CAPACITY  = 500   # episodes (Q's value)


def _env_flags() -> dict:
    return {
        'use_reward_fix':     True,
        'use_shaping':        True,
        'use_expanded_state': False,
        'use_time_features':  False,             # oracle removed — state stays 20-dim
        'shaping_beta':       0.5,
        'shaping_gamma':      0.97,               # shaping potential's own gamma, unrelated to agent's TD gamma
        'terminal_twr_bonus': 50.0,
        'shaping_linear':     False,             # exponential (from K)
        'shaping_D_max':      SHAPING_TAU_KM * 1000.0,
        # Realism bundle (train AND eval — same stochastic world everywhere):
        'wind_phase_jitter':  True,
        'wind_episode_noise': True,
        'wind_param_jitter':  True,
        'domain_rand':        True,
    }


# ── Sequence replay buffer (lifted from ablate_q_train.py) ─────────────────────

class EpisodeSequenceBuffer:
    """
    Stores complete episodes and samples fixed-length windows for R2D2 training.

    Each episode is a list of 7-tuples:
        (state, action, n_step_return, bootstrap_state, eff_gamma, done, option)
    Sampling: pick a random eligible episode, then a random L-step window from it.
    Only episodes with >= L transitions are eligible. The option column is a
    carried-over Q-format placeholder (always 0 here); train_batch_seq ignores it.
    """

    def __init__(self, capacity: int, seq_len: int, seed: int = 42):
        self._capacity = capacity
        self._seq_len  = seq_len
        self._episodes: deque = deque(maxlen=capacity)
        self.rng       = np.random.default_rng(seed)

    def push_episode(self, transitions: list):
        if len(transitions) >= self._seq_len:
            self._episodes.append(transitions)

    def can_sample(self, batch_size: int) -> bool:
        return len(self._episodes) >= batch_size

    def sample(self, batch_size: int):
        L = self._seq_len
        eps = list(self._episodes)
        chosen = self.rng.integers(len(eps), size=batch_size)
        b_s, b_a, b_G, b_ns, b_gef, b_d, b_o = [], [], [], [], [], [], []
        for i in chosen:
            ep = eps[i]
            max_start = max(0, len(ep) - L)
            start = int(self.rng.integers(max_start + 1))
            window = ep[start : start + L]
            # Pad with last transition if shorter than L (shouldn't happen after can_sample)
            while len(window) < L:
                window.append(window[-1])
            s, a, G, ns, gef, d, o = zip(*window)
            b_s.append(s);  b_a.append(a);  b_G.append(G)
            b_ns.append(ns); b_gef.append(gef); b_d.append(d); b_o.append(o)
        return (
            np.array(b_s,   dtype=np.float32),    # (B, L, D)
            np.array(b_a,   dtype=np.int64),       # (B, L)
            np.array(b_G,   dtype=np.float32),     # (B, L)
            np.array(b_ns,  dtype=np.float32),     # (B, L, D)
            np.array(b_gef, dtype=np.float32),     # (B, L)
            np.array(b_d,   dtype=np.float32),     # (B, L)
            np.array(b_o,   dtype=np.int64),       # (B, L)
        )


def _n_step_returns(raw: list, n: int, gamma: float) -> list:
    """
    Convert raw episode transitions [(s,a,r,ns,done,omega), ...] into
    n-step return tuples [(s,a,G,ns_boot,geff,done,omega), ...].

    G = Σ_{k=0}^{n-1} γ^k r_{t+k};  ns_boot = s_{t+n};  geff = γ^n.
    If a done=True lands inside the n-step window, bootstrap is zero.
    """
    T = len(raw)
    out = []
    for t in range(T):
        G, geff = 0.0, 1.0
        terminated = False
        for k in range(n):
            if t + k >= T:
                break
            s_, a_, r_, ns_, done_, o_ = raw[t + k]
            G += geff * r_
            if done_:
                terminated = True
                geff = 0.0
                break
            geff *= gamma
        s0, a0, _, _, done0, o0 = raw[t]
        # bootstrap state: s_{t+n} if available, else last seen
        boot_idx  = min(t + n, T - 1)
        ns_boot   = raw[boot_idx][3]  # next_state at boot_idx
        ep_done   = terminated or (t + n >= T)
        out.append((s0, a0, G, ns_boot, geff, float(ep_done), o0))
    return out


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
            agent.reset_hidden()
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
    seq_buf = EpisodeSequenceBuffer(
        capacity=SEQ_BUF_CAPACITY, seq_len=SEQ_LEN, seed=seed + 2,
    )
    rng = np.random.default_rng(seed * 31 + 7919)

    best_score      = -float('inf')
    best_weights    = None
    best_per_preset = None
    best_episode    = -1
    env_steps       = 0
    n_grad_steps    = 0
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
        agent.reset_hidden()
        state = env.reset(spawn_offset_km=spawn_km)
        done = False
        raw_ep: list = []

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            raw_ep.append((state, action, reward, next_state, done, 0))
            state = next_state

            # Per-step training (Q's cadence): sample sequence windows from
            # previously completed episodes while this one unrolls.
            env_steps += 1
            if env_steps % TRAIN_EVERY_STEPS == 0 and seq_buf.can_sample(config.batch_size):
                if agent.train_batch_seq(seq_buf) is not None:
                    n_grad_steps += 1

        env.close()
        agent.decay_epsilon()

        # Convert to n-step returns and push to sequence buffer.
        seq_transitions = _n_step_returns(raw_ep, config.n_step, config.gamma)
        seq_buf.push_episode(seq_transitions)

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
                                'best_per_preset': best_per_preset, 'worker_id': worker_id,
                                'n_grad_steps': n_grad_steps,
                                'train_every_steps': TRAIN_EVERY_STEPS})
                )

            result_queue.put({
                'type':         'eval',
                'worker_id':    worker_id,
                'ep':           ep,
                'elapsed_s':    time.time() - start_ts,
                'tier':         tier['label'],
                'epsilon':      agent.epsilon,
                'is_best':      new_best,
                'n_grad_steps': n_grad_steps,
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
        'n_grad_steps':    n_grad_steps,
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

    c = BASE_CONFIG
    n_params = sum(p.numel() for p in QRAgent(c).policy_net.parameters())
    tee('═' * 78)
    tee('ABLATION T: plain GRU memory — realism env, 20-dim state, 3600 eps')
    tee('═' * 78)
    tee(f'Workers:     {N_WORKERS}')
    tee(f'Network:     {c.state_dim} → enc{c.hidden_sizes} → GRU-{c.gru_hidden} '
        f'→ {c.action_count}   ({n_params:,} params)')
    tee('Curriculum:  ' + '  →  '.join(
        f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM) + f'   total {TOTAL_EPS} eps/worker')
    tee(f'Sequence:    burn-in {c.seq_burn_in} + train {c.seq_train}  '
        f'batch {c.batch_size} windows  buffer {SEQ_BUF_CAPACITY} eps  '
        f'1 grad step / {TRAIN_EVERY_STEPS} env steps')
    tee(f'Gamma:       {c.gamma}  target_update_freq={c.target_update_freq}  (R\'s recipe)')
    tee(f'Shaping:     exponential  β=0.5  γ_shape=0.97  τ={SHAPING_TAU_KM:.0f} km (10R)')
    tee(f'Realism env: phase jitter + episode noise + amp jitter + domain-rand (all ON, train+eval)')
    tee(f'Features:    NO phase features of any kind — memory must infer phase')
    tee(f'Recovery spawn: {RECOVERY_SPAWN_PROB*100:.0f}% of ≥24h episodes  '
        f'range [{RECOVERY_SPAWN_MIN_KM:.0f}, {RECOVERY_SPAWN_MAX_KM:.0f}] km')
    tee(f'Eval:        greedy (ε suppressed), hidden reset per episode')
    tee(f'Change vs R: use_recurrent + GRU-64 + Q\'s sequence replay (no options). '
        f'Memory arm of R/S/T — not comparable to H–Q scores')
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
                f'  grads {msg["n_grad_steps"]}'
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
                f'  grads {msg["n_grad_steps"]}'
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
        'ablation':              'T_gru_memory',
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
        'phase_features':        None,   # memory arm — GRU must infer phase
        'gru_hidden':            BASE_CONFIG.gru_hidden,
        'use_options':           False,
        'train_every_steps':     TRAIN_EVERY_STEPS,
        'realism_env_flags':     ['wind_phase_jitter', 'wind_episode_noise',
                                  'wind_param_jitter', 'domain_rand'],
        'curriculum':            [{'label': t['label'], 'episodes': t['episodes']} for t in CURRICULUM],
        'recovery_spawn_prob':   RECOVERY_SPAWN_PROB,
        'recovery_spawn_min_km': RECOVERY_SPAWN_MIN_KM,
        'recovery_spawn_max_km': RECOVERY_SPAWN_MAX_KM,
        'eval_greedy':           True,
        'workers': [
            {'worker_id': r['worker_id'], 'best_score': r['best_score'],
             'best_episode': r['best_episode'], 'n_grad_steps': r['n_grad_steps']}
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

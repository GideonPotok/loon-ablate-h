"""
Ablation O — Option-Critic with GRU, stabilized (v2 server, 3600 episodes):

Question this answers:
  "Ablation M (same architecture) showed visibly purposeful multi-hour
   altitude commitment in replay GIFs, but its quantitative scores were both
   low (24.7% clean score vs L's 46.9%) and wildly seed-dependent (stdev
   9.9pp vs L's 2.6pp, with 7/10 workers never breaking 30%). Is the
   architecture sound but the training recipe unstable — and if so, does
   fixing the recipe (without touching curriculum, env flags, or the option
   count/GRU size) recover L-level reliability while keeping the temporal
   commitment behaviour?"

Diagnosis from Ablation M's post-mortem:
  M changed four things at once relative to L, two of which are confounds
  unrelated to option-critic itself:
    - learning_rate 1e-4 → 3e-4 ("GRU needs faster convergence" — backwards;
      BPTT through a 3-headed loss (critic + actor + termination) is *more*
      sensitive to LR, not less)
    - batch_size 64 → 32 (noisier gradient estimates, compounding the above)
    - episode-level training cadence instead of L's per-step cadence
    - the actual architecture change: GRU-64 + 4 options
  On top of that, qr_agent.py had no gradient clipping anywhere in the
  codebase, and the option-critic loss sums three terms (QR-Huber critic +
  policy-gradient actor + policy-gradient termination) through a shared GRU
  trunk in a single backward() — exactly the setup where unclipped gradients
  compound. The termination loss also used Harb et al. 2018's deliberation
  cost ξ=0.01, low enough to let options thrash (terminate almost every
  step), which both adds policy-gradient variance and undercuts the whole
  point of using options (no real temporal commitment if ω changes constantly).

Changes vs Ablation M (stabilization only — architecture is untouched):
  - learning_rate: 3e-4 → 1e-4   (revert to L's proven-stable value)
  - batch_size:    32   → 64     (revert to L's proven-stable value)
  - grad_clip_norm: None → 5.0   (new QRConfig field; clips policy_net grad
    norm before every optimizer.step(), across all three training paths)
  - oc_term_reg:   0.01 → 0.05   (higher deliberation cost → options must
    clear a higher bar to terminate → fewer thrashing/short-lived options)
  - target_update_freq stays at M's 30 (not a confound — slower target
    updates are a legitimate stabilizer for the option-critic bootstrap
    target and were never implicated in the instability)
  - Eval is greedy from the start (M's branch predated the eval ε-greedy
    leak fix; O is built after that fix, so this was never contaminated)
  - n_options=4, gru_hidden=64, curriculum, and all env flags (Fourier time
    features, τ=500 km exponential shaping, recovery spawn) are identical
    to M and L — unchanged, so any improvement is attributable to the
    stabilization changes above, not a different question being asked.

Option-Critic losses (single backward pass per gradient step, now clipped):
  Critic: QR-Huber on Q_ω(s,a) with target U = (1−β_ω)·Q_ω(s',a*) + β_ω·V_Ω*(s')
  Actor:  −log π_ω(a|s)·(Q(s,a,ω) − V(s,ω)) − ε_H·H(π_ω)
  Term:   β_ω(s)·(Q_Ω(s,ω) − V_Ω*(s) + ξ)   [ξ=0.05 — encourage longer options]

Usage:
    python ablate_o_train.py
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
    {'episodes':  200, 'duration_s': 3600 *  2, 'label':  '2h'},
    {'episodes': 1000, 'duration_s': 3600 *  6, 'label':  '6h'},
    {'episodes':  600, 'duration_s': 3600 * 12, 'label': '12h'},
    {'episodes':  600, 'duration_s': 3600 * 24, 'label': '24h'},
    {'episodes':  800, 'duration_s': 3600 * 48, 'label': '48h'},
    {'episodes':  400, 'duration_s': 3600 * 72, 'label': '72h'},
]
TOTAL_EPS       = sum(t['episodes'] for t in CURRICULUM)   # 3600
PRESETS         = ['tropical', 'strong-shear', 'calm']
N_WORKERS       = 10
EVAL_EVERY      = 300
EVAL_RUNS       = 3
EVAL_DURATION_S = 3600 * 72

# ── Recovery spawn (carried from J/K/L) ───────────────────────────────────────

RECOVERY_SPAWN_PROB    = 0.30
RECOVERY_SPAWN_MIN_KM  = 150.0
RECOVERY_SPAWN_MAX_KM  = 500.0
RECOVERY_SPAWN_MIN_DUR = 3600 * 24

BASE_CONFIG = QRConfig(
    state_dim         = 24,            # 20-dim + 4 Fourier time features (ablation L)
    hidden_sizes      = [128, 64],
    action_count      = 17,
    n_quantiles       = 1,             # plain DQN (no QR distribution)
    huber_kappa       = 1.0,
    learning_rate     = 1e-4,          # reverted from M's 3e-4 (confound; matches L)
    optimizer         = 'adam',
    gamma             = 0.97,
    epsilon_start     = 1.0,
    epsilon_end       = 0.03,
    epsilon_decay     = 0.9988,
    target_update_freq = 30,           # longer — GRU training is noisier
    replay_capacity   = 100_000,       # unused for seq replay but kept for compat
    batch_size        = 64,            # reverted from M's 32 (confound; matches L)
    n_step            = 3,
    per_alpha         = 0.6,
    per_beta0         = 0.4,
    per_beta_anneal   = 1e-4,
    cvar_alpha        = 1.0,
    train_batches_per_step = 0,        # not used (episode-level training instead)
    device            = 'cpu',
    use_reward_fix     = False,
    use_shaping        = False,
    use_expanded_state = False,
    use_recurrent      = True,         # GRU hidden state
    use_options        = True,         # option-critic
    n_options          = 4,
    gru_hidden         = 64,
    seq_burn_in        = 16,
    seq_train          = 16,
    oc_actor_weight    = 1.0,
    oc_term_weight     = 0.5,
    oc_entropy_reg     = 0.01,
    grad_clip_norm     = 5.0,          # new — was unset (off) in M; clips policy_net grads
    oc_term_reg        = 0.05,         # raised from M's 0.01 — discourage option thrashing
)

WEIGHTS_DIR    = Path(__file__).parent / 'weights'
LOG_PATH       = Path('/tmp/train_ablate_o.log')
WEIGHTS_PREFIX = 'dqn_ablate_o'

SEQ_LEN           = BASE_CONFIG.seq_burn_in + BASE_CONFIG.seq_train   # 32
SEQ_BUF_CAPACITY  = 500   # episodes; with 10 workers this ~5 recent episodes each


def _env_flags() -> dict:
    return {
        'use_reward_fix':     True,
        'use_shaping':        True,
        'use_expanded_state': False,
        'use_time_features':  True,    # Fourier features → 24-dim state
        'shaping_beta':       0.5,
        'shaping_gamma':      0.97,
        'terminal_twr_bonus': 50.0,
        'shaping_linear':     False,   # exponential shaping, τ=500 km
        'shaping_D_max':      500_000.0,
    }


# ── Sequence replay buffer ─────────────────────────────────────────────────────

class EpisodeSequenceBuffer:
    """
    Stores complete episodes and samples fixed-length windows for R2D2/OC training.

    Each episode is a list of 7-tuples:
        (state, action, n_step_return, bootstrap_state, eff_gamma, done, option)
    Sampling: pick a random eligible episode, then a random L-step window from it.
    Only episodes with >= L transitions are eligible.
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
                action = agent.select_action(state, greedy=True)
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
            omega  = agent.get_option() or 0
            next_state, reward, done, _ = env.step(action)
            raw_ep.append((state, action, reward, next_state, done, omega))
            state = next_state

        env.close()
        agent.decay_epsilon()

        # Convert to n-step returns and push to sequence buffer.
        seq_transitions = _n_step_returns(raw_ep, config.n_step, config.gamma)
        seq_buf.push_episode(seq_transitions)

        # Train: roughly one gradient step per seq_train environment steps.
        if seq_buf.can_sample(config.batch_size):
            n_updates = max(1, len(seq_transitions) // config.seq_train)
            for _ in range(n_updates):
                agent.train_batch_options(seq_buf)

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

    c = BASE_CONFIG
    n_params = sum(p.numel() for p in QRAgent(c).policy_net.parameters())
    tee('═' * 78)
    tee('ABLATION O: option-critic + GRU, stabilized — v2 server, 24-dim state, 3600 eps')
    tee('═' * 78)
    tee(f'Workers:     {N_WORKERS}')
    tee(f'Network:     {c.state_dim} → enc{c.hidden_sizes} → GRU-{c.gru_hidden} '
        f'→ {c.n_options}×({c.action_count} Q + {c.action_count} π + 1 β)  ({n_params:,} params)')
    tee('Curriculum:  ' + '  →  '.join(
        f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM) + f'   total {TOTAL_EPS} eps/worker')
    tee(f'Options:     {c.n_options}  GRU hidden: {c.gru_hidden}  '
        f'burn-in: {c.seq_burn_in}  train window: {c.seq_train}')
    tee(f'OC weights:  actor={c.oc_actor_weight}  term={c.oc_term_weight}  '
        f'entropy={c.oc_entropy_reg}  term_reg={c.oc_term_reg}')
    tee(f'Shaping:     exponential  β=0.5  γ=0.97  τ=500 km')
    tee(f'Recovery spawn: {RECOVERY_SPAWN_PROB*100:.0f}% of ≥24h episodes  '
        f'[{RECOVERY_SPAWN_MIN_KM:.0f}, {RECOVERY_SPAWN_MAX_KM:.0f}] km')
    tee(f'Change vs M: lr={c.learning_rate:.0e} (was 3e-4), batch={c.batch_size} (was 32), '
        f'grad_clip={c.grad_clip_norm}, term_reg={c.oc_term_reg} (was 0.01)')
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
        'ablation':              'O_stable_option_critic',
        'winner_worker':         wid,
        'best_score':            winner['best_score'],
        'best_episode':          winner['best_episode'],
        'best_per_preset':       winner['best_per_preset'],
        'wall_time_s':           time.time() - launch_ts,
        'n_options':             BASE_CONFIG.n_options,
        'gru_hidden':            BASE_CONFIG.gru_hidden,
        'shaping':               'exponential',
        'shaping_tau_km':        500,
        'recovery_spawn_prob':   RECOVERY_SPAWN_PROB,
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

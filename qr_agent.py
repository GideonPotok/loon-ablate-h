"""
QR-DQN agent for the stratospheric balloon station-keeping task.

Matches the JS implementation in tactical/js/qr_agent.js but uses PyTorch
for correct mini-batch gradient descent (single optimizer.step() per batch,
not per-sample as in the JS version).

References:
  Dabney et al. 2018 — "Distributional RL with Quantile Regression"
  Bellemare et al. 2020 — "Autonomous Navigation of Stratospheric Balloons" (Loon)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass, field
from typing import Optional

from replay_buffer import PrioritizedReplayBuffer


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class QRConfig:
    # Network
    state_dim:         int   = 20
    hidden_sizes:      list  = field(default_factory=lambda: [128, 64])
    action_count:      int   = 17        # targetAlt17
    n_quantiles:       int   = 51
    huber_kappa:       float = 1.0

    # Optimiser
    learning_rate:     float = 1e-4      # lower than Phase 1 — 51× larger output head
    optimizer:         str   = 'adam'

    # RL
    gamma:             float = 0.97
    epsilon_start:     float = 1.0
    epsilon_end:       float = 0.03
    epsilon_decay:     float = 0.9988    # matches JS: slower for 2800-ep schedule
    target_update_freq: int  = 15        # hard copy every N episodes

    # Replay / n-step
    replay_capacity:   int   = 100_000
    batch_size:        int   = 64
    n_step:            int   = 3
    per_alpha:         float = 0.6
    per_beta0:         float = 0.4
    per_beta_anneal:   float = 1e-4

    # Action selection
    cvar_alpha:        float = 0.25      # Loon's value — risk-averse lower tail

    # Misc
    seed:              int   = 42
    device:            str   = 'cpu'
    train_batches_per_step: int = 2

    # ── Phase 2-v2 feature flags ─────────────────────────────────────────────
    # All default to False so existing training behaviour is unchanged.
    # Enable in BASE_CONFIG (train_phase2.py) for the bundled v2 experiment.
    use_reward_fix:     bool  = False   # TWR-aligned per-step + terminal bonus
    use_shaping:        bool  = False   # potential-based reward shaping (Ng 1999)
    use_expanded_state: bool  = False   # ~52-d state: 10 wind alts + forecast traj + time
    use_recurrent:      bool  = False   # GRU(128) over state encoder
    use_options:        bool  = False   # Option-Critic with K options
    # Sub-knobs used only when the corresponding flag is on:
    shaping_beta:       float = 0.5     # Φ(s) = β·exp(-d/2R)
    terminal_twr_bonus: float = 50.0    # +α·twr50 at episode end
    n_options:          int   = 4
    gru_hidden:         int   = 128
    seq_burn_in:        int   = 16      # replay sequence: burn-in steps
    seq_train:          int   = 16      # replay sequence: training steps
    # Option-Critic loss weights (Bacon et al. 2017, eq 5/7/8):
    oc_actor_weight:    float = 1.0     # weight on log π_ω · (Q − V) actor loss
    oc_term_weight:     float = 0.5     # weight on β_ω · A_Ω termination loss
    oc_entropy_reg:     float = 0.01    # entropy regularisation on π_ω
    oc_term_reg:        float = 0.01    # deliberation cost ξ added to advantage


# ── Network ───────────────────────────────────────────────────────────────────

class QRNetwork(nn.Module):
    """
    Feedforward QR-DQN head, optionally augmented with a GRU recurrent core.

    Feedforward mode (use_recurrent=False, default):
        state_dim → hidden_sizes (ReLU) → (A × N)
        forward(x: (B, D)) → (B, A, N)

    Recurrent mode (use_recurrent=True):
        encoder: state_dim → hidden_sizes (ReLU)
        recurrent: GRU(input=hidden_sizes[-1], hidden=gru_hidden)
        head: gru_hidden → (A × N)
        forward(x: (B, T, D), h0=(1, B, H)) → ((B, T, A, N), h_T)
        Also supports single-step inference x:(B, D) which is treated as T=1.
    """

    def __init__(self, state_dim: int, hidden_sizes: list[int],
                 action_count: int, n_quantiles: int,
                 use_recurrent: bool = False, gru_hidden: int = 128,
                 use_options: bool = False, n_options: int = 4):
        super().__init__()
        self.action_count = action_count
        self.n_quantiles = n_quantiles
        self.use_recurrent = use_recurrent
        self.gru_hidden = gru_hidden
        self.use_options = use_options
        self.n_options = n_options

        if use_options and not use_recurrent:
            raise ValueError('use_options requires use_recurrent=True in this build')

        if not use_recurrent:
            layers: list[nn.Module] = []
            in_dim = state_dim
            for h in hidden_sizes:
                layers += [nn.Linear(in_dim, h), nn.ReLU()]
                in_dim = h
            layers.append(nn.Linear(in_dim, action_count * n_quantiles))
            self.net = nn.Sequential(*layers)
            return

        enc_layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_sizes:
            enc_layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        self.encoder = nn.Sequential(*enc_layers)
        self.gru = nn.GRU(input_size=in_dim, hidden_size=gru_hidden, batch_first=True)

        if use_options:
            K, A, N = n_options, action_count, n_quantiles
            self.q_head    = nn.Linear(gru_hidden, K * A * N)
            self.pi_head   = nn.Linear(gru_hidden, K * A)
            self.beta_head = nn.Linear(gru_hidden, K)
        else:
            self.head = nn.Linear(gru_hidden, action_count * n_quantiles)

    def forward(self, x: torch.Tensor, h0: Optional[torch.Tensor] = None):
        """
        Feedforward: x:(B, D) → (B, A, N).

        Recurrent (no options):
            x:(B, T, D) → ((B, T, A, N), h_T)
            x:(B, D)    → ((B, A, N), h_T)  (single-step inference)

        Recurrent + options:
            returns ({'q': (B, T, K, A, N) or (B, K, A, N),
                      'pi_logits': (B, T, K, A) or (B, K, A),
                      'beta_logits': (B, T, K) or (B, K)}, h_T)
        """
        if not self.use_recurrent:
            return self.net(x).view(x.shape[0], self.action_count, self.n_quantiles)

        squeeze_time = (x.dim() == 2)
        if squeeze_time:
            x = x.unsqueeze(1)                         # (B, 1, D)
        B, T, _ = x.shape
        z = self.encoder(x.reshape(B * T, -1)).view(B, T, -1)
        out, h_T = self.gru(z, h0)                     # out: (B, T, H)

        if self.use_options:
            K, A, N = self.n_options, self.action_count, self.n_quantiles
            q   = self.q_head(out).view(B, T, K, A, N)
            pi  = self.pi_head(out).view(B, T, K, A)
            bet = self.beta_head(out).view(B, T, K)
            if squeeze_time:
                out_dict = {'q': q.squeeze(1), 'pi_logits': pi.squeeze(1),
                            'beta_logits': bet.squeeze(1)}
            else:
                out_dict = {'q': q, 'pi_logits': pi, 'beta_logits': bet}
            return out_dict, h_T

        q = self.head(out).view(B, T, self.action_count, self.n_quantiles)
        if squeeze_time:
            return q.squeeze(1), h_T
        return q, h_T


# ── Loss ──────────────────────────────────────────────────────────────────────

def quantile_huber_loss(
    pred:  torch.Tensor,   # (B, N) predicted quantiles for taken action
    target: torch.Tensor,  # (B, N) Bellman targets z' = r + γ·Z_target(s',a*)
    taus:  torch.Tensor,   # (N,)  quantile midpoints
    kappa: float = 1.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Eq. 10 from Dabney et al. 2018.

    delta[b,i,j] = target[b,j] - pred[b,i]   (target minus prediction)
    loss = (1/N²) Σ_i Σ_j |τ_i - 1{δ<0}| · L_κ(δ)
    """
    N = pred.shape[1]
    # (B, N, 1) vs (B, 1, N) → (B, N, N)
    delta = target.unsqueeze(1) - pred.unsqueeze(2)

    huber = torch.where(
        delta.abs() <= kappa,
        0.5 * delta.pow(2),
        kappa * (delta.abs() - 0.5 * kappa),
    )
    # |τ_i - 1{δ_ij < 0}|, taus broadcast over batch and j dim
    tau_weight = (taus.view(1, N, 1) - (delta.detach() < 0).float()).abs()

    # mean over j (dim=2), mean over i (dim=1) → (B,); then reduce over batch
    per_sample = (tau_weight * huber).mean(dim=2).mean(dim=1)   # (B,)

    if reduction == 'none':
        return per_sample
    return per_sample.mean()


# ── Agent ─────────────────────────────────────────────────────────────────────

class QRAgent:
    def __init__(self, config: QRConfig):
        self.config = config
        c = config
        torch.manual_seed(c.seed)
        self.rng = np.random.default_rng(c.seed)
        self.device = torch.device(c.device)

        self.policy_net = QRNetwork(
            c.state_dim, c.hidden_sizes, c.action_count, c.n_quantiles,
            use_recurrent=c.use_recurrent, gru_hidden=c.gru_hidden,
            use_options=c.use_options, n_options=c.n_options,
        ).to(self.device)
        self.target_net = QRNetwork(
            c.state_dim, c.hidden_sizes, c.action_count, c.n_quantiles,
            use_recurrent=c.use_recurrent, gru_hidden=c.gru_hidden,
            use_options=c.use_options, n_options=c.n_options,
        ).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Per-env hidden state for recurrent inference; reset at episode start.
        self._inference_hidden: Optional[torch.Tensor] = None
        # Current option index (set on first select_action of each episode).
        self._current_option: Optional[int] = None

        if c.optimizer == 'adam':
            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=c.learning_rate)
        else:
            self.optimizer = optim.SGD(self.policy_net.parameters(), lr=c.learning_rate)

        # τ_i = (i + 0.5) / N
        self.taus = torch.tensor(
            [(i + 0.5) / c.n_quantiles for i in range(c.n_quantiles)],
            dtype=torch.float32, device=self.device,
        )

        self.epsilon = c.epsilon_start
        self.episode_count = 0
        self.losses: list[float] = []

    # ── Action selection ──────────────────────────────────────────────────────

    def reset_hidden(self) -> None:
        """Call at the start of each episode in recurrent mode."""
        if not self.config.use_recurrent:
            return
        self._inference_hidden = torch.zeros(
            1, 1, self.config.gru_hidden, dtype=torch.float32, device=self.device,
        )
        self._current_option = None

    def get_option(self) -> Optional[int]:
        """Return current option index (or None if options disabled / not yet picked)."""
        return self._current_option

    def _cvar_of_q(self, q: torch.Tensor) -> torch.Tensor:
        """CVaR_α reduction over the last (quantile) dim. q:(..., N) → (...)."""
        c = self.config
        if c.cvar_alpha < 1.0:
            k = max(1, int(c.cvar_alpha * c.n_quantiles))
            return q.sort(dim=-1).values[..., :k].mean(dim=-1)
        return q.mean(dim=-1)

    def _q_values(self, state: np.ndarray) -> torch.Tensor:
        """(A,) Q-values (mean or CVaR) for greedy action selection."""
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.config.use_recurrent:
                q_step, h_T = self.policy_net(x, self._inference_hidden)  # q_step: (1, A, N)
                self._inference_hidden = h_T
                q = q_step.squeeze(0)
            else:
                q = self.policy_net(x).squeeze(0)   # (A, N)
        return self._cvar_of_q(q)

    def _select_action_options(self, state: np.ndarray) -> int:
        """Option-Critic action selection: terminate→reselect ω, then sample a∼π_ω."""
        c = self.config
        x = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            out, h_T = self.policy_net(x, self._inference_hidden)
        self._inference_hidden = h_T
        q  = out['q'].squeeze(0)                     # (K, A, N)
        pi = out['pi_logits'].squeeze(0)             # (K, A)
        bet = torch.sigmoid(out['beta_logits'].squeeze(0))  # (K,)

        # Per-option value V_Ω(s,ω) = Σ_a π_ω(a|s) · CVaR(q[ω,a,:])
        cvar_qa = self._cvar_of_q(q)                 # (K, A)
        pi_prob = torch.softmax(pi, dim=-1)          # (K, A)
        v_omega = (pi_prob * cvar_qa).sum(dim=-1)    # (K,)

        if self._current_option is None:
            # Episode start: pick greedy option (with ε-exploration over options).
            if self.rng.random() < self.epsilon:
                self._current_option = int(self.rng.integers(c.n_options))
            else:
                self._current_option = int(v_omega.argmax().item())
        else:
            # Termination probability of the incumbent option at this state.
            b = float(bet[self._current_option].item())
            if self.rng.random() < b:
                # Terminated → pick new option (greedy, with ε-exploration).
                if self.rng.random() < self.epsilon:
                    self._current_option = int(self.rng.integers(c.n_options))
                else:
                    self._current_option = int(v_omega.argmax().item())

        # Sample action from π_ω.
        probs = pi_prob[self._current_option].cpu().numpy().astype(np.float64)
        probs /= probs.sum()
        return int(self.rng.choice(c.action_count, p=probs))

    def select_action(self, state: np.ndarray) -> int:
        if self.config.use_recurrent and self._inference_hidden is None:
            self.reset_hidden()
        if self.config.use_options:
            return self._select_action_options(state)
        if self.rng.random() < self.epsilon:
            # Even on a random action we must advance the recurrent state.
            if self.config.use_recurrent:
                _ = self._q_values(state)
            return int(self.rng.integers(self.config.action_count))
        return int(self._q_values(state).argmax().item())

    # ── Training ──────────────────────────────────────────────────────────────

    def train_batch(self, replay_buffer: PrioritizedReplayBuffer) -> Optional[float]:
        c = self.config
        if len(replay_buffer) < c.batch_size:
            return None

        (states, actions, rewards, next_states, dones, eff_gammas), indices, is_weights = \
            replay_buffer.sample(c.batch_size)

        dev = self.device
        states_t      = torch.tensor(states,      dtype=torch.float32, device=dev)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=dev)
        actions_t     = torch.tensor(actions,     dtype=torch.long,    device=dev)
        rewards_t     = torch.tensor(rewards,     dtype=torch.float32, device=dev)
        dones_t       = torch.tensor(dones,       dtype=torch.float32, device=dev)
        eff_gammas_t  = torch.tensor(eff_gammas,  dtype=torch.float32, device=dev)
        is_weights_t  = torch.tensor(is_weights,  dtype=torch.float32, device=dev)

        # Predicted quantiles for taken actions: (B, N)
        all_pred = self.policy_net(states_t)                           # (B, A, N)
        idx      = torch.arange(c.batch_size, device=dev)
        pred_q   = all_pred[idx, actions_t]                            # (B, N)

        # Double-DQN: policy net selects a*, target net evaluates it
        with torch.no_grad():
            policy_next = self.policy_net(next_states_t)               # (B, A, N)
            a_star      = policy_next.mean(dim=-1).argmax(dim=1)       # (B,)
            target_next = self.target_net(next_states_t)               # (B, A, N)
            z_target    = target_next[idx, a_star]                     # (B, N)
            # Bellman backup: r + γ^n · Z_target(s',a*), masked at terminal
            z_prime = (
                rewards_t.unsqueeze(1)
                + eff_gammas_t.unsqueeze(1) * (1 - dones_t.unsqueeze(1)) * z_target
            )                                                           # (B, N)

        # IS-weighted quantile Huber loss — single optimizer step over full batch
        per_sample_loss = quantile_huber_loss(pred_q, z_prime, self.taus,
                                              c.huber_kappa, reduction='none')  # (B,)
        loss = (is_weights_t * per_sample_loss).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        replay_buffer.update_priorities(indices, per_sample_loss.detach().cpu().numpy())
        replay_buffer.anneal_beta(1.0, c.per_beta_anneal)

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def train_batch_seq(self, seq_buffer) -> Optional[float]:
        """
        R2D2-style training step for the recurrent (GRU) path.

        Samples (B, L) windows from `seq_buffer` where L = burn_in + train_len.
        Burns the first `burn_in` steps without gradient to warm up the hidden
        state, then computes the QR-Huber loss on the remaining `train_len`
        steps. Hidden states at window start are zero-initialised (R2D2 zero
        init + burn-in).

        Returns the mean loss, or None if the buffer is not yet ready.
        """
        c = self.config
        burn = c.seq_burn_in
        tr_n = c.seq_train
        L = burn + tr_n
        if not seq_buffer.can_sample(c.batch_size):
            return None

        states, actions, returns_, bootstrap, eff_gamma, done_mask, _options = \
            seq_buffer.sample(c.batch_size)
        dev = self.device
        s_t  = torch.tensor(states,    dtype=torch.float32, device=dev)   # (B, L, D)
        a_t  = torch.tensor(actions,   dtype=torch.long,    device=dev)   # (B, L)
        G_t  = torch.tensor(returns_,  dtype=torch.float32, device=dev)   # (B, L)
        ns_t = torch.tensor(bootstrap, dtype=torch.float32, device=dev)   # (B, L, D)
        gef  = torch.tensor(eff_gamma, dtype=torch.float32, device=dev)   # (B, L)
        d_t  = torch.tensor(done_mask, dtype=torch.float32, device=dev)   # (B, L)

        B = c.batch_size
        H = c.gru_hidden
        h0 = torch.zeros(1, B, H, dtype=torch.float32, device=dev)

        # Burn-in: no gradients, advance hidden state through first `burn` steps.
        if burn > 0:
            with torch.no_grad():
                _, h_policy = self.policy_net(s_t[:, :burn], h0)
                _, h_target = self.target_net(s_t[:, :burn], h0)
        else:
            h_policy = h0
            h_target = h0

        # Training portion: policy net (with grad), target net (no grad).
        train_states    = s_t[:, burn:]                             # (B, tr_n, D)
        train_actions   = a_t[:, burn:]                             # (B, tr_n)
        train_returns   = G_t[:, burn:]                             # (B, tr_n)
        train_bootstrap = ns_t[:, burn:]                            # (B, tr_n, D)
        train_gef       = gef[:, burn:]                             # (B, tr_n)
        train_done      = d_t[:, burn:]                             # (B, tr_n)

        # Predicted quantiles for taken actions over the training portion.
        all_pred, _ = self.policy_net(train_states, h_policy)        # (B, tr_n, A, N)
        # Gather along action dim
        action_idx = train_actions.unsqueeze(-1).unsqueeze(-1)       # (B, tr_n, 1, 1)
        action_idx = action_idx.expand(-1, -1, 1, c.n_quantiles)     # (B, tr_n, 1, N)
        pred_q = all_pred.gather(2, action_idx).squeeze(2)           # (B, tr_n, N)

        # Bellman target via Double-DQN.
        with torch.no_grad():
            policy_next, _ = self.policy_net(train_bootstrap, h_policy)  # (B, tr_n, A, N)
            a_star = policy_next.mean(dim=-1).argmax(dim=-1)              # (B, tr_n)
            target_next, _ = self.target_net(train_bootstrap, h_target)  # (B, tr_n, A, N)
            a_idx_t = a_star.unsqueeze(-1).unsqueeze(-1).expand(
                -1, -1, 1, c.n_quantiles,
            )
            z_target = target_next.gather(2, a_idx_t).squeeze(2)         # (B, tr_n, N)
            z_prime = (
                train_returns.unsqueeze(-1)
                + train_gef.unsqueeze(-1)
                * (1.0 - train_done.unsqueeze(-1))
                * z_target
            )                                                            # (B, tr_n, N)

        # Quantile Huber loss, averaged over (B × tr_n).
        pred_flat   = pred_q.reshape(B * tr_n, c.n_quantiles)
        target_flat = z_prime.reshape(B * tr_n, c.n_quantiles)
        loss = quantile_huber_loss(pred_flat, target_flat, self.taus, c.huber_kappa,
                                   reduction='mean')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def train_batch_options(self, seq_buffer) -> Optional[float]:
        """
        Option-Critic training step (Bacon et al. 2017) combined with
        QR-DQN distributional critic and R2D2-style burn-in.

        Losses (all summed into a single backward pass):
          - Critic (Q):  quantile-Huber on Q(s,a,ω) with target
                         U(s',ω) = (1−β_ω(s'))·Q(s',a*_ω,ω)
                                 +     β_ω(s') ·max_ω' Q(s',a*_ω',ω')
                         where a*_ω = argmax_a π_ω(a|s').
          - Actor (π):   −log π_ω(a|s)·(Q(s,a,ω) − V(s,ω)).detach()  − ent·H(π_ω)
          - Term  (β):    β_ω(s_t)·(Q_Ω(s_t,ω_{t−1}) − V_Ω(s_t) + ξ).detach()
        """
        c = self.config
        burn = c.seq_burn_in
        tr_n = c.seq_train
        L = burn + tr_n
        if not seq_buffer.can_sample(c.batch_size):
            return None

        states, actions, returns_, bootstrap, eff_gamma, done_mask, options = \
            seq_buffer.sample(c.batch_size)

        dev = self.device
        s_t  = torch.tensor(states,    dtype=torch.float32, device=dev)
        a_t  = torch.tensor(actions,   dtype=torch.long,    device=dev)
        G_t  = torch.tensor(returns_,  dtype=torch.float32, device=dev)
        ns_t = torch.tensor(bootstrap, dtype=torch.float32, device=dev)
        gef  = torch.tensor(eff_gamma, dtype=torch.float32, device=dev)
        d_t  = torch.tensor(done_mask, dtype=torch.float32, device=dev)
        o_t  = torch.tensor(options,   dtype=torch.long,    device=dev)
        # Clamp any -1 option (transitions logged before option-tracking) to 0.
        o_t = o_t.clamp(min=0)

        B, K, A, N = c.batch_size, c.n_options, c.action_count, c.n_quantiles
        H = c.gru_hidden
        h0 = torch.zeros(1, B, H, dtype=torch.float32, device=dev)

        # Burn-in (no grad)
        if burn > 0:
            with torch.no_grad():
                _, h_policy = self.policy_net(s_t[:, :burn], h0)
                _, h_target = self.target_net(s_t[:, :burn], h0)
        else:
            h_policy = h0
            h_target = h0

        train_s   = s_t[:, burn:]                                    # (B, T, D)
        train_a   = a_t[:, burn:]                                    # (B, T)
        train_G   = G_t[:, burn:]                                    # (B, T)
        train_ns  = ns_t[:, burn:]                                   # (B, T, D)
        train_gef = gef[:, burn:]                                    # (B, T)
        train_d   = d_t[:, burn:]                                    # (B, T)
        train_o   = o_t[:, burn:]                                    # (B, T)
        T = tr_n

        # Forward policy (with grad) on training states.
        out_pol, _ = self.policy_net(train_s, h_policy)              # dict
        q_pol      = out_pol['q']                                    # (B, T, K, A, N)
        pi_logits  = out_pol['pi_logits']                            # (B, T, K, A)
        beta_logits = out_pol['beta_logits']                         # (B, T, K)
        beta_prob   = torch.sigmoid(beta_logits)                     # (B, T, K)
        pi_prob     = torch.softmax(pi_logits, dim=-1)               # (B, T, K, A)

        # Bootstrap forward through target + policy (no grad).
        with torch.no_grad():
            out_tgt, _ = self.target_net(train_ns, h_target)
            q_tgt      = out_tgt['q']                                # (B, T, K, A, N)
            beta_tgt   = torch.sigmoid(out_tgt['beta_logits'])       # (B, T, K)

            out_pol_ns, _ = self.policy_net(train_ns, h_policy)
            pi_logits_ns  = out_pol_ns['pi_logits']                  # (B, T, K, A)
            a_star = pi_logits_ns.argmax(dim=-1)                     # (B, T, K)

        # Gather Q(s,a,ω) at taken (a, ω).
        # q_pol: (B, T, K, A, N) → select on K dim (current option) then on A.
        opt_idx = train_o.view(B, T, 1, 1, 1).expand(-1, -1, 1, A, N)
        q_pol_omega = q_pol.gather(2, opt_idx).squeeze(2)             # (B, T, A, N)
        act_idx = train_a.view(B, T, 1, 1).expand(-1, -1, 1, N)
        pred_q = q_pol_omega.gather(2, act_idx).squeeze(2)            # (B, T, N)

        # Target distribution:
        #   U(s',ω) = (1−β_ω(s'))·Q(s',a*_ω,ω) + β_ω(s')·max_ω' Q(s',a*_ω',ω')
        with torch.no_grad():
            # Q_target(s', a*_ω, ω): gather along A then K
            a_idx_t = a_star.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 1, N)
            q_at_astar = q_tgt.gather(3, a_idx_t).squeeze(3)          # (B, T, K, N)
            # Continuation distribution under incumbent option ω:
            opt_idx_kn = train_o.view(B, T, 1, 1).expand(-1, -1, 1, N)
            q_cont = q_at_astar.gather(2, opt_idx_kn).squeeze(2)      # (B, T, N)
            # Switch distribution: option chosen by V_Ω* = argmax (q_at_astar mean over N).
            best_omega = q_at_astar.mean(dim=-1).argmax(dim=-1)       # (B, T)
            best_idx = best_omega.view(B, T, 1, 1).expand(-1, -1, 1, N)
            q_switch = q_at_astar.gather(2, best_idx).squeeze(2)      # (B, T, N)
            beta_omega = beta_tgt.gather(2, train_o.unsqueeze(-1)).squeeze(-1)  # (B, T)
            u_target = (
                (1.0 - beta_omega).unsqueeze(-1) * q_cont
                + beta_omega.unsqueeze(-1) * q_switch
            )                                                          # (B, T, N)
            z_prime = (
                train_G.unsqueeze(-1)
                + train_gef.unsqueeze(-1)
                * (1.0 - train_d.unsqueeze(-1))
                * u_target
            )

        # ── Critic loss (QR-Huber) ────────────────────────────────────────────
        pred_flat   = pred_q.reshape(B * T, N)
        target_flat = z_prime.reshape(B * T, N)
        critic_loss = quantile_huber_loss(pred_flat, target_flat, self.taus,
                                          c.huber_kappa, reduction='mean')

        # ── Actor loss (Bacon et al. eq 6) ────────────────────────────────────
        # Use CVaR mean over quantiles as the scalar Q-value used in the
        # advantage A(s,a,ω) = Q(s,a,ω) − V(s,ω). Detach Q (critic provides it).
        with torch.no_grad():
            q_scalar = self._cvar_of_q(q_pol_omega)                   # (B, T, A)
            v_scalar = (pi_prob.gather(2, train_o.view(B, T, 1, 1)
                                              .expand(-1, -1, 1, A))
                              .squeeze(2)
                        * q_scalar).sum(dim=-1, keepdim=True)         # (B, T, 1)
            advantage = q_scalar - v_scalar                           # (B, T, A)

        log_pi = torch.log_softmax(pi_logits, dim=-1)                  # (B, T, K, A)
        log_pi_omega = log_pi.gather(2, train_o.view(B, T, 1, 1)
                                                  .expand(-1, -1, 1, A)).squeeze(2)
        log_pi_act = log_pi_omega.gather(2, train_a.unsqueeze(-1)).squeeze(-1)  # (B, T)
        adv_act = advantage.gather(2, train_a.unsqueeze(-1)).squeeze(-1)        # (B, T)
        actor_loss = -(log_pi_act * adv_act).mean()

        # Entropy regularisation on π_ω (over current option).
        pi_prob_omega = pi_prob.gather(2, train_o.view(B, T, 1, 1)
                                                .expand(-1, -1, 1, A)).squeeze(2)
        entropy = -(pi_prob_omega * log_pi_omega).sum(dim=-1).mean()
        actor_loss = actor_loss - c.oc_entropy_reg * entropy

        # ── Termination loss (Bacon et al. eq 7) ──────────────────────────────
        # β_ω(s_t) · (Q_Ω(s_t, ω) − V_Ω(s_t) + ξ)
        # Q_Ω(s,ω) = Σ_a π_ω(a|s)·CVaR(q_pol[ω,a,:])
        with torch.no_grad():
            cvar_all = self._cvar_of_q(q_pol)                          # (B, T, K, A)
            v_omega_all = (torch.softmax(pi_logits, dim=-1) * cvar_all).sum(dim=-1)
            #     (B, T, K)
            v_max = v_omega_all.max(dim=-1).values                     # (B, T)
            q_omega = v_omega_all.gather(2, train_o.unsqueeze(-1)).squeeze(-1)
            term_adv = q_omega - v_max + c.oc_term_reg                # (B, T)
        beta_now = beta_prob.gather(2, train_o.unsqueeze(-1)).squeeze(-1)  # (B, T)
        term_loss = (beta_now * term_adv).mean()

        loss = critic_loss + c.oc_actor_weight * actor_loss + c.oc_term_weight * term_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = float(loss.item())
        self.losses.append(loss_val)
        return loss_val

    def decay_epsilon(self):
        self.epsilon = max(self.config.epsilon_end,
                           self.epsilon * self.config.epsilon_decay)
        self.episode_count += 1
        if self.episode_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # ── Serialisation ─────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        c = self.config
        return {
            'agent_type':   'qr-dqn',
            'policy_net':   self.policy_net.state_dict(),
            'optimizer':    self.optimizer.state_dict(),
            'epsilon':      self.epsilon,
            'episode_count': self.episode_count,
            'config': {
                'state_dim':         c.state_dim,
                'hidden_sizes':      c.hidden_sizes,
                'action_count':      c.action_count,
                'n_quantiles':       c.n_quantiles,
                'learning_rate':     c.learning_rate,
                'gamma':             c.gamma,
                'n_step':            c.n_step,
                'cvar_alpha':        c.cvar_alpha,
            },
        }

    def load_state_dict(self, d: dict):
        self.policy_net.load_state_dict(d['policy_net'])
        self.target_net.load_state_dict(d['policy_net'])  # sync target
        self.optimizer.load_state_dict(d['optimizer'])
        self.epsilon       = d['epsilon']
        self.episode_count = d['episode_count']

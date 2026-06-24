"""
Prioritized Experience Replay with n-step return accumulation.

SumTree gives O(log N) priority updates and O(log N) stratified sampling.
NStepAccumulator wraps the PER buffer: it holds the last n transitions and
emits one n-step transition each step (flushing at episode end).
"""
import numpy as np
from collections import deque
from typing import Optional


class SumTree:
    """Binary heap where each leaf holds a priority; internal nodes are sums."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list = [None] * capacity
        self.ptr = 0
        self.size = 0

    def _propagate(self, idx: int):
        while idx > 1:
            idx >>= 1
            self.tree[idx] = self.tree[2 * idx] + self.tree[2 * idx + 1]

    def update(self, data_idx: int, priority: float):
        tree_idx = data_idx + self.capacity
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)

    def add(self, priority: float, data):
        idx = self.ptr
        self.data[idx] = data
        self.update(idx, priority)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, value: float) -> tuple[int, float, object]:
        """Walk tree to find leaf whose prefix sum >= value."""
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity
        return data_idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[1])


class PrioritizedReplayBuffer:
    """
    PER buffer (Schaul et al. 2016).

    Stored transition tuple: (state, action, reward, next_state, done, eff_gamma)
    where eff_gamma = γ^k for k-step returns (0 if terminal before k steps).
    """

    def __init__(self, capacity: int, alpha: float = 0.6, beta0: float = 0.4,
                 seed: int = 42):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta0
        self._eps = 1e-6
        self._max_priority = 1.0
        self.rng = np.random.default_rng(seed)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool, eff_gamma: float):
        self.tree.add(
            self._max_priority ** self.alpha,
            (state, action, reward, next_state, done, eff_gamma),
        )

    def sample(self, batch_size: int):
        total = self.tree.total
        segment = total / batch_size
        indices, priorities = [], []

        for i in range(batch_size):
            v = self.rng.uniform(segment * i, segment * (i + 1))
            idx, p, _ = self.tree.get(v)
            indices.append(idx)
            priorities.append(p)

        probs = np.array(priorities, dtype=np.float64) / total
        is_weights = (self.tree.size * probs) ** (-self.beta)
        is_weights = (is_weights / is_weights.max()).astype(np.float32)

        batch = [self.tree.data[i] for i in indices]
        states, actions, rewards, next_states, dones, eff_gammas = zip(*batch)

        arrays = (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
            np.array(eff_gammas,  dtype=np.float32),
        )
        return arrays, indices, is_weights

    def update_priorities(self, indices: list[int], td_errors: np.ndarray):
        for idx, err in zip(indices, td_errors):
            p = (float(abs(err)) + self._eps) ** self.alpha
            self.tree.update(idx, p)
            self._max_priority = max(self._max_priority, p)

    def anneal_beta(self, final: float = 1.0, rate: float = 1e-4):
        self.beta = min(final, self.beta + rate)

    def __len__(self) -> int:
        return self.tree.size


class NStepAccumulator:
    """
    Wraps a PER buffer, emitting n-step transitions.

    Usage per episode step:
        acc.push(state, action, reward, next_state, done)
        acc.flush_to_buffer()          # emits ready transitions to per_buf
    At episode end:
        acc.drain_to_buffer(terminal_state)
    """

    def __init__(self, n: int, gamma: float, per_buffer: PrioritizedReplayBuffer):
        self.n = n
        self.gamma = gamma
        self.buf = per_buffer
        self._deque: deque = deque()

    def push(self, state, action, reward, next_state, done):
        self._deque.append((state, action, reward, next_state, done))

    def _make_transition(self, bootstrap_state: np.ndarray, episode_done: bool):
        """
        Build one n-step transition from the front of the deque.
        Returns the transition tuple and pops the front entry.
        """
        G = 0.0
        gamma_acc = 1.0
        early_done = False

        for s, a, r, ns, d in self._deque:
            G += gamma_acc * r
            gamma_acc *= self.gamma
            if d:
                early_done = True
                break

        s0, a0, _, _, _ = self._deque[0]
        self._deque.popleft()

        if early_done:
            # Terminal before n steps — no bootstrap
            return s0, a0, G, bootstrap_state, True, 0.0
        else:
            return s0, a0, G, bootstrap_state, episode_done, gamma_acc

    def flush_to_buffer(self, next_state: np.ndarray, episode_done: bool = False):
        """Emit transitions while deque is full (length >= n) or episode ended."""
        while len(self._deque) >= self.n or (episode_done and self._deque):
            s, a, G, ns, done, geff = self._make_transition(next_state, episode_done)
            self.buf.push(s, a, G, ns, done, geff)

    def reset(self):
        self._deque.clear()

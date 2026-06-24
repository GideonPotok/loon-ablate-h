"""
Gym-style balloon station-keeping environment.

Bridges to the JS physics engine via a long-lived subprocess running
balloon_env_server.mjs (NDJSON over stdin/stdout).  One subprocess is
spawned per BalloonEnv instance and reused across many reset/step cycles.

State vector (20-dim float32), matching rl_agent.js / qr_agent.js
extractState compact mode:
  [0]     dist / STATION_RADIUS_M
  [1]     sin(bearing),  [2] cos(bearing)
  [3]     (alt_m - altBandLow) / altBandRange   clamped [0,1]
  [4]     vv_m_s / 2.5,  [5] ballast_kg / capacity
  [6]     wind_u_cur / 20,  [7] wind_v_cur / 20
  [8..19] 4 × (u/20, v/20, uncertainty/10)
          at alts 16625, 17125, 17625, 18125 m

Action space (17 discrete): index 0..16 maps to target altitudes
evenly spaced across the navigable band.  The JS side applies a
bang-bang chase command every 60 s for the 300 s nav interval.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


STATE_DIM  = 20
ACTION_DIM = 17

_SCRIPT_DIR  = Path(__file__).parent
_SERVERS_DIR = _SCRIPT_DIR / "servers"
_SERVER_MJS    = _SERVERS_DIR / "balloon_env_server.mjs"
_SERVER_V2_MJS = _SERVERS_DIR / "balloon_env_server_v2.mjs"


class BalloonEnv:
    """
    Gym-style balloon station-keeping environment.

    Each instance owns a persistent Node.js subprocess.  The subprocess
    stays alive across episodes (reset() reinitialises the episode state
    without restarting Node).

    Parameters
    ----------
    preset : str
        Wind preset: 'tropical' | 'strong-shear' | 'calm'
    duration_s : float
        Episode wall-clock length in seconds.
    seed : int
        RNG seed for spawn position and forecast degrader.
    node_bin : str, optional
        Path to the Node.js executable (default: 'node' from PATH).
    server_version : str, optional
        Which env server to spawn: 'v1' (current shipping, default) or 'v2'
        (in-development variant with new reward/state/shaping). Allows the
        ongoing training to keep using v1 while v2 features are built.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        preset: str = 'tropical',
        duration_s: float = 3600 * 6,
        seed: int = 42,
        node_bin: str = 'node',
        server_version: str = 'v1',
        flags: dict | None = None,
    ):
        self.preset         = preset
        self.duration_s     = duration_s
        self.seed           = seed
        self.server_version = server_version
        # v2 feature flags merged into every reset request. v1 server ignores unknown keys.
        # Example: {'use_reward_fix': True, 'terminal_twr_bonus': 50.0}
        self.flags          = dict(flags) if flags else {}

        self.observation_space_shape = (STATE_DIM,)
        self.n_actions = ACTION_DIM

        if server_version == 'v1':
            server_path = _SERVER_MJS
        elif server_version == 'v2':
            server_path = _SERVER_V2_MJS
        else:
            raise ValueError(f"Unknown server_version: {server_version!r} (expected 'v1' or 'v2')")
        self._server_path = server_path

        self._proc = subprocess.Popen(
            [node_bin, str(server_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,        # inherit parent stderr so JS errors are visible
            bufsize=0,          # unbuffered — we flush manually
            cwd=str(server_path.parent),
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _send(self, obj: dict) -> None:
        line = (json.dumps(obj, separators=(',', ':')) + '\n').encode()
        self._proc.stdin.write(line)
        self._proc.stdin.flush()

    def _recv(self) -> dict:
        raw = self._proc.stdout.readline()
        if not raw:
            raise RuntimeError(
                "balloon_env_server.mjs closed stdout unexpectedly "
                "(check stderr / Node.js exit code)"
            )
        resp = json.loads(raw.decode())
        if not resp.get('ok'):
            raise RuntimeError(f"BalloonEnv server error: {resp.get('error')}")
        return resp

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(self, spawn_offset_km: float | None = None) -> np.ndarray:
        """
        Reinitialise the episode and return the initial state vector.

        Parameters
        ----------
        spawn_offset_km : float, optional
            Override the spawn distance from station. Defaults to server's
            built-in 30 km. Used for varying initial-state distribution
            during training.
        """
        msg = {
            'cmd':        'reset',
            'preset':     self.preset,
            'duration_s': self.duration_s,
            'seed':       int(self.seed),
        }
        if spawn_offset_km is not None:
            msg['spawn_offset_km'] = float(spawn_offset_km)
        # Merge any v2 feature flags (server ignores unknown keys).
        for k, v in self.flags.items():
            msg[k] = v
        self._send(msg)
        resp = self._recv()
        return np.array(resp['state'], dtype=np.float32)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Apply one NAV_INTERVAL (5-minute) decision.

        Parameters
        ----------
        action : int
            Target-altitude index in [0, 16].

        Returns
        -------
        next_state : np.ndarray  shape (20,)
        reward     : float
        done       : bool
        info       : dict  {'dist_m': ..., 'twr50': ..., 'time_s': ..., 'alt_m': ...}
        """
        self._send({'cmd': 'step', 'action': int(action)})
        resp = self._recv()
        state = np.array(resp['state'], dtype=np.float32)
        return state, float(resp['reward']), bool(resp['done']), resp['info']

    def heuristic_step(self) -> tuple[int, np.ndarray, float, bool, dict[str, Any]]:
        """
        Let the JS navigator heuristic pick the action, then step the env.

        Returns
        -------
        action     : int   heuristic's chosen target-altitude bin (0–16)
        next_state : np.ndarray  shape (20,)
        reward     : float
        done       : bool
        info       : dict
        """
        self._send({'cmd': 'heuristic_step'})
        resp = self._recv()
        state = np.array(resp['state'], dtype=np.float32)
        return int(resp['action']), state, float(resp['reward']), bool(resp['done']), resp['info']

    def close(self) -> None:
        """Terminate the subprocess cleanly."""
        try:
            self._send({'cmd': 'close'})
        except OSError:
            pass
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

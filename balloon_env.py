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

# The helium/sand gas-balloon variant has its own dims (2 resource gauges → +1
# state dim; discrete release ladder → 11 actions instead of 17 target-alt bins).
GASSAND_STATE_DIM  = 21
GASSAND_ACTION_DIM = 11

_SCRIPT_DIR  = Path(__file__).parent
_SERVERS_DIR = _SCRIPT_DIR / "servers"
_SERVER_MJS         = _SERVERS_DIR / "balloon_env_server.mjs"
_SERVER_V2_MJS      = _SERVERS_DIR / "balloon_env_server_v2.mjs"
_SERVER_GASSAND_MJS = _SERVERS_DIR / "balloon_env_server_gassand.mjs"


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
        Which env server to spawn: 'v1' (current shipping, default), 'v2'
        (in-development variant with new reward/state/shaping), or 'gassand'
        (zero-pressure gas-balloon model: altitude via venting finite helium →
        sink and dropping finite sand → rise; 21-dim state, 11-action release
        ladder). Allows the ongoing training to keep using v1 while variants
        are built.
        
    wind_source : str, optional
        'preset' (default) — the synthetic layered winds every ablation
        through T was trained on. 'era5' — real reanalysis sampled from a
        WindArchive; requires server_version='v2' or 'gassand'.

        Under 'era5' the station is NOT the hardcoded (0°N, 170°E): each
        episode's target is wherever the archive sample landed, and the
        balloon spawns relative to that. `last_reset_info['wind']` carries
        the cell and start time, which an ERA5 result cannot be reproduced
        without.

        Two caveats before reading a score off this. ERA5's pressure-level
        product has no level between 70 and 100 hPa, and the balloon band
        (16.5–18.5 km ≈ 95–69 hPa) falls entirely in that gap, so in-band
        shear is an interpolation between two numbers — ~10 m/s median on
        the tropical Pacific tile, against ~16 m/s in IGRA soundings and a
        21.9 m/s step in the `tropical` preset. And sampleEpisode pins the
        column at the spawn cell for the whole episode, so the balloon
        drifts but its weather does not follow. Always run the navigator
        heuristic on the same episodes as a control.
    era5_dir : str, optional
        Directory of era5_wind_YYYY_MM.json files. Falls back to the
        LOON_ERA5_DIR env var. Required when wind_source='era5'.
    era5_min_shear_ms : float, optional
        Rejection-sample episodes until the band has opposing u-winds of at
        least half this on each side. 0 (default) accepts any cell, which is
        the honest setting — filtering hands the agent a world with more
        usable shear than the atmosphere has.
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
        wind_source: str = 'preset',
        era5_dir: str | None = None,
        era5_min_shear_ms: float = 0.0,
    ):
        self.preset         = preset
        self.duration_s     = duration_s
        self.seed           = seed
        self.server_version = server_version

        if wind_source not in ('preset', 'era5'):
            raise ValueError(f"Unknown wind_source: {wind_source!r} (expected 'preset' or 'era5')")
        if wind_source == 'era5' and server_version not in ('v2', 'gassand'):
            raise ValueError(
                f"wind_source='era5' needs server_version='v2' or 'gassand' "
                f"(got {server_version!r}); the v1 server has no ERA5 path"
            )
        self.wind_source       = wind_source
        self.era5_dir          = era5_dir or os.environ.get('LOON_ERA5_DIR')
        self.era5_min_shear_ms = float(era5_min_shear_ms)
        if wind_source == 'era5' and not self.era5_dir:
            raise ValueError(
                "wind_source='era5' needs era5_dir (or the LOON_ERA5_DIR env var) "
                "pointing at a directory of era5_wind_YYYY_MM.json files"
            )

        # Populated on each reset(). For ERA5 this carries the grid cell and
        # start time the episode drew, which an ERA5 result is not reproducible
        # without. For presets it is just {'source': 'preset', 'preset': ...}.
        self.last_reset_info: dict[str, Any] = {}
        # v2 feature flags merged into every reset request. v1 server ignores unknown keys.
        # Example: {'use_reward_fix': True, 'terminal_twr_bonus': 50.0}
        self.flags          = dict(flags) if flags else {}

        if server_version == 'v1':
            server_path = _SERVER_MJS
            state_dim, n_actions = STATE_DIM, ACTION_DIM
        elif server_version == 'v2':
            server_path = _SERVER_V2_MJS
            state_dim, n_actions = STATE_DIM, ACTION_DIM
        elif server_version == 'gassand':
            server_path = _SERVER_GASSAND_MJS
            state_dim, n_actions = GASSAND_STATE_DIM, GASSAND_ACTION_DIM
        else:
            raise ValueError(
                f"Unknown server_version: {server_version!r} "
                f"(expected 'v1', 'v2' or 'gassand')")
        self._server_path = server_path

        self.observation_space_shape = (state_dim,)
        self.n_actions = n_actions

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
        if self.wind_source != 'preset':
            msg['wind_source']       = self.wind_source
            msg['era5_dir']          = self.era5_dir
            msg['era5_min_shear_ms'] = self.era5_min_shear_ms
        # Merge any v2 feature flags (server ignores unknown keys).
        for k, v in self.flags.items():
            msg[k] = v
        self._send(msg)
        resp = self._recv()
        self.last_reset_info = resp.get('info', {})
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

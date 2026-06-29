"""
Run one training worker and save results to weights/.
Used by GitHub Actions — each matrix job runs one worker independently.

Usage:
    python run_worker.py --worker-id 3
"""
from __future__ import annotations

import argparse
import json
import queue
import time
from pathlib import Path

from ablate_j_train import worker_fn

WEIGHTS_DIR    = Path(__file__).parent / "weights"
WEIGHTS_PREFIX = "dqn_ablate_j"


class _LogQueue:
    def __init__(self, worker_id: int):
        self._q   = queue.Queue()
        self._wid = worker_id

    def put(self, msg: dict):
        if msg["type"] == "eval":
            pp = msg["per_preset"]
            m, s = divmod(int(msg["elapsed_s"]), 60)
            mark = " ★" if msg["is_best"] else ""
            print(
                f"[w{self._wid:02d}] {m}m{s:02d}s  ep {msg['ep']:4d}"
                f" [{msg['tier']:3s}]"
                f"  score {msg['score']*100:5.1f}%"
                f"  trop {pp['tropical']*100:5.1f}%"
                f"  shear {pp['strong-shear']*100:5.1f}%"
                f"  calm {pp['calm']*100:5.1f}%"
                f"  ε {msg['epsilon']:.3f}" + mark,
                flush=True,
            )
        self._q.put(msg)

    def get(self, *a, **kw): return self._q.get(*a, **kw)
    def empty(self):          return self._q.empty()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="Cap total episodes (0 = full run). Use for smoke tests.")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Starting worker_id={args.worker_id}", flush=True)
    q  = _LogQueue(args.worker_id)
    t0 = time.time()

    worker_fn(args.worker_id, q, max_episodes=args.max_episodes)

    m, s  = divmod(int(time.time() - t0), 60)
    ckpt  = WEIGHTS_DIR / f"{WEIGHTS_PREFIX}_w{args.worker_id:02d}.pt"
    jpath = WEIGHTS_DIR / f"{WEIGHTS_PREFIX}_w{args.worker_id:02d}.json"

    if ckpt.exists() and jpath.exists():
        info = json.loads(jpath.read_text())
        print(f"\n[w{args.worker_id:02d}] Finished in {m}m{s:02d}s", flush=True)
        print(f"[w{args.worker_id:02d}] Best score {info['best_score']*100:.1f}%"
              f"  ep {info['best_episode']}", flush=True)
        print(f"[w{args.worker_id:02d}] Weights → {ckpt}", flush=True)
    else:
        print(f"\n[w{args.worker_id:02d}] WARNING: no checkpoint found after {m}m{s:02d}s"
              " — worker may not have reached its first eval", flush=True)


if __name__ == "__main__":
    main()

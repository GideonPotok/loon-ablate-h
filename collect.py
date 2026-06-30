"""
Collect worker results after a GitHub Actions matrix run and pick the winner.

Usage:
    python collect.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

WEIGHTS_DIR    = Path(__file__).parent / "weights"
WEIGHTS_PREFIX = "dqn_ablate_l"


def main():
    json_files = sorted(WEIGHTS_DIR.glob(f"{WEIGHTS_PREFIX}_w*.json"))
    if not json_files:
        raise FileNotFoundError(
            f"No worker JSON files found in {WEIGHTS_DIR}.\n"
            "Download worker artifacts before running this script."
        )

    results = []
    for jf in json_files:
        data = json.loads(jf.read_text())
        pt   = WEIGHTS_DIR / jf.name.replace(".json", ".pt")
        if not pt.exists():
            print(f"  WARNING: weights file missing for {jf.name}, skipping")
            continue
        results.append(data)

    if not results:
        raise RuntimeError("No complete worker results (both .json and .pt) found.")

    print(f"\n{'─'*60}")
    print(f"Results ({len(results)} workers):")
    for r in sorted(results, key=lambda x: x["worker_id"]):
        bp = r.get("best_per_preset") or {}
        print(
            f"  w{r['worker_id']:02d}  score {r['best_score']*100:5.1f}%"
            f"  trop {bp.get('tropical', 0)*100:5.1f}%"
            f"  shear {bp.get('strong-shear', 0)*100:5.1f}%"
            f"  calm {bp.get('calm', 0)*100:5.1f}%"
            f"  ep {r['best_episode']}"
        )

    winner     = max(results, key=lambda r: r["best_score"])
    winner_src = WEIGHTS_DIR / f"{WEIGHTS_PREFIX}_w{winner['worker_id']:02d}.pt"
    winner_dst = WEIGHTS_DIR / f"{WEIGHTS_PREFIX}.pt"
    shutil.copy2(winner_src, winner_dst)

    summary = {
        "ablation":          "L_fourier_time_features",
        "winner_worker":     winner["worker_id"],
        "best_score":        winner["best_score"],
        "best_episode":      winner["best_episode"],
        "best_per_preset":   winner["best_per_preset"],
        "n_workers":         len(results),
        "workers": [
            {"worker_id": r["worker_id"], "best_score": r["best_score"],
             "best_episode": r["best_episode"]}
            for r in sorted(results, key=lambda r: r["worker_id"])
        ],
    }
    summary_path = WEIGHTS_DIR / f"{WEIGHTS_PREFIX}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print(f"\nWinner: w{winner['worker_id']:02d}  "
          f"score {winner['best_score']*100:.1f}%  ep {winner['best_episode']}")
    print(f"Weights → {winner_dst}")
    print(f"Summary → {summary_path}")


if __name__ == "__main__":
    main()

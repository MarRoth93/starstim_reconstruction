#!/usr/bin/env python3
"""Old-lab ridge -> lab sessions (results history: R5). Per session and pooled, for pulse-free blocks and all no-pulse trials.

    python scripts/run_transfer_sessions.py [--kind stim]

Readout: ridge on all usable old-lab trials (25 shared channels, ZCA per person, image averages; images that also appear in a
session are dropped), alpha by 4-fold image CV on the old data only. Per session: ZCA (label-free), per-image mean of the
chosen trials, coarse8 against the session's own images; pooled null shuffles image labels within each session.
Output: results/transfer_sessions.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import reconstruct  # noqa: E402
from starstim_recon.data_sessions import session_ids  # noqa: E402
from starstim_recon.images import pixel32_many  # noqa: E402
from starstim_recon.metrics import coarse8, pooled_score  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", default=None); ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args()
    S = reconstruct.load_sessions(session_ids(args.kind))
    shared = {f.name for s in S.values() for f in s["files"].values()}
    predict, alpha = reconstruct.old_lab_readout(exclude_basenames=shared)
    print(f"old-lab readout alpha {alpha:g}; sessions {list(S)}")
    results = {}
    for trials in ("pulse_free", "all_nopulse"):
        cells = {}
        for sid, s in S.items():
            X, trig = reconstruct.session_image_means(s, trials)
            if len(trig) < 3:
                continue
            pred = torch.from_numpy(predict(X).reshape(-1, 3, 32, 32)).float()
            values, matrix = coarse8(pred, pixel32_many([s["files"][int(u)] for u in trig]), np.arange(len(trig)))
            cells[sid] = (values, matrix, np.arange(len(trig)))
            results[f"{trials}/{sid}"] = pooled_score([cells[sid]], args.n_perm)
        results[f"{trials}/pooled"] = pooled_score(list(cells.values()), args.n_perm)
        for k, r in results.items():
            if k.startswith(trials):
                print(f"{k:24s} n {r['n_images']:4d} coarse8 {r['coarse8']:+.4f} p {r['permutation_p']:.4f} pair {r['pairwise_id']:.3f}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "transfer_sessions.json").write_text(json.dumps({"alpha": alpha, "results": results}, indent=2))


if __name__ == "__main__":
    main()

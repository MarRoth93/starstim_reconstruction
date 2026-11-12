#!/usr/bin/env python3
"""One decoder on all usable rapid-stream recordings, averaged per image.

    python scripts/run_rsvp_pooled.py

Every usable rapid-stream session of configs/sessions.yaml (data_sessions.rsvp_usable) is loaded with load_rsvp, whitened
per recording (ZCA, label-free) and averaged per image; an image's recordings are then averaged, as the old-lab result
pools people. One decoder is scored over all images in 8 image folds. Fixed on 2026-11-12, before the first run:
  filter     none | 0.1-30 Hz on the continuous data (load_rsvp's bandpass)
  channels   the 31 lab-contract channels | the 10 posterior ones (channels.POSTERIOR_10)
  readouts   own_decoder       ridge on the averaged recordings, alpha in 4 inner image folds
             things_zeroshot   linear THINGS ridge (alpha 1e-3), fitted without any image shown in the lab
             things_prior      own_decoder shrunk toward the THINGS ridge weights
Scores: coarse8 with a permutation null over image labels, against all images and within each image set (the images one
group of recordings shares), which is comparable with the per-recording scores of scripts/run_rsvp_reconstruction.py.
Output: results/rsvp_pooled.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import reconstruct  # noqa: E402
from starstim_recon.channels import POSTERIOR_10, STARSTIM_31  # noqa: E402
from starstim_recon.data_sessions import load_rsvp, rsvp_usable, session_ids  # noqa: E402
from starstim_recon.images import pixel32_many  # noqa: E402
from starstim_recon.metrics import score  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import nested_cv, nested_prior_cv, zca_by_subject  # noqa: E402

ZCA_SCALE = 1.1                                      # the THINGS lab-contract cache is ZCA-whitened x1.1
FILTERS = {"none": None, "0.1-30 Hz": (0.1, 30.0)}
CHANNELS = {"31 channels": STARSTIM_31, "10 posterior": POSTERIOR_10}


def load(sessions, bandpass):
    """-> {session: (image means (images, 31, 250) after ZCA, image files, kept trials)}"""
    D = {}
    for s in sessions:
        R = load_rsvp(s, bandpass=bandpass); trig = R["tr"].trigger.values.astype(int); uniq = np.array(sorted(set(trig)))
        x = zca_by_subject(R["ep"], np.array(["session"] * len(trig)))
        D[s] = (torch.stack([x[trig == u].mean(0) for u in uniq]), [R["files"][int(u)] for u in uniq], len(trig))
    return D


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-perm", type=int, default=2000); args = ap.parse_args()
    sessions = [s for s in session_ids(rsvp=True) if rsvp_usable(s)]
    out = {"sessions": sessions, "results": {}}
    for fname, bandpass in FILTERS.items():
        D = load(sessions, bandpass); paths = {f.name: f for _, files, _ in D.values() for f in files}; names = sorted(paths)
        true = pixel32_many([paths[n] for n in names]); n = len(names); idx = np.arange(n)
        rows = {s: {f.name: k for k, f in enumerate(files)} for s, (_, files, _) in D.items()}
        avg = torch.stack([torch.stack([D[s][0][rows[s][nm]] for s in sessions if nm in rows[s]]).mean(0) for nm in names])
        sets = {}                                                 # recordings that share an image set -> its images
        for s in sessions:
            sets.setdefault(frozenset(rows[s]), []).append(s)
        sets = {"+".join(v): np.array([names.index(nm) for nm in sorted(k)]) for k, v in sets.items()}
        print(f"\nfilter {fname}: {len(sessions)} recordings, {sum(d[2] for d in D.values())} kept trials, {n} images; "
              f"image sets {[(k, len(v)) for k, v in sets.items()]}", flush=True)
        out.update(n_images=n, image_sets={k: len(v) for k, v in sets.items()}); out["results"][fname] = {}
        for cname, chans in CHANNELS.items():
            X = avg[:, [STARSTIM_31.index(c) for c in chans]].flatten(1).numpy().astype(np.float64)
            f_T = reconstruct.things_readout(exclude_basenames=set(names), channels=chans)
            P = {"own_decoder": nested_cv(X, idx, true, idx), "things_zeroshot": f_T(torch.from_numpy(X * ZCA_SCALE).float()).numpy(),
                 "things_prior": nested_prior_cv(X * ZCA_SCALE, idx, true, idx, f_T)[0]}
            out["results"][fname][cname] = {}
            for k, p in P.items():
                p = torch.from_numpy(np.asarray(p).reshape(-1, 3, 32, 32)).float()
                r = {"all": score(p, true, idx, np.zeros(n), args.n_perm)[0]}
                r.update({key: score(p[sub], true[sub], np.arange(len(sub)), np.zeros(len(sub)), args.n_perm)[0] for key, sub in sets.items()})
                out["results"][fname][cname][k] = r
                print(f"  {cname:12s} {k:16s} all images {r['all']['coarse8']:+.4f} (p {r['all']['permutation_p']:.4f}, pair id "
                      f"{r['all']['pairwise_id']:.4f}) | " + "  ".join(f"{key} {r[key]['coarse8']:+.4f} (p {r[key]['permutation_p']:.3f})" for key in sets),
                      flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rsvp_pooled.json").write_text(json.dumps(out, indent=2))
    print("wrote", RESULTS / "rsvp_pooled.json")


if __name__ == "__main__":
    main()

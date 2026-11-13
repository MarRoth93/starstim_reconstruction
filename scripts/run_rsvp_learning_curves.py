#!/usr/bin/env python3
"""What the pooled rapid-stream decoders gain from more images and from more recordings per image.

    python scripts/run_rsvp_learning_curves.py [--draws 20] [--null-draws 40] [--recompute]

Input, image folds and readouts as scripts/run_rsvp_pooled.py (31 channels; own_decoder and things_prior; without and
with the 0.1-30 Hz band-pass). The training data of every fold are thinned at random, the test images stay the same:
  images       training on 25, 50, 100, 150, 200 or all images of the other folds (nested random subsets)
  recordings   every training image averaged over 1, 2, 3 or all of its recordings (drawn without replacement)
on the full grid images x recordings, with the test images at their full average. At all training images also: only the
test images reduced to r recordings, and both. coarse8 over all images, per-image values averaged over the draws;
differences by paired bootstrap over images. Design fixed on 2026-11-12, before the first run.

Null, added after the first run: the same decoders with the EEG shuffled against the images, per number of training
images (all recordings). A cross-validated decoder without signal does not score 0: its prediction contains the mean
target of its training fold, which is anti-correlated with the held-out images. Scores in cells with different numbers
of training images are therefore compared against their own null.

The ridge is solved from Gram matrices and on the 8x8 targets (the prediction is linear in the targets, so pooling it
equals predicting the pooled targets), which makes the many small fits fast. With all images and all recordings it must
reproduce results/rsvp_pooled.json; the script checks that.
Output: results/rsvp_learning_curves.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import reconstruct  # noqa: E402
from starstim_recon.data_sessions import rsvp_usable, session_ids  # noqa: E402
from starstim_recon.images import pixel32_many  # noqa: E402
from starstim_recon.metrics import coarse8, paired_bootstrap  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import ALPHAS, PRIOR_ALPHAS  # noqa: E402

ZCA_SCALE = 1.1                                      # the THINGS lab-contract cache is ZCA-whitened x1.1
FILTERS = {"none": None, "0.1-30 Hz": (0.1, 30.0)}
FOLDS, INNER = 8, 4                                  # as readout.nested_cv
IMAGES, RECORDINGS = (25, 50, 100, 150, 200, None), (1, 2, 3, None)          # None = all
READOUTS = ("own_decoder", "things_prior")
DIFFERENCES = {"images_all_minus_100": ((None, None), (100, None)), "recordings_all_minus_1": ((None, None), (None, 1)),
               "same_budget_200x1_minus_100x2": ((200, 1), (100, 2))}        # (images, recordings) in training, test at full average
THREADS = 8                                          # thousands of 200 x 200 problems: more threads are slower


def lab(q):
    return "all" if q is None else str(q)


def ridge_path(Gtt, Get, T, tr, te, alphas):
    """readout.dual_ridge_predict for several alphas, from Gram matrices: Gtt = training x training inputs, Get = test x
    training inputs (all images x all images; tr, te index them). -> one prediction of the te rows per alpha."""
    A = Gtt[np.ix_(tr, tr)]; m = A.mean(1); g = A.mean(); B = Get[np.ix_(te, tr)]
    w, V = np.linalg.eigh(A - m[:, None] - m[None, :] + g)
    mu = T[tr].mean(0, keepdims=True); VtY = V.T @ (T[tr] - mu); KeV = (B - B.mean(1, keepdims=True) - m[None, :] + g) @ V
    return [mu + (KeV / (w + a)) @ VtY for a in alphas]


def c8(pred8, true, local):
    return coarse8(torch.from_numpy(pred8.reshape(-1, 3, 8, 8)).float(), true, local)[0]


def oof(Gtt, Get, T, base_tr, base_te, true, fold, train, alphas):
    """Out-of-fold 8x8 predictions as readout.nested_cv / nested_prior_cv: alpha by coarse8 in inner image folds of the
    training images train[k] of fold k. T = targets (minus the prior's predictions base_tr for things_prior; base_te is
    added back to the test predictions; both 0 for own_decoder)."""
    preds = np.zeros_like(T)
    for k in range(FOLDS):
        tr = np.sort(train[k]); te = np.flatnonzero(fold == k); ifold = np.arange(len(tr)) % INNER; sc = np.zeros(len(alphas))
        for j in range(INNER):
            itr, ite = tr[ifold != j], tr[ifold == j]
            for a, p in enumerate(ridge_path(Gtt, Gtt, T, itr, ite, alphas)):
                sc[a] += c8(base_tr[ite] + p, true, ite).mean()
        preds[te] = base_te[te] + ridge_path(Gtt, Get, T, tr, te, [alphas[int(np.argmax(sc))]])[0]
    return preds


def both(Gtt, Get, Y, F0_tr, F0_te, true, fold, train):
    """-> {readout: out-of-fold predictions}; F0_* = the THINGS ridge's predictions of the training / test inputs."""
    zero = np.zeros_like(Y); s = ZCA_SCALE ** 2
    return {"own_decoder": oof(Gtt, Get, Y, zero, zero, true, fold, train, ALPHAS),
            "things_prior": oof(Gtt * s, Get * s, Y - F0_tr, F0_tr, F0_te, true, fold, train, PRIOR_ALPHAS)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--draws", type=int, default=20); ap.add_argument("--null-draws", type=int, default=40)
    ap.add_argument("--recompute", action="store_true", help="rebuild the pooled recordings instead of reading cache/")
    args = ap.parse_args(); torch.set_num_threads(THREADS); limit = threadpool_limits(limits=THREADS)   # noqa: F841
    sessions = [s for s in session_ids(rsvp=True) if rsvp_usable(s)]
    assert (RESULTS / "rsvp_pooled.json").exists(), "run scripts/run_rsvp_pooled.py first"
    pooled = json.loads((RESULTS / "rsvp_pooled.json").read_text())
    assert pooled["sessions"] == sessions, f"results/rsvp_pooled.json covers {pooled['sessions']}, usable are {sessions}: rerun it"
    out = {"sessions": sessions, "draws": args.draws, "null_draws": args.null_draws, "images": [lab(m) for m in IMAGES],
           "recordings": [lab(r) for r in RECORDINGS], "results": {}}
    for fname, bandpass in FILTERS.items():
        Z = reconstruct.rsvp_pooled(sessions, bandpass, args.recompute); n = len(Z["names"]); idx = np.arange(n)
        true = pixel32_many(Z["paths"]); Y = F.adaptive_avg_pool2d(true, 8).flatten(1).numpy().astype(np.float64)
        perm = np.random.default_rng(0).permutation(n); fold = np.empty(n, int); fold[perm] = np.arange(n) % FOLDS      # nested_cv's
        rec_x = Z["rec_x"].flatten(1).numpy().astype(np.float64); rows = [np.flatnonzero(Z["rec_img"] == i) for i in range(n)]
        per_image = {s: Z["kept"][s] / (Z["rec_session"] == s).sum() for s in sessions}             # kept trials per image
        trials = np.array([per_image[s] for s in Z["rec_session"]])
        Xfull = Z["avg"].flatten(1).numpy().astype(np.float64); Gfull = Xfull @ Xfull.T
        f_T = reconstruct.things_readout(exclude_basenames=set(Z["names"]))

        def prior8(X):
            return F.adaptive_avg_pool2d(f_T(torch.from_numpy(X * ZCA_SCALE).float()).view(-1, 3, 32, 32), 8).flatten(1).numpy().astype(np.float64)
        F0full = prior8(Xfull); vals, null, budget = {}, {}, {r: [] for r in RECORDINGS}

        def add(store, key, preds):
            for ro, p in preds.items():
                store.setdefault((ro,) + key, []).append(c8(p, true, idx))

        for d in range(args.draws):
            rng = np.random.default_rng(100 + d); order = {k: rng.permutation(np.flatnonzero(fold != k)) for k in range(FOLDS)}
            for r in RECORDINGS:
                pick = [q if r is None else rng.choice(q, min(r, len(q)), replace=False) for q in rows]
                Xr = Xfull if r is None else np.stack([rec_x[q].mean(0) for q in pick]); budget[r].append(np.mean([trials[q].sum() for q in pick]))
                Grr, Gfr = Xr @ Xr.T, Xfull @ Xr.T; F0r = F0full if r is None else prior8(Xr)
                for m in IMAGES:
                    train = {k: order[k] if m is None else order[k][:m] for k in range(FOLDS)}
                    add(vals, ("grid", m, r), both(Grr, Gfr, Y, F0r, F0full, true, fold, train))
                if r is not None:
                    add(vals, ("both_reduced", None, r), both(Grr, Grr, Y, F0r, F0r, true, fold, order))
                    add(vals, ("test_reduced", None, r), both(Gfull, Gfr.T, Y, F0full, F0r, true, fold, order))
        for d in range(args.null_draws):
            rng = np.random.default_rng(500 + d); q = rng.permutation(n); Gq = Gfull[np.ix_(q, q)]
            order = {k: rng.permutation(np.flatnonzero(fold != k)) for k in range(FOLDS)}
            for m in IMAGES:
                train = {k: order[k] if m is None else order[k][:m] for k in range(FOLDS)}
                add(null, (m,), both(Gq, Gq, Y, F0full[q], F0full[q], true, fold, train))

        V = {k: np.mean(v, 0) for k, v in vals.items()}; sd = {k: float(np.std([q.mean() for q in v])) for k, v in vals.items()}
        print(f"\nfilter {fname}: {n} images, {np.mean([len(q) for q in rows]):.2f} recordings per image; kept trials per image at "
              + ", ".join(f"{lab(r)} recordings {np.mean(b):.0f}" for r, b in budget.items()), flush=True)
        res = {"trials_per_image": {lab(r): float(np.mean(b)) for r, b in budget.items()},      # what "all" stands for on both axes:
               "training_images_all": float(np.mean([(fold != k).sum() for k in range(FOLDS)])),
               "recordings_per_image_all": float(np.mean([len(q) for q in rows]))}
        for ro in READOUTS:
            full = float(V[ro, "grid", None, None].mean()); ref = pooled["results"][fname]["31 channels"][ro]["all"]["coarse8"]
            assert abs(full - ref) < 1e-4, f"{ro}, {fname}: {full:+.5f} with all images and recordings, results/rsvp_pooled.json has {ref:+.5f}"
            res[ro] = {"grid": {lab(m): {lab(r): {"coarse8": float(V[ro, "grid", m, r].mean()), "sd_draws": sd[ro, "grid", m, r]}
                                        for r in RECORDINGS} for m in IMAGES},
                       "null": {lab(m): {"coarse8": float(np.mean(null[ro, m])), "sd_draws": float(np.std([q.mean() for q in null[ro, m]]))}
                                for m in IMAGES},
                       **{c: {lab(r): float(V[ro, c, None, r].mean()) for r in RECORDINGS[:-1]} for c in ("test_reduced", "both_reduced")},
                       "differences": {name: paired_bootstrap(V[ro, "grid", a[0], a[1]], V[ro, "grid", b[0], b[1]]) for name, (a, b) in DIFFERENCES.items()}}
            res[ro]["differences"]["test_recordings_all_minus_1"] = paired_bootstrap(V[ro, "grid", None, None], V[ro, "test_reduced", None, 1])
            print(f"  {ro}: training images (rows) x recordings per training image (columns), test images at their full average")
            print("      images " + "".join(f"{lab(r):>9s}" for r in RECORDINGS) + "   EEG shuffled (SD)")
            for m in IMAGES:
                print(f"      {lab(m):>6s} " + "".join(f"  {res[ro]['grid'][lab(m)][lab(r)]['coarse8']:+.4f}" for r in RECORDINGS)
                      + f"   {res[ro]['null'][lab(m)]['coarse8']:+.4f} ({res[ro]['null'][lab(m)]['sd_draws']:.3f})")
            for c in ("test_reduced", "both_reduced"):
                print(f"      all images, {c.replace('_', ' ')}: " + "  ".join(f"{r} recordings {v:+.4f}" for r, v in res[ro][c].items()))
            for name, bt in res[ro]["differences"].items():
                print(f"      {name:32s} {bt['mean_diff']:+.4f} [{bt['ci95'][0]:+.3f}, {bt['ci95'][1]:+.3f}]", flush=True)
        out["results"][fname] = res
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rsvp_learning_curves.json").write_text(json.dumps(out, indent=2))
    print("wrote", RESULTS / "rsvp_learning_curves.json")


if __name__ == "__main__":
    main()

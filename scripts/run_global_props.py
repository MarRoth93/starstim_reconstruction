#!/usr/bin/env python3
"""8 global image properties instead of a map, and per-person readouts averaged in prediction space (results history: R7).

    python scripts/run_global_props.py [--kind stim]

Targets (images.PROPS): mean L*, a*, b*, RMS contrast, chroma, spatial-frequency slope, left-right and top-bottom L*
difference, z-scored over the old-lab training images. Old lab (TRAIN split, 31 channels, ZCA per person, 8 outer image folds):
  A  pooled ridge on image-averaged EEG (0-1 s);  A5  the same, 0-0.5 s
  B  one ridge per person on single trials, predictions averaged over the people who saw the image
  C  properties computed from the 32x32 pixel ridge reconstructions (pixel-route baseline)
Transfer: A and B fitted on all old-lab images (25 shared channels) -> session pulse-free image means, centred within session.
Ridge alpha per property by 4-fold inner image CV (Pearson r), grid 1e0-1e6. Output: results/global_props.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import data_old_lab, reconstruct  # noqa: E402
from starstim_recon.channels import SHARED_25  # noqa: E402
from starstim_recon.data_sessions import session_ids  # noqa: E402
from starstim_recon.images import PROPS, image_properties, pixel32_many, properties  # noqa: E402
from starstim_recon.metrics import corr_cols  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import aggregate, dual_ridge_predict, nested_cv, zca_by_subject  # noqa: E402

ALPHAS = tuple(10.0 ** k for k in range(0, 7))


def ridge_cv_fit(X, Y, img, seed=0, n_inner=4):
    """Per-property alpha by inner image-fold CV (r) -> predict(Xnew) fitted on all rows."""
    images = np.unique(img); perm = np.random.default_rng(seed).permutation(len(images))
    fold_of = {int(images[perm[i]]): i % n_inner for i in range(len(images))}; fold = np.array([fold_of[int(t)] for t in img])
    oof = {a: np.zeros_like(Y) for a in ALPHAS}
    for k in range(n_inner):
        tr, te = fold != k, fold == k
        if tr.sum() < 3 or te.sum() < 1:
            continue
        for a in ALPHAS:
            oof[a][te] = dual_ridge_predict(X[tr], Y[tr], X[te], a)
    best = [ALPHAS[int(np.argmax([corr_cols(oof[a][:, [j]], Y[:, [j]])[0] for a in ALPHAS]))] for j in range(Y.shape[1])]
    return lambda Xn: np.stack([dual_ridge_predict(X, Y[:, [j]], Xn, best[j])[:, 0] for j in range(Y.shape[1])], 1)


def score(P, T, groups, n_perm, seed=1):
    """Mean-over-properties r; P and T centred within groups (sessions); labels permuted within groups."""
    P, T = P.copy(), T.copy()
    for g in np.unique(groups):
        m = groups == g; P[m] -= P[m].mean(0); T[m] -= T[m].mean(0)
    r = corr_cols(P, T); obs = float(r.mean()); rng = np.random.default_rng(seed); ge = 0
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    for _ in range(n_perm):
        perm = np.arange(len(P))
        for ix in idx:
            perm[ix] = rng.permutation(ix)
        ge += corr_cols(P, T[perm]).mean() >= obs
    return {"n_images": int(len(P)), "mean_r": obs, "permutation_p": (ge + 1) / (n_perm + 1),
            "r_by_property": dict(zip(PROPS, np.round(r, 4).tolist()))}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--kind", default=None); ap.add_argument("--n-perm", type=int, default=5000)
    args = ap.parse_args(); results = {}

    eeg, targets, subjects, ibt = data_old_lab.load("train")
    uniq = np.array(sorted(ibt)); props = image_properties([ibt[int(t)] for t in uniq])
    mu, sd = props.mean(0), props.std(0); Z = (props - mu) / sd; pos = {int(t): i for i, t in enumerate(uniq)}
    x = zca_by_subject(eeg, subjects)
    perm = np.random.default_rng(0).permutation(len(uniq)); fold_of = {int(uniq[perm[i]]): i % 8 for i in range(len(uniq))}
    zeros = lambda n: np.zeros(n)

    def pooled(win):
        Xa, img, _ = aggregate(x[..., :win], targets, subjects, "image")
        P = np.zeros((len(img), len(PROPS))); f = np.array([fold_of[int(t)] for t in img]); Y = Z[[pos[int(t)] for t in img]]
        for k in range(8):
            P[f == k] = ridge_cv_fit(Xa[f != k], Y[f != k], img[f != k])(Xa[f == k])
        return P, Y

    P, Y = pooled(250); results["old/A_pooled_0-1s"] = score(P, Y, zeros(len(Y)), args.n_perm)
    P, Y = pooled(125); results["old/A5_pooled_0-0.5s"] = score(P, Y, zeros(len(Y)), args.n_perm)

    X1 = x.flatten(1).numpy().astype(np.float64); P = np.zeros((len(uniq), len(PROPS))); cnt = np.zeros(len(uniq))
    f_trial = np.array([fold_of[int(t)] for t in targets])
    for s in np.unique(subjects):
        for k in range(8):
            tr = (subjects == s) & (f_trial != k)
            te_imgs = sorted({int(t) for t in targets[(subjects == s) & (f_trial == k)]})
            if not te_imgs:
                continue
            predict = ridge_cv_fit(X1[tr], Z[[pos[int(t)] for t in targets[tr]]], targets[tr])
            Xte = np.stack([X1[(subjects == s) & (targets == t)].mean(0) for t in te_imgs])
            for t, p in zip(te_imgs, predict(Xte)):
                P[pos[t]] += p; cnt[pos[t]] += 1
    seen = cnt > 0
    results["old/B_per_person"] = score(P[seen] / cnt[seen, None], Z[seen], zeros(seen.sum()), args.n_perm)

    Xa, img, _ = aggregate(x, targets, subjects, "image")
    true32 = pixel32_many([ibt[int(u)] for u in uniq])
    pix = nested_cv(Xa, img, true32, uniq).reshape(-1, 3, 32, 32)
    up = torch.nn.functional.interpolate(torch.from_numpy(pix).float(), size=128, mode="bilinear").numpy()
    Pc = (np.stack([properties(u) for u in up]) - mu) / sd
    results["old/C_via_pixel_ridge"] = score(Pc, Z[[pos[int(t)] for t in img]], zeros(len(img)), args.n_perm)

    S = reconstruct.load_sessions(session_ids(args.kind))
    shared = {f.name for s in S.values() for f in s["files"].values()}
    eeg2, t2, s2, ibt2 = data_old_lab.load(None, SHARED_25, shared)
    u2 = np.array(sorted(ibt2)); pos2 = {int(t): i for i, t in enumerate(u2)}
    Z2 = (image_properties([ibt2[int(t)] for t in u2]) - mu) / sd; x2 = zca_by_subject(eeg2, s2)
    Xi = torch.stack([x2[t2 == t].mean(0) for t in u2]).flatten(1).numpy().astype(np.float64)
    fit_A = ridge_cv_fit(Xi, Z2, u2); X2 = x2.flatten(1).numpy().astype(np.float64)
    fit_B = [ridge_cv_fit(X2[s2 == p], Z2[[pos2[int(t)] for t in t2[s2 == p]]], t2[s2 == p]) for p in np.unique(s2)]
    rows = {"A": [], "B": []}; truth, groups = [], []
    for sid, s in S.items():
        Xs, trig = reconstruct.session_image_means(s, "pulse_free")
        if len(trig) < 3:
            continue
        rows["A"].append(fit_A(Xs)); rows["B"].append(np.mean([f(Xs) for f in fit_B], 0))
        truth.append((image_properties([s["files"][int(t)] for t in trig]) - mu) / sd); groups += [sid] * len(trig)
    T = np.concatenate(truth); groups = np.array(groups)
    for k in ("A", "B"):
        results[f"transfer/{k}"] = score(np.concatenate(rows[k]), T, groups, args.n_perm)
    for k, v in results.items():
        print(f"{k:24s} n {v['n_images']:4d} mean r {v['mean_r']:+.4f} p {v['permutation_p']:.4f} |",
              " ".join(f"{a} {b:+.2f}" for a, b in v["r_by_property"].items()))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "global_props.json").write_text(json.dumps({"results": results, "properties": PROPS}, indent=2))


if __name__ == "__main__":
    main()

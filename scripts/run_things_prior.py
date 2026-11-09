#!/usr/bin/env python3
"""THINGS as a weight prior for the old-lab ridge (results history: R6).

    python scripts/run_things_prior.py

f_T: linear THINGS ridge (31 channels, lab contract, old-lab images dropped), alpha chosen by coarse8 on THINGS test-200.
Old lab (TRAIN split, image level, R1's 8 outer image folds, alpha tuned in 4 inner folds):
  ridge_eeg    the plain lab ridge (the bar)
  things_only  f_T applied to the lab image averages (zero-shot linear THINGS readout)
  prior_ridge  argmin ||Y - Xb||^2 + alpha ||b - b_T||^2 = f_T + a ridge on the lab residuals Y - f_T(X)
Output: results/things_prior.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import data_old_lab, data_things  # noqa: E402
from starstim_recon.images import pixel32_many  # noqa: E402
from starstim_recon.metrics import coarse8, paired_bootstrap, score  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import ALPHAS, aggregate, dual_ridge_predict, fit_primal_ridge, nested_cv, zca_by_subject  # noqa: E402

THINGS_ALPHAS = tuple(10.0 ** k for k in range(-7, 3))
PRIOR_ALPHAS = ALPHAS + (1e7, 1e8, 1e9)
ZCA_SCALE = 1.1                                     # the THINGS lab-contract cache is ZCA-whitened x1.1


def things_readout():
    keep = torch.tensor([i for i in range(16540) if i not in data_things.lab_image_conditions()])
    Xtr, Ytr = data_things.eeg_means("train")[keep], data_things.targets("train")[keep].flatten(1)
    Xte, Yte = data_things.eeg_means("test"), data_things.targets("test")
    best, scores = None, []
    for a in THINGS_ALPHAS:
        predict = fit_primal_ridge(Xtr, Ytr, a)
        scores.append(float(coarse8(predict(Xte).view(-1, 3, 32, 32), Yte, np.arange(len(Yte)))[0].mean()))
        if scores[-1] == max(scores):
            best = (a, predict)
    print(f"THINGS ridge alpha {best[0]:g}; test-200 coarse8 by alpha {np.round(scores, 4).tolist()}")
    return best[1], best[0], scores


def nested_prior_cv(X, img, true32, uniq, f_T, outer_folds=8, inner_folds=4, seed=0):
    images = np.array(sorted(set(img.tolist()))); perm = np.random.default_rng(seed).permutation(len(images))
    fold_of = {int(images[perm[i]]): i % outer_folds for i in range(len(images))}; fold = np.array([fold_of[int(t)] for t in img])
    position = {int(t): i for i, t in enumerate(uniq)}
    Y = np.stack([true32[position[int(t)]].flatten().numpy() for t in img]).astype(np.float64)
    F0 = f_T(torch.from_numpy(X).float()).numpy().astype(np.float64)
    preds, chosen = np.zeros_like(Y), []
    for k in range(outer_folds):
        tr, te = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        tr_images = np.array(sorted(set(img[tr].tolist())))
        inner_of = {int(tr_images[i]): i % inner_folds for i in range(len(tr_images))}
        ifold = np.array([inner_of[int(t)] for t in img[tr]]); scores = []
        for alpha in PRIOR_ALPHAS:
            vals = []
            for j in range(inner_folds):
                itr, ite = tr[ifold != j], tr[ifold == j]
                pred = F0[ite] + dual_ridge_predict(X[itr], Y[itr] - F0[itr], X[ite], alpha)
                loc = np.array([position[int(t)] for t in img[ite]])
                vals.append(coarse8(torch.from_numpy(pred.reshape(-1, 3, 32, 32)).float(), true32, loc)[0].mean())
            scores.append(float(np.mean(vals)))
        alpha = PRIOR_ALPHAS[int(np.argmax(scores))]; chosen.append(alpha)
        preds[te] = F0[te] + dual_ridge_predict(X[tr], Y[tr] - F0[tr], X[te], alpha)
    return preds, chosen


def main():
    f_T, t_alpha, t_scores = things_readout()
    eeg, targets, subjects, ibt = data_old_lab.load("train")
    uniq = np.array(sorted(ibt)); true32 = pixel32_many([ibt[int(u)] for u in uniq])
    x = zca_by_subject(eeg, subjects)
    X0, img, subj = aggregate(x, targets, subjects, "image")
    X1, _, _ = aggregate(x * ZCA_SCALE, targets, subjects, "image")
    position = {int(t): i for i, t in enumerate(uniq)}; local = np.array([position[int(t)] for t in img])
    cells = {"ridge_eeg": nested_cv(X0, img, true32, uniq),
             "things_only": f_T(torch.from_numpy(X1).float()).numpy()}
    cells["prior_ridge"], chosen = nested_prior_cv(X1, img, true32, uniq, f_T)
    results, values = {}, {}
    for name, pred in cells.items():
        results[name], values[name] = score(torch.from_numpy(pred.reshape(-1, 3, 32, 32)).float(), true32, local, subj)
        r = results[name]; print(f"{name:12s} coarse8 {r['coarse8']:+.4f} p {r['permutation_p']:.4f} pair {r['pairwise_id']:.3f}")
    paired = {n: paired_bootstrap(values[n], values["ridge_eeg"]) for n in ("things_only", "prior_ridge")}
    print("prior alpha per fold", chosen, "| paired vs ridge_eeg", paired)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "things_prior.json").write_text(json.dumps({"results": results, "paired_vs_ridge_eeg": paired,
                                                           "things_alpha": t_alpha, "things_test_scores": t_scores,
                                                           "prior_alpha_per_fold": chosen}, indent=2))


if __name__ == "__main__":
    main()

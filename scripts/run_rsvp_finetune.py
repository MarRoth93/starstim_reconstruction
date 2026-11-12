#!/usr/bin/env python3
"""The THINGS-pretrained pixel network on all usable rapid-stream recordings: as it is, linear probes, fine-tuning.

    python scripts/run_rsvp_finetune.py [--recompute]

Input and scoring as scripts/run_rsvp_pooled.py: the usable recordings whitened per recording and averaged per image
(reconstruct.rsvp_pooled, 31 channels), one decoder over all images in the same 8 image folds, without and with the
0.1-30 Hz band-pass. The network is starstim_recon/things_network.py (data/things_pixel_network/best.pt); the lab data
enter through its identity adapter slot. Variants and regime fixed on 2026-11-12, before the first run:
  ridge_own       the pooled own decoder (reference; own_decoder of run_rsvp_pooled.py)
  things_prior    the pooled linear THINGS prior (reference; its THINGS ridge is fitted without any image shown in the lab)
  net_zeroshot    the network as it is
  probe_hidden    ridge (nested_cv) on the frozen network's hidden features     probe_temporal   on its temporal features
  ft_adapter      fine-tune the lab adapter slot (lr 1e-3)                      ft_temporal      + the temporal layer (lr 1e-4)
  ft_full         + every other layer (lr 1e-5)
  ft_full_rec     as ft_full, trained on every recording's image means instead of their average (more, noisier rows)
Fine-tuning (the regime of R4): MSE, AdamW without weight decay, 40 epochs, batch 32, the last epoch, network in eval mode
(no dropout), a fresh copy of the pretrained network per fold, torch.manual_seed(1000 * seed + fold) with seed 0.
Added after the first run: ft_full with seeds 0-4 (each seed, and ft_full_ensemble = the mean of their predictions), and
the split of every score by the network's pretraining set.

Leakage: the network was pretrained on the THINGS-EEG2 training images with their pixel targets, and only the 92 old-lab
images were left out. Most rapid-stream images are therefore images it was already trained to reconstruct (from other
people's EEG); the linear THINGS readouts do not have this, they are fitted without the lab images. Every readout is
scored on all images, on the images of the network's pretraining set, and on the images it never saw (candidates = that
subset; the images of its validation split are in neither). Only the never-seen score is free of leakage, and a clean
test on all images needs a network pretrained without them.
Output: results/rsvp_finetune.json (the scores, and ft_full_ensemble against ridge_own and things_prior by paired bootstrap).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import reconstruct, things_network  # noqa: E402
from starstim_recon.data_sessions import rsvp_usable, session_ids  # noqa: E402
from starstim_recon.images import pixel32_many  # noqa: E402
from starstim_recon.metrics import paired_bootstrap, score  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import nested_cv, nested_prior_cv  # noqa: E402

ZCA_SCALE = 1.1                                      # the THINGS lab-contract cache is ZCA-whitened x1.1
FILTERS = {"none": None, "0.1-30 Hz": (0.1, 30.0)}
DEV = torch.device("cuda")
EPOCHS, BATCH, FOLDS, SEEDS = 40, 32, 8, (0, 1, 2, 3, 4)
LR = {"adapter": 1e-3, "temporal": 1e-4, "rest": 1e-5}
SCOPES = {"ft_adapter": ("adapter",), "ft_temporal": ("adapter", "temporal"), "ft_full": ("adapter", "temporal", "rest")}


def lab_slot(model, n):
    return torch.full((n,), model.adapter.identity_slot, dtype=torch.long, device=DEV)


def run(model, x):
    with torch.inference_mode():
        return torch.cat([model(x[a:a + 128].to(DEV), lab_slot(model, len(x[a:a + 128]))).cpu() for a in range(0, len(x), 128)])


def features(model, x, layer):
    with torch.inference_mode():
        h = model.norm(model.temporal(model.adapter(x.to(DEV), lab_slot(model, len(x)))).flatten(1))
        if layer == "hidden":
            h = model.activation(model.hidden(h))
    return h.cpu().numpy().astype(np.float64)


def finetune(x_train, y_train, img_train, x_test, fold, scope, seed=0):
    """Out-of-fold predictions for the test rows (one per image); every training row carries its image index."""
    oof = torch.zeros(len(x_test), 3, 32, 32)
    for k in range(FOLDS):
        torch.manual_seed(1000 * seed + k); model = things_network.load(DEV)
        for q in model.parameters():
            q.requires_grad_(False)
        named = {"adapter": list(model.adapter.parameters()), "temporal": list(model.temporal.parameters())}
        taken = {id(q) for v in named.values() for q in v}; named["rest"] = [q for q in model.parameters() if id(q) not in taken]
        groups = [{"params": named[g], "lr": LR[g]} for g in scope]
        for g in groups:
            for q in g["params"]:
                q.requires_grad_(True)
        opt = AdamW(groups, weight_decay=0.0)
        tr = np.flatnonzero(fold[img_train] != k); te = np.flatnonzero(fold == k)
        for _ in range(EPOCHS):
            order = tr[torch.randperm(len(tr)).numpy()]
            for a in range(0, len(order), BATCH):
                i = order[a:a + BATCH]
                loss = F.mse_loss(model(x_train[i].to(DEV), lab_slot(model, len(i))), y_train[i].to(DEV))
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        oof[te] = run(model, x_test[te]); del model, opt
    return oof


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--recompute", action="store_true", help="rebuild the pooled recordings instead of reading cache/")
    args = ap.parse_args()
    sessions = [s for s in session_ids(rsvp=True) if rsvp_usable(s)]
    trained, validation = things_network.pretraining_images()
    out = {"sessions": sessions, "seeds": list(SEEDS), "results": {}, "ft_full_ensemble_vs": {}}
    for fname, bandpass in FILTERS.items():
        Z = reconstruct.rsvp_pooled(sessions, bandpass, args.recompute); avg, names = Z["avg"], Z["names"]
        true = pixel32_many(Z["paths"]); n = len(names); idx = np.arange(n)
        perm = np.random.default_rng(0).permutation(n); fold = np.empty(n, int); fold[perm] = np.arange(n) % FOLDS   # nested_cv's
        subsets = {"all": idx, "pretraining_images": np.flatnonzero([nm in trained for nm in names]),
                   "never_seen": np.flatnonzero([nm not in trained and nm not in validation for nm in names])}
        X = avg.flatten(1).numpy().astype(np.float64); xs = avg * ZCA_SCALE
        f_T = reconstruct.things_readout(exclude_basenames=set(names)); model = things_network.load(DEV)
        P = {"ridge_own": nested_cv(X, idx, true, idx), "things_prior": nested_prior_cv(X * ZCA_SCALE, idx, true, idx, f_T)[0],
             "net_zeroshot": run(model, xs)}
        for layer in ("hidden", "temporal"):                      # features rescaled to the EEG's total variance: the same alpha grid
            f = features(model, xs, layer); P[f"probe_{layer}"] = nested_cv(f * np.sqrt(X.var(0).sum() / f.var(0).sum()), idx, true, idx)
        del model
        for name in ("ft_adapter", "ft_temporal"):
            P[name] = finetune(xs, true, idx, xs, fold, SCOPES[name])
        for seed in SEEDS:
            P[f"ft_full_seed{seed}"] = finetune(xs, true, idx, xs, fold, SCOPES["ft_full"], seed)
        P["ft_full_ensemble"] = torch.stack([P[f"ft_full_seed{seed}"] for seed in SEEDS]).mean(0)
        P["ft_full_rec"] = finetune(Z["rec_x"] * ZCA_SCALE, true[Z["rec_img"]], Z["rec_img"], xs, fold, SCOPES["ft_full"])
        print(f"\nfilter {fname}: {len(sessions)} recordings, {n} images, {len(Z['rec_img'])} recording x image rows; in the network's "
              f"pretraining set {len(subsets['pretraining_images'])}, in its validation split {sum(nm in validation for nm in names)}, "
              f"never seen {len(subsets['never_seen'])}", flush=True)
        out.update(n_images=n, image_subsets={k: len(v) for k, v in subsets.items()}); out["results"][fname] = {}; vals = {}
        for name, p in P.items():
            p = torch.as_tensor(np.asarray(p).reshape(-1, 3, 32, 32)).float(); r = {}
            for key, sub in subsets.items():
                r[key], vals[name, key] = score(p[sub], true[sub], np.arange(len(sub)), np.zeros(len(sub)), args.n_perm)
            out["results"][fname][name] = r
            print(f"  {name:17s} " + " | ".join(f"{key.replace('_', ' ')} {r[key]['coarse8']:+.4f} (p {r[key]['permutation_p']:.4f})"
                                                  for key in subsets), flush=True)
        out["ft_full_ensemble_vs"][fname] = {ref: {key: paired_bootstrap(vals["ft_full_ensemble", key], vals[ref, key]) for key in subsets}
                                             for ref in ("ridge_own", "things_prior")}
        for ref, d in out["ft_full_ensemble_vs"][fname].items():
            print(f"  ft_full_ensemble - {ref:13s} " + " | ".join(f"{key.replace('_', ' ')} {v['mean_diff']:+.4f} [{v['ci95'][0]:+.3f}, "
                                                                      f"{v['ci95'][1]:+.3f}]" for key, v in d.items()), flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "rsvp_finetune.json").write_text(json.dumps(out, indent=2))
    print("wrote", RESULTS / "rsvp_finetune.json")


if __name__ == "__main__":
    main()

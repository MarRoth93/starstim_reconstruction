#!/usr/bin/env python3
"""Low-level reconstruction from a rapid-stream block (5 images per second, no stimulation), e.g. E21: 144 images x 20.

    python scripts/run_rsvp_reconstruction.py --session E21

Fixed before the first run (2026-11-09). Input: data_sessions.load_rsvp (THINGS lab contract, 31 channels, 0-1 s, 250 Hz),
ZCA per session (label-free), each image averaged over its kept repeats. Scored per image against the session's own images
(coarse8, permutation null over image labels, pairwise identification). Readouts:
  things_zeroshot  linear THINGS ridge (alpha 1e-3, the infographic's), the session's images dropped from the THINGS fit
  within_person    ridge on this session only, 8 outer image folds, alpha in 4 inner folds (R1 protocol)
  things_prior     within_person shrunk toward the THINGS ridge weights (alpha grid up to 1e9 = THINGS alone)
  old_lab          the old-lab ridge (25 shared channels, image averages), zero-shot
Repeat curve: test-image averages over k = 2, 5, 10 random repeats (20 draws each; training inputs unchanged).
Output: results/rsvp_<session>.json and results/rsvp_<session>_examples.png (8 random images, seed 0).
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import reconstruct  # noqa: E402
from starstim_recon.channels import SHARED_25, STARSTIM_31  # noqa: E402
from starstim_recon.data_sessions import load_rsvp  # noqa: E402
from starstim_recon.images import enhanced, photo, pixel32_many, to8  # noqa: E402
from starstim_recon.metrics import coarse8, paired_bootstrap, score  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402
from starstim_recon.readout import nested_cv, nested_prior_cv, zca_by_subject  # noqa: E402

ZCA_SCALE = 1.1                                      # the THINGS lab-contract cache is ZCA-whitened x1.1
DOSE = (2, 5, 10)


def image_means(x, trig, uniq, k=None, rng=None):
    out = []
    for u in uniq:
        rows = np.flatnonzero(trig == u)
        if k is not None and len(rows) > k:
            rows = rng.choice(rows, k, replace=False)
        out.append(x[rows].mean(0))
    return torch.stack(out).flatten(1).numpy().astype(np.float64)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", default="E21"); ap.add_argument("--n-perm", type=int, default=2000)
    args = ap.parse_args()
    R = load_rsvp(args.session)
    tr = R["tr"]; trig = tr.trigger.values.astype(int); uniq = np.array(sorted(set(trig)))
    reps = np.array([np.sum(trig == u) for u in uniq])
    print(f"{args.session} rapid stream: {len(tr)} of {R['n_all']} trials kept, {len(uniq)} images, repeats {reps.min()}-{reps.max()}; "
          f"flat {R['flat']} bad {R['bad']}")
    names = {R["files"][int(u)].name for u in uniq}
    true32 = pixel32_many([R["files"][int(u)] for u in uniq]); local = np.arange(len(uniq)); groups = np.zeros(len(uniq))

    x = zca_by_subject(R["ep"], np.array(["session"] * len(tr)))                   # 31 channels, unscaled
    X0 = image_means(x, trig, uniq); X1 = X0 * ZCA_SCALE
    ci = [STARSTIM_31.index(c) for c in SHARED_25]
    e25 = R["ep"][:, ci] - R["ep"][:, ci].mean(1, keepdim=True)
    x25 = zca_by_subject(e25, np.array(["session"] * len(tr))); X25 = image_means(x25, trig, uniq)

    f_T = reconstruct.things_readout(exclude_basenames=names)
    things_np = lambda X: f_T(torch.from_numpy(X).float()).numpy()
    old_predict, old_alpha = reconstruct.old_lab_readout(exclude_basenames=names)
    preds = {"things_zeroshot": things_np(X1),
             "within_person": nested_cv(X0, uniq, true32, uniq)}
    preds["things_prior"], prior_alphas = nested_prior_cv(X1, uniq, true32, uniq, f_T)
    preds["old_lab"] = old_predict(X25)

    results, values = {}, {}
    for name, p in preds.items():
        results[name], values[name] = score(torch.from_numpy(np.asarray(p).reshape(-1, 3, 32, 32)).float(), true32, local, groups, args.n_perm)
        r = results[name]
        print(f"{name:16s} coarse8 {r['coarse8']:+.4f}  p {r['permutation_p']:.4f}  pairwise id {r['pairwise_id']:.3f}")
    paired = {n: paired_bootstrap(values[n], values["within_person"]) for n in preds if n != "within_person"}
    for n, d in paired.items():
        print(f"{n:16s} vs within_person: diff {d['mean_diff']:+.4f}  95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]")
    print("things_prior alpha per fold:", prior_alphas, "| old-lab alpha", old_alpha)

    rng = np.random.default_rng(1); dose = {}
    for k in DOSE:
        v = {"things_zeroshot": [], "within_person": []}
        for _ in range(20):
            Xk = image_means(x, trig, uniq, k, rng)
            for name, p in (("things_zeroshot", things_np(Xk * ZCA_SCALE)), ("within_person", nested_cv(X0, uniq, true32, uniq, X_eval=Xk))):
                v[name].append(float(coarse8(torch.from_numpy(p.reshape(-1, 3, 32, 32)).float(), true32, local)[0].mean()))
        dose[k] = {n: round(float(np.mean(s)), 4) for n, s in v.items()}
    dose["all"] = {n: round(results[n]["coarse8"], 4) for n in ("things_zeroshot", "within_person")}
    print("repeat curve (mean coarse8 over 20 draws):", dose)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"rsvp_{args.session}.json").write_text(json.dumps(
        {"session": args.session, "n_trials": int(len(tr)), "n_images": int(len(uniq)), "repeats": [int(reps.min()), int(reps.max())],
         "results": results, "paired_vs_within_person": paired, "things_prior_alpha_per_fold": prior_alphas,
         "repeat_curve": {str(k): v for k, v in dose.items()}}, indent=2))

    pick = np.sort(np.random.default_rng(0).choice(len(uniq), 8, replace=False))
    t8 = to8(true32); rows = [("photo", [photo(R["files"][int(uniq[i])]) for i in pick]), ("true 8x8", t8[pick])]
    for name in ("things_zeroshot", "within_person", "things_prior"):
        p8 = to8(torch.from_numpy(np.asarray(preds[name]).reshape(-1, 3, 32, 32)).float())
        rows.append((name.replace("_", " "), enhanced(p8, t8, groups)[pick], values[name][pick]))
    fig, axs = plt.subplots(len(rows), 8, figsize=(13, 1.75 * len(rows)))
    for r, row in enumerate(rows):
        for j in range(8):
            ax = axs[r, j]; ax.imshow(row[1][j], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(row[0], fontsize=9)
            if len(row) == 3:
                ax.set_title(f"{row[2][j]:+.2f}", fontsize=8, pad=2)
    fig.suptitle(f"{args.session} rapid stream: reconstructions (contrast-enhanced), number = coarse8 of that image", fontsize=11)
    fig.tight_layout(); fig.savefig(RESULTS / f"rsvp_{args.session}_examples.png", dpi=110)
    print("wrote", RESULTS / f"rsvp_{args.session}.json")


if __name__ == "__main__":
    main()

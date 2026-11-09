#!/usr/bin/env python3
"""Infographic: low-level reconstructions on THINGS-EEG2, the old lab data and the lab sessions, and what limits the lab data.

    python scripts/make_infographic.py              # reuses cache/infographic_data.pt if present
    python scripts/make_infographic.py --recompute  # after adding sessions to configs/sessions.yaml

Examples: 8 per dataset drawn at random (seed 0). Reconstructions are contrast-enhanced for display: set mean of the
targets + (prediction - set mean of the predictions) x k, one k per dataset matching the spread of the targets.
Output: results/infographic_lowlevel.{png,pdf} and results/infographic_numbers.json.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import reconstruct  # noqa: E402
from starstim_recon.data_sessions import session_ids  # noqa: E402
from starstim_recon.images import photo, to8  # noqa: E402
from starstim_recon.paths import CACHE, RESULTS  # noqa: E402

COL = {"things": "#2b6cb0", "old": "#c05621", "new": "#9b2c2c"}
INFO = {
    "things": ("THINGS-EEG2", "public benchmark",
               ["Research-grade amplifier, 31 StarStim channels used", "10 people, rapid series of 5 images per second",
                "Readout trained on 16,448 images", "Each test image: 80 repeats x 10 people"], "800 trials / image"),
    "old": ("Old lab data", "May-June 2026",
            ["StarStim 32 consumer cap, 31 channels", "6 people, single images, ~4 s each",
             "Readout trained on these data (8-fold image cross-validation)",
             "~1.25 trials per image per person, pooled over 6 people"], "~7 trials / image"),
    "new": ("New lab sessions", "memorability_stim E06-E20",
            ["StarStim 32 with brain stimulation, 25 usable channels", "5 people over 15 sessions, single images, 2 s each",
             "Readout: the old-lab ridge (nothing better transfers)", "Pulse-free blocks only"], "2-4 trials / image"),
}
PROBLEMS = [
    ("Too few repeats per image",
     "Single trials are at chance in every lab dataset; the signal only appears after averaging: 800 trials per THINGS test "
     "image vs ~7 (old lab) and 2-4 (new sessions)."),
    ("Too little data per person",
     "Decoding scales with trials per person (~5,000 needed, Banville et al. 2025). Lab people give ~100-200, and the "
     "readout must work on people it never saw."),
    ("THINGS does not transfer",
     "Rapid 5-per-second series vs single images, research amplifier vs consumer cap, other people: the THINGS network, "
     "its features, a fine-tuned adapter and a THINGS-weight prior all fail on lab data."),
    ("Stimulation costs signal",
     "Six stimulation electrodes are excluded (PO3/PO4 among them), P8/O2/FC2/Cz go flat in pulse runs, and pulse "
     "artifacts leave only the pulse-free blocks."),
]
FIXES = [
    "A dedicated block before any stimulation: ~40 shared images x 30-40 repeats, shown rapidly (~200 ms image + "
    "200-400 ms blank), about 15 minutes.",
    "The same few people across sessions, so each person accumulates data.",
    "Linear readouts per person, averaged as predictions; target the coarse layout (top-bottom lightness is the most "
    "decodable property).",
]


def load_data(recompute: bool):
    """Reconstructions are computed once and cached, so layout changes do not recompute them."""
    cache = CACHE / "infographic_data.pt"
    if cache.exists() and not recompute:
        return torch.load(cache, weights_only=False)
    D = {"things": reconstruct.things(), "old": reconstruct.old_lab(),
         "new": reconstruct.sessions(session_ids(kind="stim"))}
    CACHE.mkdir(parents=True, exist_ok=True); torch.save(D, cache)
    return D


def enhanced(pred8, true8, groups):
    dp, dt, base = np.zeros_like(pred8), np.zeros_like(true8), np.zeros_like(true8)
    for g in np.unique(groups):
        m = groups == g
        dp[m] = pred8[m] - pred8[m].mean(0); dt[m] = true8[m] - true8[m].mean(0); base[m] = true8[m].mean(0)
    return np.clip(base + dp * (dt.std() / (dp.std() + 1e-12)), 0, 1)


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def box(ax, xy, w, h, color, alpha=0.08):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.008,rounding_size=0.012", transform=ax.transAxes,
                                facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--recompute", action="store_true"); args = ap.parse_args()
    D = load_data(args.recompute)
    numbers = {k: {"n_images": int(len(d["vals"])), "coarse8": round(float(np.mean(d["vals"])), 4)} for k, d in D.items()}
    print(json.dumps(numbers))
    fig = plt.figure(figsize=(16, 26), facecolor="white")
    fig.text(0.04, 0.985, "Low-level EEG image reconstruction: THINGS vs. our lab data", fontsize=25, weight="bold", va="top")
    fig.text(0.04, 0.966, "Target: the coarse 8x8 colour layout of the image (no detail, no semantics). Linear ridge readouts. "
             "8 examples per dataset, drawn at random.\nScore = coarse8: correlation of the reconstructed 8x8 layout with the "
             "true one, relative to the other images of the set (0 = chance, 1 = perfect).", fontsize=12.5, va="top", color="#333",
             linespacing=1.5)
    for (key, d), top in zip(D.items(), [0.925, 0.715, 0.505]):
        title, sub, facts, badge = INFO[key]; c = COL[key]
        p8, t8 = to8(d["pred"]), to8(d["true"]); disp = enhanced(p8, t8, d["groups"])
        pick = np.sort(np.random.default_rng(0).choice(len(d["vals"]), 8, replace=False))
        info = fig.add_axes([0.03, top - 0.185, 0.25, 0.185]); info.axis("off"); box(info, (0, 0), 1, 1, c)
        info.text(0.06, 0.93, title, fontsize=17, weight="bold", color=c, va="top", transform=info.transAxes)
        info.text(0.06, 0.82, sub, fontsize=11, color="#555", va="top", transform=info.transAxes)
        body = "\n".join("• " + wrap(f, 40).replace("\n", "\n   ") for f in facts)
        info.text(0.06, 0.70, body, fontsize=10.5, va="top", linespacing=1.45, transform=info.transAxes)
        info.text(0.06, 0.13, f"coarse8 {np.mean(d['vals']):+.3f}", fontsize=19, weight="bold", color=c, transform=info.transAxes)
        info.text(0.06, 0.05, badge, fontsize=11, color="#333", transform=info.transAxes)
        rows = [("photo", [photo(d["paths"][i]) for i in pick]), ("true 8x8 layout", t8[pick]), ("reconstruction", disp[pick])]
        for r, (label, ims) in enumerate(rows):
            for j in range(8):
                ax = fig.add_axes([0.305 + j * 0.0855, top - 0.06 * (r + 1) - 0.004 * r, 0.078, 0.056])
                ax.imshow(ims[j], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values():
                    sp.set_color("#aaa")
                if j == 0:
                    ax.set_ylabel(label, fontsize=10)
                if r == 2:
                    ax.set_xlabel(f"{d['vals'][pick[j]]:+.2f}", fontsize=10.5, color=c)
    keys = list(D); names = ["THINGS", "old lab", "new lab"]; cols = [COL[k] for k in keys]
    axt = fig.add_axes([0.07, 0.2, 0.22, 0.07])
    axt.barh(names[::-1], [3, 7, 800], color=cols[::-1]); axt.set_xscale("log")
    axt.set_title("Trials averaged per image (log scale)", fontsize=12, loc="left")
    axt.spines[["top", "right"]].set_visible(False); axt.set_xlim(1, 3000)
    for i, (v, lab) in enumerate([(3, "2-4"), (7, "~7"), (800, "800")]):
        axt.text(v * 1.2, i, lab, va="center", fontsize=10.5)
    axb = fig.add_axes([0.07, 0.035, 0.22, 0.12])
    scores = [float(np.mean(D[k]["vals"])) for k in keys]
    axb.bar(names, scores, color=cols); axb.axhline(0, color="k", lw=0.8); axb.set_ylim(-0.08, 0.52)
    for i, v in enumerate(scores):
        axb.text(i, v + (0.015 if v >= 0 else -0.05), f"{v:+.3f}", ha="center", fontsize=11, weight="bold")
    axb.set_title("Reconstruction score (coarse8)", fontsize=12, loc="left"); axb.spines[["top", "right"]].set_visible(False)
    pr = fig.add_axes([0.34, 0.125, 0.63, 0.165]); pr.axis("off"); box(pr, (0, 0), 1, 1, "#742a2a", 0.06)
    pr.text(0.025, 0.95, "Why the lab reconstructions fail", fontsize=16, weight="bold", color="#742a2a", va="top")
    for i, (h, t) in enumerate(PROBLEMS):
        x0, y0 = 0.025 + (i % 2) * 0.5, 0.74 - (i // 2) * 0.4
        pr.text(x0, y0, f"{i + 1}  {h}", fontsize=12.5, weight="bold", va="top")
        pr.text(x0 + 0.025, y0 - 0.09, wrap(t, 56), fontsize=10.5, va="top", linespacing=1.4, color="#222")
    fx = fig.add_axes([0.34, 0.03, 0.63, 0.08]); fx.axis("off"); box(fx, (0, 0), 1, 1, "#276749", 0.07)
    fx.text(0.025, 0.88, "What would fix it", fontsize=16, weight="bold", color="#276749", va="top")
    fx.text(0.025, 0.58, "\n".join("• " + wrap(f, 118).replace("\n", "\n   ") for f in FIXES), fontsize=10.5,
            va="top", linespacing=1.45)
    fig.text(0.04, 0.006, "Reconstructions are contrast-enhanced for display (one gain per dataset). THINGS ridge penalty chosen "
             "on its test images (1 of 10 values).\nOld lab: 8-fold image cross-validation. New sessions: readout trained on the old "
             "lab data, scored within each session.", fontsize=9, color="#555")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "infographic_lowlevel.png"
    fig.savefig(out, dpi=110); fig.savefig(out.with_suffix(".pdf"))
    (RESULTS / "infographic_numbers.json").write_text(json.dumps(numbers, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

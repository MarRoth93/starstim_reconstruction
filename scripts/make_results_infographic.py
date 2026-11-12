#!/usr/bin/env python3
"""Infographic: where the low-level reconstruction of the lab data stands (rapid-stream sessions, timing correction).

    python scripts/make_results_infographic.py

Draws only from result files, so run for every rapid-stream session of configs/sessions.yaml first:
    python scripts/check_rsvp_timing.py --session E<nn>
    python scripts/run_rsvp_reconstruction.py --session E<nn>                 # corrected timing
    python scripts/run_rsvp_reconstruction.py --session E<nn> --offset-ms 0   # time zero at the marker
python scripts/run_rsvp_pooled.py for the decoders on all usable recordings together (results/rsvp_pooled.json),
python scripts/run_rsvp_finetune.py for the fine-tuned THINGS network on the same recordings (results/rsvp_finetune.json), and
python scripts/make_infographic.py for the three reference numbers (results/infographic_numbers.json).
A session counts as usable when its timing check found an image-based estimate above its null and the automatic offset fits.
Examples: 8 images of the usable session with the highest scores, drawn at random (seed 0), contrast-enhanced for display.
Output: results/infographic_results.{png,pdf} and results/infographic_results_numbers.json.
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon.data_sessions import _folder, clock_error, rsvp_display_lag_ms, rsvp_usable, session_ids  # noqa: E402
from starstim_recon.images import enhanced, photo, pixel32_many, to8  # noqa: E402
from starstim_recon.paths import CACHE, RESULTS, THINGS_OBJECT_IMAGES  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OWN, THINGS, PRIOR, NET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # the readouts keep these colours in every panel
OLD_H, SPLIT, EXTRA, FOOT = 21.0, 0.60, 4.2, 0.55                # inches added to the 21-inch layout: a band for the pooled
NEW_H = OLD_H + EXTRA + FOOT                                     # panel, inserted at SPLIT, and taller text boxes at the foot
POOLED = {"own_decoder": ("Own decoder", OWN, "within_person"), "things_zeroshot": ("THINGS decoder", THINGS, "things_zeroshot"),
          "things_prior": ("THINGS as prior", PRIOR, "things_prior")}   # readout of run_rsvp_pooled -> label, colour, per-recording key
READOUT = {"within_person": ("Own decoder (trained on the same recording)", OWN),
           "things_zeroshot": ("THINGS decoder (trained on THINGS-EEG2 only)", THINGS)}


def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width))


def fy(y):
    """Height in the 21-inch layout the panels were placed in -> height in the figure with the band and the foot inserted."""
    return (y * OLD_H + FOOT + (EXTRA if y >= SPLIT else 0)) / NEW_H


def R(rect):
    return [rect[0], fy(rect[1]), rect[2], rect[3] * OLD_H / NEW_H]


def panel(fig, rect, title, subtitle=None):
    ax = fig.add_axes(rect, facecolor=SURFACE)
    ax.set_title(title, fontsize=14, weight="bold", loc="left", color=INK, pad=24 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=10.5, color=INK2, va="bottom")
    ax.spines[["top", "right"]].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=10, length=0); ax.set_axisbelow(True)
    return ax


def load():
    S = {}
    for sid in session_ids(rsvp=True):
        files = {k: RESULTS / f for k, f in (("j", f"rsvp_{sid}.json"), ("marker", f"rsvp_{sid}_offset0.json"),
                                             ("timing", f"rsvp_timing_{sid}.json"), ("pred", f"rsvp_{sid}_predictions.npz"))}
        missing = [str(f) for f in files.values() if not f.exists()]
        assert not missing, f"{sid}: missing {missing} (see the docstring for the commands)"
        s = {k: json.loads(files[k].read_text()) for k in ("j", "marker", "timing")}
        s["pred"] = files["pred"]; s["usable"] = rsvp_usable(sid)
        s["person"] = _folder(sid).name.split("_")[-1]           # only counted, never shown
        S[sid] = s
    return S


def score(s, readout, timing="j"):
    r = s[timing]["results"][readout]
    return r["coarse8"], r["permutation_p"]


def main():
    S = load(); ids = list(S); usable = [i for i in ids if S[i]["usable"]]; unusable = [i for i in ids if not S[i]["usable"]]
    ref = json.loads((RESULTS / "infographic_numbers.json").read_text())
    assert (RESULTS / "rsvp_pooled.json").exists(), "run scripts/run_rsvp_pooled.py first"
    pooled = json.loads((RESULTS / "rsvp_pooled.json").read_text())
    assert pooled["sessions"] == usable, f"results/rsvp_pooled.json covers {pooled['sessions']}, usable are {usable}: rerun it"
    pool = lambda f, c, r: pooled["results"][f][c][r]["all"]   # noqa: E731
    best_pool = pool("0.1-30 Hz", "31 channels", "things_prior")["coarse8"]
    assert (RESULTS / "rsvp_finetune.json").exists(), "run scripts/run_rsvp_finetune.py first"
    tuned = json.loads((RESULTS / "rsvp_finetune.json").read_text())
    assert tuned["sessions"] == usable, f"results/rsvp_finetune.json covers {tuned['sessions']}, usable are {usable}: rerun it"
    net = lambda f, subset="all", r="ft_full_ensemble": tuned["results"][f][r][subset]["coarse8"]   # noqa: E731
    best_net = net("0.1-30 Hz"); n_pre, n_new = tuned["image_subsets"]["pretraining_images"], tuned["image_subsets"]["never_seen"]
    mean = {r: float(np.mean([score(S[i], r)[0] for i in usable])) for r in READOUT}
    n_sig = {r: sum(score(S[i], r)[1] < 0.05 for i in usable) for r in READOUT}
    drift = {}
    for sid in session_ids():
        clock_error(sid); z = np.load(CACHE / f"{sid}_clock.npz")
        drift[sid] = ((z["t"] - z["t"][0]) / 60, z["e"]) if len(z["t"]) else None
    drifting = [i for i, d in drift.items() if d is not None and d[1].any()]
    steady = [i for i, d in drift.items() if d is not None and not d[1].any()]
    numbers = {"sessions": {i: {"usable": S[i]["usable"], "images_x_repeats": [S[i]["j"]["n_images"], S[i]["j"]["repeats"][1]],
                                **{r: dict(zip(("coarse8", "p"), score(S[i], r))) for r in READOUT},
                                **{f"{r}_at_marker": score(S[i], r, "marker")[0] for r in READOUT}} for i in ids},
               "mean_usable": mean, "n_significant_usable": n_sig, "usable": usable, "drifting": drifting, "reference": ref}
    numbers["pooled"] = {f: {c: {r: pool(f, c, r)["coarse8"] for r in POOLED} for c in v} for f, v in pooled["results"].items()}
    numbers["pooled_finetuned_network"] = {f: {subset: {r: net(f, subset, r) for r in ("ft_full_ensemble", "ridge_own", "things_prior")}
                                               for subset in tuned["image_subsets"]} for f in tuned["results"]}
    numbers["pooled_finetuned_network"].update(image_subsets=tuned["image_subsets"], seeds=tuned["seeds"],
                                               minus_things_prior=tuned["ft_full_ensemble_vs"]["0.1-30 Hz"]["things_prior"])
    print(json.dumps(numbers["mean_usable"]), "usable", usable, "drifting", drifting)

    fig = plt.figure(figsize=(15, NEW_H), facecolor="white")
    fig.text(0.04, fy(0.985), "Low-level EEG image reconstruction of our lab data: where we stand", fontsize=24, weight="bold",
             va="top", color=INK)
    fig.text(0.04, fy(0.966), f"Rapid-stream recordings {ids[0]}-{ids[-1]} ({len(ids)} recordings, {len({S[i]['person'] for i in ids})} people): "
             "144 images at 5 per second, StarStim 32, no stimulation during the block. Target: the coarse 8x8 colour layout.\n"
             "Score = coarse8: correlation of the reconstructed layout with the true one, relative to the other images of the "
             "recording (0 = chance, 1 = perfect).", fontsize=11.5, va="top", color=INK2, linespacing=1.5)

    # ---------------------------------------------------------------------------------------------- headline numbers
    tiles = [("THINGS-EEG2 benchmark", f"{ref['things']['coarse8']:+.2f}", "10 people, 800 trials per test image"),
             ("Old lab data, 6 people pooled", f"{ref['old']['coarse8']:+.3f}", "single images, about 7 trials per image"),
             (f"Rapid stream, own decoder", f"{mean['within_person']:+.3f}",
              f"mean of {len(usable)} usable recordings; {n_sig['within_person']} above chance"),
             ("Rapid stream, THINGS decoder", f"{mean['things_zeroshot']:+.3f}",
              f"mean of {len(usable)} usable recordings; {n_sig['things_zeroshot']} above chance")]
    for k, (label, value, note) in enumerate(tiles):
        ax = fig.add_axes(R([0.04 + k * 0.235, 0.876, 0.215, 0.058])); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.03), 0.98, 0.94, boxstyle="round,pad=0,rounding_size=0.05", transform=ax.transAxes,
                                    facecolor=SURFACE, edgecolor=GRID, linewidth=1))
        ax.text(0.06, 0.80, label, fontsize=10.5, color=INK2, va="center", transform=ax.transAxes)
        ax.text(0.06, 0.46, value, fontsize=24, weight="bold", color=INK, va="center", transform=ax.transAxes)
        ax.text(0.06, 0.14, note, fontsize=9, color=MUTED, va="center", transform=ax.transAxes)
        if k >= 2:
            ax.add_patch(plt.Rectangle((0.01, 0.12), 0.014, 0.76, transform=ax.transAxes, color=(OWN, THINGS)[k - 2], lw=0))

    # ---------------------------------------------------------------------------------------------- scores per recording
    ax = panel(fig, R([0.07, 0.655, 0.55, 0.165]), "Reconstruction score per recording, corrected timing",
               "Two ways to decode the same recording. * = above chance (permutation test, p < .05).")
    w = 0.2; top = max(max(score(S[i], r)[0] for i in ids for r in READOUT), ref["old"]["coarse8"]) + 0.03
    for k, (r, (label, col)) in enumerate(READOUT.items()):
        for j, i in enumerate(ids):
            v, p = score(S[i], r); x = j + (k - 0.5) * 0.42
            ax.bar(x, v, w, color=col if S[i]["usable"] else GRID, label=label if j == 0 else None, zorder=3)
            ax.text(x, v + (0.004 if v >= 0 else -0.004), f"{v:+.3f}" + ("*" if p < 0.05 and S[i]["usable"] else ""), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=9, color=INK if S[i]["usable"] else MUTED)
    ax.axhline(0, color=AXIS, lw=1, zorder=2)
    ax.axhline(ref["old"]["coarse8"], color=MUTED, lw=1, zorder=1)
    ax.text(len(ids) - 0.45, ref["old"]["coarse8"] + 0.003, f"old lab data, 6 people pooled: {ref['old']['coarse8']:+.3f}", ha="right",
            fontsize=9.5, color=INK2)
    ax.set_xticks(range(len(ids)))
    ax.set_xticklabels([f"{i}\n{S[i]['j']['n_images']} x {S[i]['j']['repeats'][1]}" if S[i]["usable"] else f"{i}\nnot usable"
                        for i in ids], fontsize=10, color=INK2)
    ax.set_ylim(min(-0.03, min(score(S[i], r)[0] for i in ids for r in READOUT) - 0.02), top); ax.set_xlim(-0.6, len(ids) - 0.4)
    ax.yaxis.grid(True, color=GRID, lw=1); ax.set_ylabel("coarse8", fontsize=10.5, color=INK2)
    ax.legend(loc="upper left", fontsize=10, frameon=False, ncol=2, bbox_to_anchor=(0, 0.995), handlelength=1.2, labelcolor=INK2)
    ax.text(0, -0.23, "Below each recording: images x repeats kept. " + ("Grey: timing could not be verified (" + ", ".join(unusable)
            + ")." if unusable else ""), transform=ax.transAxes, fontsize=9.5, color=MUTED)

    # ---------------------------------------------------------------------------------------------- on the benchmark's scale
    ax = panel(fig, R([0.74, 0.655, 0.22, 0.165]), "On the scale of the benchmark", "The same score, all datasets.")
    rows = [("THINGS-EEG2", ref["things"]["coarse8"], INK2), ("Old lab", ref["old"]["coarse8"], INK2),
            ("Rapid stream,\nown decoder", mean["within_person"], OWN), ("Rapid stream,\nTHINGS decoder", mean["things_zeroshot"], THINGS),
            ("All recordings together,\nTHINGS as prior, filtered", best_pool, PRIOR),
            ("All recordings together,\nfine-tuned network, filtered *", best_net, NET),
            ("Slow trials E06-E20,\nold-lab decoder", ref["new"]["coarse8"], INK2)]
    for k, (label, v, col) in enumerate(rows):
        y = len(rows) - 1 - k; ax.barh(y, v, 0.34, color=col, zorder=3)
        ax.text(max(v, 0) + 0.012, y, f"{v:+.3f}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9, color=INK2)
    ax.axvline(0, color=AXIS, lw=1); ax.set_xlim(-0.03, 0.56); ax.xaxis.grid(True, color=GRID, lw=1); ax.spines["left"].set_visible(False)
    ax.text(-0.5, -0.23, f"* pretrained on {n_pre} of these {pooled['n_images']} images;\n   on the {n_new} it never saw: "
            f"{net('0.1-30 Hz', 'never_seen'):+.3f}", transform=ax.transAxes, fontsize=9.5, color=MUTED, va="center")

    # ---------------------------------------------------------------------------------------------- all recordings together
    single = {r: float(np.mean([score(S[i], POOLED[r][2])[0] for i in usable])) for r in POOLED}
    labels = [v[0] for v in POOLED.values()] + ["Fine-tuned network"]; n_rows = len(labels)   # the network: 31 channels only
    net_note = (f"Fine-tuned network: the pixel network pretrained on THINGS-EEG2, all layers fine-tuned on our images, mean of "
                f"{len(tuned['seeds'])} seeds; it has no single-recording value. It was pretrained on {n_pre} of the {pooled['n_images']} "
                f"images, so its score is not free of leakage: on the {n_new} images it never saw it scores {net('none', 'never_seen'):+.3f} "
                f"without and {net('0.1-30 Hz', 'never_seen'):+.3f} with the band-pass (own decoder on these: "
                f"{net('none', 'never_seen', 'ridge_own'):+.3f} and {net('0.1-30 Hz', 'never_seen', 'ridge_own'):+.3f}).")
    for k, cname in enumerate(("31 channels", "10 posterior")):
        ax = panel(fig, [0.21 + k * 0.33, (SPLIT * OLD_H + FOOT + 1.35) / NEW_H, 0.27, 2.05 / NEW_H],
                   f"All {len(usable)} usable recordings together: averaged per image, one decoder on {pooled['n_images']} images" if k == 0 else " ",
                   "31 channels" if k == 0 else "10 posterior channels")
        for j, (r, (label, col, _)) in enumerate(POOLED.items()):
            y = n_rows - 1 - j; a, b = pool("none", cname, r)["coarse8"], pool("0.1-30 Hz", cname, r)["coarse8"]
            ax.plot([a, b], [y, y], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
            ax.plot(single[r], y, "|", ms=13, mew=2, color=MUTED, zorder=3)
            ax.plot(a, y, "o", ms=8, color=SURFACE, mec=MUTED, mew=1.5, zorder=3)
            ax.plot(b, y, "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=4)
            ax.text(max(a, b) + 0.004, y, f"{b:+.3f}", va="center", fontsize=9.5, color=INK)
        if k == 0:
            a, b = net("none"), net("0.1-30 Hz")
            ax.plot([a, b], [0, 0], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
            ax.plot(a, 0, "o", ms=8, color=SURFACE, mec=MUTED, mew=1.5, zorder=3)
            ax.plot(b, 0, "o", ms=9, color=NET, mec=SURFACE, mew=2, zorder=4)
            ax.text(max(a, b) + 0.004, 0, f"{b:+.3f}", va="center", fontsize=9.5, color=INK)
        else:
            ax.text(0.003, 0, "Fine-tuned network: not run on the posterior channels", va="center", fontsize=9.5, color=MUTED)
        ax.axvline(0, color=AXIS, lw=1, zorder=1); ax.set_xlim(-0.005, 0.105); ax.set_ylim(-0.6, n_rows - 0.4)
        ax.set_yticks(range(n_rows)); ax.set_yticklabels(labels[::-1] if k == 0 else [], fontsize=10, color=INK2)
        ax.xaxis.grid(True, color=GRID, lw=1); ax.spines["left"].set_visible(False); ax.set_xticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
        if k == 0:
            ax.plot([], [], "|", ms=13, mew=2, color=MUTED, label="mean of the single recordings")
            ax.plot([], [], "o", ms=8, color=SURFACE, mec=MUTED, mew=1.5, label="all together, no filter")
            ax.plot([], [], "o", ms=9, color=INK2, mec=SURFACE, mew=2, label="all together, 0.1-30 Hz band-pass")
            ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.165), fontsize=9.5, frameon=False, ncol=3, handletextpad=0.3,
                      columnspacing=1.4, labelcolor=INK2)
            ax.text(0, -0.36, wrap(net_note, 128), transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")

    # ---------------------------------------------------------------------------------------------- the clock
    big = [i for i in drifting if np.abs(drift[i][1]).max() > 500]
    ax = panel(fig, R([0.07, 0.425, 0.40, 0.135]), "The EEG clock drifts in some sessions",
               f"Error of the EEG time stamps against the marker clock, from the stimulation pulses. {len(drifting)} of "
               f"{len(drifting) + len(steady)} sessions drift.")
    for i in steady:
        ax.plot(drift[i][0], drift[i][1], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
    ends = []
    for i in drifting:
        if i in big:
            continue
        t, e = drift[i]
        for seg in np.split(np.arange(len(t)), np.flatnonzero(np.diff(t) > 4) + 1):       # a new recording starts a new line
            ax.plot(t[seg], e[seg], color=INK2, lw=2, zorder=3, solid_capstyle="round")
            if seg[-1] != len(t) - 1:
                ax.text(t[seg[-1]] + 0.5, e[seg[-1]] - 4, f"{i}, first recording", fontsize=9, color=INK2, va="top")
        ax.plot(t[-1], e[-1], "o", ms=7, color=INK2, mec=SURFACE, mew=2, zorder=4); ends.append((e[-1], t[-1], i))
    ends.sort(); y_last = -1e9
    for e, t, i in ends:                                           # end labels, pushed apart where lines end close together
        y = max(e, y_last + 22); y_last = y
        ax.annotate(i, (t, e), (t + 1.6, y), fontsize=9.5, color=INK, va="center",
                    arrowprops=dict(arrowstyle="-", color=AXIS, lw=0.8) if abs(y - e) > 4 else None)
    t_max = max(drift[i][0][-1] for i in drifting + steady if i not in big)
    ax.text(t_max + 3, 14, f"{len(steady)} sessions without drift (grey)", fontsize=9.5, color=INK2, va="bottom", ha="right")
    ax.set_xlim(0, t_max + 7); ax.set_ylim(-340, 60); ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_xlabel("minutes since the first pulse", fontsize=10.5, color=INK2); ax.set_ylabel("clock error (ms)", fontsize=10.5, color=INK2)
    if big:
        ax.text(0, -0.36, wrap(" ".join(f"{i}: the other way round, EEG stamped {drift[i][1][0] / 1000:+.1f} s to {drift[i][1][-1] / 1000:+.1f} s "
                "against the markers (not drawn)." for i in big) + " Negative = an event shows in the EEG before its marker.", 92),
                transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")

    # ---------------------------------------------------------------------------------------------- what the correction does
    lag = rsvp_display_lag_ms()
    for k, (r, (label, col)) in enumerate(READOUT.items()):
        ax = panel(fig, R([0.60 + k * 0.19, 0.425, 0.16, 0.135]), "What the timing correction changes" if k == 0 else " ",
                   "Own decoder" if k == 0 else "THINGS decoder")
        for j, i in enumerate(usable):
            y = len(usable) - 1 - j; a, b = score(S[i], r, "marker")[0], score(S[i], r)[0]
            ax.plot([a, b], [y, y], color=AXIS, lw=2, zorder=2, solid_capstyle="round")
            ax.plot(a, y, "o", ms=8, color=SURFACE, mec=MUTED, mew=1.5, zorder=3)
            ax.plot(b, y, "o", ms=9, color=col, mec=SURFACE, mew=2, zorder=4)
            ax.text(max(a, b) + 0.008, y, f"{b:+.3f}", va="center", fontsize=9.5, color=INK)
        ax.axvline(0, color=AXIS, lw=1, zorder=1); ax.set_xlim(-0.06, 0.135); ax.set_ylim(-0.6, len(usable) - 0.4)
        ax.set_yticks(range(len(usable))); ax.set_yticklabels(usable[::-1] if k == 0 else [], fontsize=10, color=INK2)
        ax.xaxis.grid(True, color=GRID, lw=1); ax.spines["left"].set_visible(False); ax.set_xticks([-0.05, 0, 0.05, 0.1])
        if k == 0:
            ax.plot([], [], "o", ms=8, color=SURFACE, mec=MUTED, mew=1.5, label="time zero at the marker")
            ax.plot([], [], "o", ms=9, color=INK2, mec=SURFACE, mew=2, label="corrected")
            ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.16), fontsize=9.5, frameon=False, ncol=2, handletextpad=0.2,
                      columnspacing=1.2, labelcolor=INK2)
            ax.text(0, -0.38, wrap(f"Corrected = clock error removed, plus {lag:g} ms for the delay of the image on our screen.", 78),
                    transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")

    # ---------------------------------------------------------------------------------------------- examples
    best = max(usable, key=lambda i: sum(score(S[i], r)[0] for r in READOUT)); P = np.load(S[best]["pred"])
    pick = np.sort(np.random.default_rng(0).choice(len(P["image"]), 8, replace=False)); groups = np.zeros(len(P["image"]))
    paths = [THINGS_OBJECT_IMAGES / str(p) for p in P["image"]]; t8 = to8(pixel32_many(paths))
    rows = [("photo", [photo(paths[i]) for i in pick], None), ("true 8x8 layout", t8[pick], None)]
    for r, (label, col) in READOUT.items():
        rows.append((("own decoder" if r == "within_person" else "THINGS decoder"), enhanced(to8(P[r]), t8, groups)[pick],
                     P[f"coarse8_{r}"][pick]))
    fig.text(0.04, fy(0.335), f"What the reconstructions look like: {best}, 8 images drawn at random", fontsize=14, weight="bold", color=INK)
    fig.text(0.04, fy(0.321), "Contrast-enhanced for display. The number above a reconstruction is the coarse8 of that image.",
             fontsize=10.5, color=INK2)
    for q, (label, ims, vals) in enumerate(rows):
        for j in range(8):
            ax = fig.add_axes(R([0.16 + j * 0.101, 0.266 - q * 0.0525, 0.062, 0.0443]))
            ax.imshow(ims[j], interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_color(AXIS)
            if vals is not None:
                ax.set_title(f"{vals[j]:+.2f}", fontsize=8.5, color=INK2, pad=1.5)
            if j == 0:
                ax.text(-0.25, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=10.5, color=INK2)

    # ---------------------------------------------------------------------------------------------- what it means
    best_things = max(usable, key=lambda i: score(S[i], "things_zeroshot")[0])
    new = [f"{len(ids)} rapid-stream recordings from {len({S[i]['person'] for i in ids})} people; {len(usable)} are usable, with "
           f"{pooled['n_images']} distinct images. Averaged per image and decoded together they reach {best_pool:+.3f} with a linear "
           f"readout and {best_net:+.3f} with a THINGS-pretrained network fine-tuned on them.",
           f"The EEG clock runs off the marker clock in {len(drifting)} of {len(drifting) + len(steady)} sessions. Both loaders now "
           "measure this from the stimulation pulses and correct it, in the same way as memorability_stim.",
           f"With corrected timing a decoder trained only on THINGS-EEG2 scores above chance in {n_sig['things_zeroshot']} of "
           f"{len(usable)} recordings without any lab training data (best: {best_things}, {score(S[best_things], 'things_zeroshot')[0]:+.3f})."]
    vs = tuned["ft_full_ensemble_vs"]["0.1-30 Hz"]["things_prior"]["all"]
    limits = [f"Together the recordings reach about {best_pool / ref['things']['coarse8']:.0%} of the benchmark with a linear readout and "
              f"{best_net / ref['things']['coarse8']:.0%} with the fine-tuned network ({best_pool:+.2f} and {best_net:+.2f} against "
              f"{ref['things']['coarse8']:+.2f}): {pooled['n_images']} distinct images against 16,540 in THINGS-EEG2. Every recording shows "
              "one of two image sets, so new recordings add repeats, not images.",
              f"The fine-tuned network is not reliably better than the linear readouts ({vs['mean_diff']:+.3f}, 95% interval "
              f"{vs['ci95'][0]:+.3f} to {vs['ci95'][1]:+.3f}, against THINGS as prior), and it was pretrained on {n_pre} of the "
              f"{pooled['n_images']} images. On the {n_new} it never saw it scores {net('0.1-30 Hz', 'never_seen'):+.3f}, the own decoder "
              f"{net('0.1-30 Hz', 'never_seen', 'ridge_own'):+.3f}.",
              f"A single recording decodes weakly ({mean['within_person']:+.3f} with its own decoder), and the old-lab decoder does not "
              f"transfer to the slow image trials ({ref['new']['coarse8']:+.3f}).",
              "The timing is inferred from the data (pulses and images), not yet confirmed with a photodiode."
              + (f" {', '.join(unusable)} could not be used." if unusable else "")]
    for k, (title, items) in enumerate((("What is new", new), ("What still limits the reconstruction", limits))):
        ax = fig.add_axes([0.04 + k * 0.47, 0.008 * OLD_H / NEW_H, 0.45, (0.094 * OLD_H + FOOT) / NEW_H]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.005, 0.02), 0.99, 0.96, boxstyle="round,pad=0,rounding_size=0.03", transform=ax.transAxes,
                                    facecolor=SURFACE, edgecolor=GRID, linewidth=1))
        ax.text(0.025, 0.92, title, fontsize=13, weight="bold", color=INK, va="top", transform=ax.transAxes)
        ax.text(0.025, 0.77, "\n".join("• " + wrap(t, 88).replace("\n", "\n   ") for t in items), fontsize=9.5, color=INK2, va="top",
                linespacing=1.4, transform=ax.transAxes)

    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "infographic_results.png"
    fig.savefig(out, dpi=110); fig.savefig(out.with_suffix(".pdf"))
    (RESULTS / "infographic_results_numbers.json").write_text(json.dumps(numbers, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Every reconstruction as an image: one sheet per rapid-stream recording and per filter for all recordings together.

    python scripts/make_reconstruction_sheets.py

Draws only from the saved predictions, so run first
    python scripts/run_rsvp_reconstruction.py --session E<nn>   # for every rapid-stream session: results/rsvp_<session>_predictions.npz
    python scripts/run_rsvp_pooled.py                            # results/rsvp_pooled[_bp0.1-30]_predictions.npz
    python scripts/run_rsvp_finetune.py                          # results/rsvp_finetune[_bp0.1-30]_predictions.npz
Each image is one block: its true 8x8 colour layout, then the reconstruction of every readout, with the coarse8 of that image
below it (against the same candidates as the reported score, so the mean of a column is the reported score). Images in
alphabetical order. The reconstructions are contrast-enhanced for display (images.enhanced: one linear stretch per readout
and sheet, as in the infographic); the scores are computed on the unchanged predictions. No photos: the THINGS images are
not ours to redistribute, the file name identifies each one.
Output: results/sheets/rsvp_<session>.png, results/sheets/all_recordings[_bp0.1-30].png.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon.data_sessions import rsvp_usable, session_ids  # noqa: E402
from starstim_recon.images import enhanced, pixel32_many, to8  # noqa: E402
from starstim_recon.paths import RESULTS, THINGS_OBJECT_IMAGES  # noqa: E402

INK, INK2, MUTED, SURFACE = "#0b0b0b", "#52514e", "#898781", "#fcfcfb"   # as in make_results_infographic.py
OWN, THINGS, PRIOR, NET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CELL, GAP, BLOCK_GAP, ROW_GAP, COLS, MARGIN, HEAD = 80, 6, 30, 14, 4, 40, 146   # pixels; an 8x8 layout is drawn 80 x 80
LINE_H = 22                                                                      # one line of the header text, in pixels
LABEL_H, SCORE_H = 24, 22
PER_RECORDING = (("within_person", "Own decoder", OWN), ("things_zeroshot", "THINGS decoder", THINGS),
                 ("things_prior", "THINGS as prior", PRIOR), ("old_lab", "Old-lab decoder", INK2))
POOLED = (("own_decoder", "Own decoder", OWN), ("things_zeroshot", "THINGS decoder", THINGS),
          ("things_prior", "THINGS as prior", PRIOR))


def render(path, title, lines, images, true32, readouts, star=None, star_note=""):
    """readouts: [(label, colour, predictions (n, 3, 32, 32), coarse8 (n,))]; star: mark the last readout's score of these images."""
    n = len(images); order = np.argsort(images); t8 = to8(true32); groups = np.zeros(n); head = HEAD + LINE_H * len(lines)
    shown = [enhanced(to8(p), t8, groups) for _, _, p, _ in readouts]
    k = 1 + len(readouts); bw = k * CELL + (k - 1) * GAP; bh = LABEL_H + CELL + SCORE_H; rows = -(-n // COLS)
    W = 2 * MARGIN + COLS * bw + (COLS - 1) * BLOCK_GAP; H = head + rows * (bh + ROW_GAP) + MARGIN
    canvas = np.ones((H, W, 3)) * np.array(to_rgb(SURFACE)); texts = []
    for slot, i in enumerate(order):
        x0 = MARGIN + (slot % COLS) * (bw + BLOCK_GAP); y0 = head + (slot // COLS) * (bh + ROW_GAP); y = y0 + LABEL_H
        texts.append((x0, y0 + LABEL_H - 6, Path(images[i]).stem, 9, INK2))
        for j, img in enumerate([t8[i]] + [s[i] for s in shown]):
            x = x0 + j * (CELL + GAP)
            canvas[y:y + CELL, x:x + CELL] = np.repeat(np.repeat(img, CELL // 8, 0), CELL // 8, 1)
            if j:
                label, col, _, v = readouts[j - 1]
                canvas[y + CELL + 2:y + CELL + 5, x:x + CELL] = to_rgb(col)            # which readout, in its colour
                mark = "*" if star is not None and j == k - 1 and star[i] else ""
                texts.append((x, y + CELL + SCORE_H - 2, f"{round(float(v[i]), 2) + 0.0:+.2f}{mark}", 9, INK))   # + 0.0: no "-0.00"
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100); ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas, interpolation="nearest"); ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5); ax.axis("off")
    ax.text(MARGIN, MARGIN + 6, title, fontsize=17, weight="bold", color=INK, va="top")
    ax.text(MARGIN, MARGIN + 42, "\n".join(lines), fontsize=10.5, color=INK2, va="top", linespacing=1.5)
    x = MARGIN; yl = head - 34                                                         # legend: the order of the squares in a block
    for label, col in [("true layout", None)] + [(f"{r[0]} (mean {r[3].mean():+.3f})", r[1]) for r in readouts]:
        if col:
            ax.add_patch(plt.Rectangle((x, yl - 8), 22, 5, color=col, lw=0))
        t = ax.text(x + (30 if col else 0), yl, label, fontsize=10.5, color=INK2, va="center")
        fig.canvas.draw(); x = t.get_window_extent().transformed(ax.transData.inverted()).x1 + 34
    if star_note:
        ax.text(MARGIN, head - 12, star_note, fontsize=10, color=MUTED, va="center")
    for x, y, s, size, col in texts:
        ax.text(x, y, s, fontsize=size, color=col)
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=100); plt.close(fig)
    print("wrote", path)


def main():
    out = RESULTS / "sheets"
    for sid in session_ids(rsvp=True):
        f = RESULTS / f"rsvp_{sid}_predictions.npz"
        assert f.exists(), f"{f} missing: python scripts/run_rsvp_reconstruction.py --session {sid}"
        P = np.load(f); J = json.loads((RESULTS / f"rsvp_{sid}.json").read_text()); images = P["image"]; lo, hi = J["repeats"]
        render(out / f"rsvp_{sid}.png", f"{sid}: every image of the rapid-stream block, decoded from this recording alone"
               + ("" if rsvp_usable(sid) else " (NOT USABLE: its timing could not be verified)"),
               [f"{J['n_images']} images x {lo if lo == hi else f'{lo}-{hi}'} repeats, corrected timing, no filter, 31 channels. "
                "Own decoder and THINGS as prior are trained on this recording in 8 image folds;",
                "the THINGS decoder and the old-lab decoder never saw it. Number = coarse8 of that image among the "
                f"{J['n_images']} images of this recording (0 = chance, 1 = perfect). Reconstructions contrast-enhanced for display."],
               images, pixel32_many([THINGS_OBJECT_IMAGES / q for q in images]),
               [(label, col, P[key], P[f"coarse8_{key}"]) for key, label, col in PER_RECORDING])
    for tag, fname in (("", "no filter"), ("_bp0.1-30", "0.1-30 Hz band-pass")):
        f, g = RESULTS / f"rsvp_pooled{tag}_predictions.npz", RESULTS / f"rsvp_finetune{tag}_predictions.npz"
        assert f.exists() and g.exists(), "run scripts/run_rsvp_pooled.py and scripts/run_rsvp_finetune.py first"
        P, Q = np.load(f), np.load(g); images = P["image"]
        assert (Q["image"] == images).all(), f"{f} and {g} list different images: rerun both"
        sessions = json.loads((RESULTS / "rsvp_pooled.json").read_text())["sessions"]
        render(out / f"all_recordings{tag}.png", f"All {len(sessions)} usable recordings together: every image, averaged over its recordings",
               [f"{', '.join(sessions)}: each recording whitened, averaged per image, then the recordings of every image averaged; "
                f"{fname}, 31 channels.",
                "Own decoder, THINGS as prior and the fine-tuned network are trained on these recordings in 8 image folds; the "
                "THINGS decoder never saw them.",
                f"Number = coarse8 of that image among all {len(images)} images (0 = chance, 1 = perfect). "
                "Reconstructions contrast-enhanced for display."],
               images, pixel32_many([THINGS_OBJECT_IMAGES / q for q in images]),
               [(label, col, P[key], P[f"coarse8_{key}"]) for key, label, col in POOLED]
               + [("Fine-tuned network", NET, Q["ft_full_ensemble"], Q["coarse8_ft_full_ensemble"])],
               star=Q["pretraining_image"],
               star_note=f"* the network was pretrained on this image (with its true layout, from other people's EEG): "
                         f"{int(Q['pretraining_image'].sum())} of {len(images)} images; {int(Q['never_seen'].sum())} it never saw.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Timing check for a rapid-stream block: does the automatic offset put time zero at the image onset?

    python scripts/check_rsvp_timing.py --session E23

Run it for every new rapid-stream session. The loader moves every image marker by the session's clock error (measured from
the stimulation pulses, data_sessions.clock_error) plus rsvp_display_lag_ms. This script applies the clock error and then
estimates from the images themselves what is left; nothing has to be entered for a session where that matches the display
lag.

Every image marker moved by the clock error at its time, 0.5 Hz high-pass instead of a baseline, average reference without
the channels that are flat or noisy in the block (SD > 3x median), 150 µV rejection, 20-ms bins, 10 posterior channels.
  remainder    shift s that maximises the similarity between this session at t + s and the THINGS-EEG2 lab-contract
               responses to the same images at t = 60-300 ms (label-free: only the pairing of images is used). Null: images
               re-paired, maximum |r| over all shifts. Controls: one THINGS subject against the other nine gives 0 ms,
               the old lab against THINGS +20 ms.
  by quarter   the same for each quarter of the block in time: the delay of the display relative to THINGS-EEG2, which
               should match rsvp_display_lag_ms. People differ by a few tens of ms, and the estimate moves in 20-ms steps,
               so the automatic offset counts as fitting within 40 ms.
  reliability  odd against even repeats of an image, per bin: when is there image-specific signal at all?
  start        mean response of the posterior channels to the start of a sequence
If it does not fit (for example a drifting session without stimulation pulses), the script prints an rsvp_offset_ms line
for configs/sessions.yaml, fitted to the quarters. The image order is shuffled between repeats, so the previous image
cannot produce an image-specific signal.
Output: results/rsvp_timing_<session>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import data_things  # noqa: E402
from starstim_recon.channels import STARSTIM_31  # noqa: E402
from starstim_recon.data_sessions import (FS, REJECT_UV, _folder, image_table, rsvp_display_lag_ms, rsvp_offset_ms,  # noqa: E402
                                          rsvp_runs)
from starstim_recon.paths import RESULTS  # noqa: E402

POSTERIOR = ("O1", "Oz", "O2", "PO3", "PO4", "P7", "P8", "P3", "P4", "Pz")
T0, T1, BIN = -1.0, 2.0, 10                                     # epoch around the marker (s), samples per 20-ms bin
THINGS_WINDOW = (0.06, 0.30)                                    # THINGS latencies that are matched (s)
SHIFTS_MS = np.arange(-700, 401, 20)                            # searched after the clock error is removed
TOLERANCE_MS = 40
N_NULL = 300


def col_r(a, b):                                                # correlation across images, per (channel, bin)
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); args = ap.parse_args()
    rng = np.random.default_rng(0)
    sos = signal.butter(2, 0.5, btype="high", fs=FS, output="sos"); a, b = int(T0 * FS), int(T1 * FS)
    runs = list(rsvp_runs(args.session))
    assert runs, f"{args.session}: no rapid-stream block"
    blocks = []
    for _, x, trigger, on, seq, clock_ms in runs:
        t_min = on / FS / 60; on = on + np.round(clock_ms / 1000 * FS).astype(int)       # the clock error removed
        lo, hi = max(on.min() + a - 5 * FS, 0), min(on.max() + b + 5 * FS, len(x))
        pos = np.array([np.sum(seq[:i + 1] == seq[i]) for i in range(len(seq))])            # 1..24 within its sequence
        flat = (np.diff(x[lo:hi], axis=0) == 0).mean(0) > 0.5
        blocks.append((signal.sosfiltfilt(sos, x[lo:hi], axis=0), on - lo, trigger, pos, flat, t_min, clock_ms))
    xall = np.concatenate([q[0] for q in blocks]); flat = np.any([q[4] for q in blocks], axis=0)
    sd = (xall - xall.mean(1, keepdims=True)).std(0)
    left_out = [c for c, v, f in zip(STARSTIM_31, sd, flat) if f or v > 3 * np.median(sd)]
    ch = [c for c in STARSTIM_31 if c not in left_out]; ci = [STARSTIM_31.index(c) for c in ch]
    post = [ch.index(c) for c in POSTERIOR if c in ch]

    ep, trig, pos, minute, clock = [], [], [], [], []
    for xh, on, trigger, p, _, t, cl in blocks:
        xr = xh[:, ci] - xh[:, ci].mean(1, keepdims=True)
        ok = (on + a >= 0) & (on + b <= len(xr))
        ep.append(np.stack([xr[o + a:o + b] for o in on[ok]]).transpose(0, 2, 1)); trig.append(trigger[ok]); pos.append(p[ok])
        minute.append(t[ok]); clock.append(cl[ok])
    ep, trig, pos = np.concatenate(ep), np.concatenate(trig), np.concatenate(pos)
    minute, clock = np.concatenate(minute), np.concatenate(clock)
    i0 = -a; first = pos == 1                                                                 # ep: (n, C, 1500), marker at i0
    start = ep[first & (np.ptp(ep[..., i0 - 250:i0 + 250], axis=-1).max(1) <= REJECT_UV)]
    keep = np.ptp(ep[..., i0 - 300:i0 + 600], axis=-1).max(1) <= REJECT_UV
    n_all = len(ep); ep, trig, minute, clock = ep[keep], trig[keep], minute[keep], clock[keep]
    uniq = np.array(sorted(set(trig))); nb = ep.shape[-1] // BIN; tb = T0 + (np.arange(nb) + 0.5) * BIN / FS
    eb = ep.reshape(len(ep), len(ch), nb, BIN).mean(-1)
    print(f"{args.session}: {len(ep)} of {n_all} trials kept, {len(uniq)} images, channels left out {left_out}")

    # reliability: odd against even repeats
    odd = np.stack([eb[np.flatnonzero(trig == u)[0::2]].mean(0) for u in uniq])
    even = np.stack([eb[np.flatnonzero(trig == u)[1::2]].mean(0) for u in uniq])
    rel = col_r(odd, even)[post].mean(0)
    rel_thr = float(np.percentile([col_r(odd, even[rng.permutation(len(uniq))])[post].mean(0).max() for _ in range(N_NULL)], 95))
    above = [int(round(t * 1000)) for t in tb[rel > rel_thr]]
    print(f"reliability (odd vs even repeats): above the threshold {rel_thr:+.3f} at {above} ms after the corrected marker; "
          f"peak {rel.max():+.3f} at {tb[rel.argmax()] * 1000:+.0f} ms")

    # offset: best match with the THINGS-EEG2 responses to the same images
    files = {t - 2000: p for t, p in image_table(_folder(args.session), "ssvep").items()}
    things_index = {p.name: i for i, p in enumerate(data_things.image_paths("train"))}
    shared = [(k, things_index[files[int(u)].name]) for k, u in enumerate(uniq) if files[int(u)].name in things_index]
    assert len(shared) >= 20, f"{args.session}: only {len(shared)} images are THINGS-EEG2 training images"
    T = data_things.eeg_means("train")[[j for _, j in shared]].view(-1, 31, 250).numpy().astype(np.float64)[:, ci]
    Tb = T.reshape(len(shared), len(ch), 50, 5).mean(-1); tt = (np.arange(50) + 0.5) * 0.02
    win = np.flatnonzero((tt >= THINGS_WINDOW[0]) & (tt < THINGS_WINDOW[1]))
    cov = np.mean([LedoitWolf().fit(ep[i][:, i0 - 200:i0 + 500:2].T).covariance_
                   for i in rng.choice(len(ep), min(600, len(ep)), replace=False)], axis=0)       # the cache is ZCA-whitened
    w, V = np.linalg.eigh(cov); W = V @ np.diag(np.clip(w, w.max() * 1e-6, None) ** -0.5) @ V.T
    Tc = (Tb - Tb.mean(0))[:, post][..., win]
    shifts = SHIFTS_MS // 20; marker_bin = int(round(-T0 * FS / BIN))

    def curve(image_means):
        E = np.einsum("cd,idt->ict", W, image_means)[[k for k, _ in shared]]; E = (E - E.mean(0))[:, post]
        out = []
        for s in shifts:
            e = E[..., win + marker_bin + s]
            out.append(float((e * Tc).sum() / np.sqrt((e ** 2).sum() * (Tc ** 2).sum())))
        return np.array(out)

    means = np.stack([eb[trig == u].mean(0) for u in uniq])
    c = curve(means); k = int(np.argmax(c)); order = [q for q, _ in shared]
    null = []
    for _ in range(N_NULL):
        m = means.copy(); m[order] = means[rng.permutation(order)]; null.append(np.abs(curve(m)).max())
    null95 = float(np.percentile(null, 95)); found = bool(c[k] > null95)
    print(f"remainder after the clock error: best shift {SHIFTS_MS[k]:+d} ms (r {c[k]:+.3f}; null 95% {null95:.3f}; "
          f"{len(shared)} images shared with THINGS-EEG2)")

    by_quarter = []
    for idx in np.array_split(np.arange(len(ep)), 4):            # trials are in time order
        m = np.zeros(len(ep), bool); m[idx] = True
        cm = curve(np.stack([eb[m & (trig == u)].mean(0) if (m & (trig == u)).any() else np.zeros(eb.shape[1:]) for u in uniq]))
        by_quarter.append({"minute": round(float(minute[idx].mean()), 1), "clock_ms": round(float(clock[idx].mean()), 1),
                           "display_lag_ms": int(SHIFTS_MS[cm.argmax()]), "r": round(float(cm.max()), 4)})
    print("by quarter of the block (minute of the run, clock error from the pulses, remaining display lag):")
    for q in by_quarter:
        print(f"  {q['minute']:5.1f} min: clock {q['clock_ms']:+7.1f} ms   display lag {q['display_lag_ms']:+5d} ms (r {q['r']:+.3f})")

    erp = start.mean(0)[post].mean(0).reshape(nb, BIN).mean(-1); m = (tb > -0.5) & (tb < 0.4)
    print(f"start of a sequence ({len(start)} sequences), posterior channels (µV) by time after the first corrected marker (ms):")
    print("  " + " ".join(f"{int(round(t * 1000)):+5d}" for t in tb[m][::2]))
    print("  " + " ".join(f"{v:+5.1f}" for v in erp[m][::2]))

    lag = rsvp_display_lag_ms(); implied = float(np.median([q["display_lag_ms"] for q in by_quarter]))
    agrees = found and abs(implied - lag) <= TOLERANCE_MS
    print(f"clock error from the pulses: {clock[0]:+.0f} ms at the first image, {clock[-1]:+.0f} ms at the last"
          + ("" if clock.any() else " (no correction for this session)"))
    print(f"display lag implied by the images: {implied:+.0f} ms (median of the quarters); configured {lag:+g} ms")
    override, suggested = rsvp_offset_ms(args.session), None
    if not found:
        print("-> the image-based estimate is NOT above its null: nothing to check the automatic offset against")
    elif agrees:
        print("-> the automatic offset fits" + (f"; the rsvp_offset_ms {list(override)} of this session is not needed" if override else ""))
    else:
        t = np.array([q["minute"] for q in by_quarter]); v = np.array([q["clock_ms"] + q["display_lag_ms"] for q in by_quarter])
        slope = float(np.polyfit(t, v, 1)[0]) if np.ptp(v) >= 40 else 0.0
        suggested = [round(float(v.mean() - slope * t.mean()), 1), round(slope, 2)]
        print(f"-> the automatic offset does NOT fit; enter for this session  rsvp_offset_ms: [{suggested[0]:g}, {suggested[1]:g}]"
              + (f"  (configs/sessions.yaml has {list(override)})" if override else ""))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"rsvp_timing_{args.session}.json").write_text(json.dumps(
        {"session": args.session, "n_trials": int(len(ep)), "n_images": int(len(uniq)), "channels_left_out": left_out,
         "n_shared_with_things": len(shared), "remainder_ms": int(SHIFTS_MS[k]), "r": float(c[k]), "null95": null95,
         "above_null": found, "clock_ms": [float(clock[0]), float(clock[-1])], "by_quarter": by_quarter,
         "display_lag_implied_ms": implied, "display_lag_configured_ms": lag, "automatic_offset_fits": bool(agrees),
         "configured_override_ms": override, "suggested_override_ms": suggested,
         "similarity": {"shift_ms": SHIFTS_MS.tolist(), "r": np.round(c, 4).tolist()},
         "reliability": {"t_ms": np.round(tb * 1000).astype(int).tolist(), "r": np.round(rel, 4).tolist(), "threshold": rel_thr},
         "sequence_start": {"n": int(len(start)), "t_ms": np.round(tb * 1000).astype(int).tolist(),
                            "uV": np.round(erp, 3).tolist()}}, indent=2))
    print("wrote", RESULTS / f"rsvp_timing_{args.session}.json")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Timing check for a rapid-stream block: where is the image-specific response relative to the image marker?

    python scripts/check_rsvp_timing.py --session E23

Run it for every new rapid-stream session and enter the suggested rsvp_offset_ms in configs/sessions.yaml. In some sessions
(E16-E19, E21, E23) the EEG time stamps drift against the marker clock by 2-6 ms per minute, so late in a recording the
response to an image starts well before its marker.

Marker timing as recorded (data_sessions.rsvp_runs), 0.5 Hz high-pass instead of a baseline, average reference without the
channels that are flat or noisy in the block (SD > 3x median), 150 µV rejection, 20-ms bins, 10 posterior channels.
  clock        EEG clock against marker clock from the stimulation pulses of the same run: latency of the pulse artifact
               relative to the controller's pulse event, as a line over the run. Without drift the artifact follows its
               event by 6-10 ms (E06-E15, E20).
  offset       shift s that maximises the similarity between this session at t + s and the THINGS-EEG2 lab-contract
               responses to the same images at t = 60-300 ms (label-free: only the pairing of images is used). Null: images
               re-paired, maximum |r| over all shifts. Controls: one THINGS subject against the other nine gives 0 ms,
               the old lab against THINGS +20 ms.
  by quarter   the same for each quarter of the block in time, next to the clock line: what remains is the delay of the
               display relative to THINGS-EEG2 (about 40 ms in E21 and E23)
  reliability  odd against even repeats of an image, per bin: when is there image-specific signal at all?
  start        mean response of the posterior channels to the start of a sequence
The image order is shuffled between repeats, so the previous image cannot produce an image-specific signal.
Output: results/rsvp_timing_<session>.json.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.stats import theilslopes
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon import data_things  # noqa: E402
from starstim_recon.channels import STARSTIM_31  # noqa: E402
from starstim_recon.data_sessions import FS, REJECT_UV, _folder, image_table, rsvp_offset_ms, rsvp_runs  # noqa: E402
from starstim_recon.paths import RESULTS  # noqa: E402

POSTERIOR = ("O1", "Oz", "O2", "PO3", "PO4", "P7", "P8", "P3", "P4", "Pz")
T0, T1, BIN = -1.0, 2.0, 10                                     # epoch around the marker (s), samples per 20-ms bin
THINGS_WINDOW = (0.06, 0.30)                                    # THINGS latencies that are matched (s)
SHIFTS_MS = np.arange(-700, 401, 20)
N_NULL = 300
PULSE_LATENCY_MS = 8.0                                          # pulse artifact after its event when the clocks agree


def col_r(a, b):                                                # correlation across images, per (channel, bin)
    a = a - a.mean(0); b = b - b.mean(0)
    return (a * b).sum(0) / (np.sqrt((a ** 2).sum(0) * (b ** 2).sum(0)) + 1e-12)


def pulse_line(x, pulses):
    """Latency of the pulse artifact (first sample-to-sample jump > 50x the median, within 1 s of the pulse event) relative
    to its event, as a line over the run (Theil-Sen) -> dict, or None with fewer than 20 pulses."""
    d = np.abs(np.diff(x, axis=0)).max(1); thr = 50 * np.median(d); minute, latency = [], []
    for o in pulses:
        big = np.flatnonzero(d[max(o - FS, 0):o + FS] > thr)
        if o >= FS and len(big):
            minute.append(o / FS / 60); latency.append((big[0] - FS) / FS * 1000)
    if len(minute) < 20:
        return None
    slope, intercept, _, _ = theilslopes(latency, minute)
    res = np.abs(np.array(latency) - (intercept + slope * np.array(minute)))
    return {"n_pulses": len(minute), "minutes": [round(min(minute), 1), round(max(minute), 1)], "at_start_ms": round(float(intercept), 1),
            "per_min_ms": round(float(slope), 2), "residual_median_abs_ms": round(float(np.median(res)), 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--session", required=True); args = ap.parse_args()
    rng = np.random.default_rng(0)
    sos = signal.butter(2, 0.5, btype="high", fs=FS, output="sos"); a, b = int(T0 * FS), int(T1 * FS)
    blocks, lines = [], []
    for run, x, trigger, on, seq, pulses in rsvp_runs(args.session):
        lo, hi = max(on.min() + a - 5 * FS, 0), min(on.max() + b + 5 * FS, len(x))
        pos = np.array([np.sum(seq[:i + 1] == seq[i]) for i in range(len(seq))])            # 1..24 within its sequence
        flat = (np.diff(x[lo:hi], axis=0) == 0).mean(0) > 0.5
        blocks.append((signal.sosfiltfilt(sos, x[lo:hi], axis=0), on - lo, trigger, pos, flat, on / FS / 60))
        lines.append(pulse_line(x, pulses))
        print(f"clock, run {run[-12:]}: " + (f"pulse artifact relative to its event {lines[-1]['at_start_ms']:+.0f} ms at the start of the "
              f"run, {lines[-1]['per_min_ms']:+.2f} ms per minute ({lines[-1]['n_pulses']} pulses at {lines[-1]['minutes'][0]}-"
              f"{lines[-1]['minutes'][1]} min, residual {lines[-1]['residual_median_abs_ms']} ms); rapid stream at "
              f"{on.min() / FS / 60:.1f}-{on.max() / FS / 60:.1f} min" if lines[-1] else "fewer than 20 stimulation pulses, no clock line"))
    assert blocks, f"{args.session}: no rapid-stream block"
    xall = np.concatenate([q[0] for q in blocks]); flat = np.any([q[4] for q in blocks], axis=0)
    sd = (xall - xall.mean(1, keepdims=True)).std(0)
    left_out = [c for c, v, f in zip(STARSTIM_31, sd, flat) if f or v > 3 * np.median(sd)]
    ch = [c for c in STARSTIM_31 if c not in left_out]; ci = [STARSTIM_31.index(c) for c in ch]
    post = [ch.index(c) for c in POSTERIOR if c in ch]

    ep, trig, pos, minute = [], [], [], []
    for xh, on, trigger, p, _, t in blocks:
        xr = xh[:, ci] - xh[:, ci].mean(1, keepdims=True)
        ok = (on + a >= 0) & (on + b <= len(xr))
        ep.append(np.stack([xr[o + a:o + b] for o in on[ok]]).transpose(0, 2, 1)); trig.append(trigger[ok]); pos.append(p[ok])
        minute.append(t[ok])
    ep, trig, pos, minute = np.concatenate(ep), np.concatenate(trig), np.concatenate(pos), np.concatenate(minute)
    i0 = -a; first = pos == 1                                                                 # ep: (n, C, 1500), marker at i0
    start = ep[first & (np.ptp(ep[..., i0 - 250:i0 + 250], axis=-1).max(1) <= REJECT_UV)]
    keep = np.ptp(ep[..., i0 - 300:i0 + 600], axis=-1).max(1) <= REJECT_UV
    n_all = len(ep); ep, trig, minute = ep[keep], trig[keep], minute[keep]
    uniq = np.array(sorted(set(trig))); nb = ep.shape[-1] // BIN; tb = T0 + (np.arange(nb) + 0.5) * BIN / FS
    eb = ep.reshape(len(ep), len(ch), nb, BIN).mean(-1)
    print(f"{args.session}: {len(ep)} of {n_all} trials kept, {len(uniq)} images, channels left out {left_out}")

    # reliability: odd against even repeats
    odd = np.stack([eb[np.flatnonzero(trig == u)[0::2]].mean(0) for u in uniq])
    even = np.stack([eb[np.flatnonzero(trig == u)[1::2]].mean(0) for u in uniq])
    rel = col_r(odd, even)[post].mean(0)
    rel_thr = float(np.percentile([col_r(odd, even[rng.permutation(len(uniq))])[post].mean(0).max() for _ in range(N_NULL)], 95))
    above = [int(round(t * 1000)) for t in tb[rel > rel_thr]]
    print(f"reliability (odd vs even repeats): above the threshold {rel_thr:+.3f} at {above} ms relative to the marker; "
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
    print(f"offset: best shift {SHIFTS_MS[k]:+d} ms (r {c[k]:+.3f}; r at the marker {c[list(SHIFTS_MS).index(0)]:+.3f}; "
          f"null 95% {null95:.3f}; {len(shared)} images shared with THINGS-EEG2)")

    line = lines[0] if len(lines) == 1 else None                 # one clock line can be compared only with one rapid-stream run
    by_quarter = []
    for idx in np.array_split(np.arange(len(ep)), 4):            # trials are in time order
        m = np.zeros(len(ep), bool); m[idx] = True
        cm = curve(np.stack([eb[m & (trig == u)].mean(0) if (m & (trig == u)).any() else np.zeros(eb.shape[1:]) for u in uniq]))
        q = {"minute": round(float(minute[idx].mean()), 1), "best_shift_ms": int(SHIFTS_MS[cm.argmax()]), "r": round(float(cm.max()), 4)}
        if line:
            q["clock_ms"] = round(line["at_start_ms"] + line["per_min_ms"] * q["minute"] - PULSE_LATENCY_MS, 1)
            q["display_lag_ms"] = round(q["best_shift_ms"] - q["clock_ms"], 1)
        by_quarter.append(q)
    print("by quarter of the block (minute of the run, best shift"
          + (", EEG clock from the pulses, remaining display lag):" if line else "):"))
    for q in by_quarter:
        print(f"  {q['minute']:5.1f} min: {q['best_shift_ms']:+5d} ms (r {q['r']:+.3f})"
              + (f"   clock {q['clock_ms']:+7.1f} ms   display lag {q['display_lag_ms']:+6.1f} ms" if line else ""))

    erp = start.mean(0)[post].mean(0).reshape(nb, BIN).mean(-1); m = (tb > -0.5) & (tb < 0.4)
    print(f"start of a sequence ({len(start)} sequences), posterior channels (µV) by time relative to the first marker (ms):")
    print("  " + " ".join(f"{int(round(t * 1000)):+5d}" for t in tb[m][::2]))
    print("  " + " ".join(f"{v:+5.1f}" for v in erp[m][::2]))

    if line and found:
        lag = float(np.median([q["display_lag_ms"] for q in by_quarter]))
        suggested = [round(line["at_start_ms"] - PULSE_LATENCY_MS + lag, 1), line["per_min_ms"]]
        how = f"pulse line - {PULSE_LATENCY_MS:g} ms + display lag {lag:+.0f} ms"
    else:
        suggested = [float(SHIFTS_MS[k]), 0.0]
        how = "constant: best shift over the block" + ("" if found else ", NOT above the null: do not use")
    configured = list(rsvp_offset_ms(args.session))
    print(f"-> suggested rsvp_offset_ms: [{suggested[0]:g}, {suggested[1]:g}]  ({how}); configs/sessions.yaml has {configured}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"rsvp_timing_{args.session}.json").write_text(json.dumps(
        {"session": args.session, "n_trials": int(len(ep)), "n_images": int(len(uniq)), "channels_left_out": left_out,
         "clock_lines": lines, "n_shared_with_things": len(shared), "best_shift_ms": int(SHIFTS_MS[k]), "r": float(c[k]),
         "null95": null95, "above_null": found, "by_quarter": by_quarter, "suggested_offset_ms": suggested,
         "configured_offset_ms": configured, "similarity": {"shift_ms": SHIFTS_MS.tolist(), "r": np.round(c, 4).tolist()},
         "reliability": {"t_ms": np.round(tb * 1000).astype(int).tolist(), "r": np.round(rel, 4).tolist(), "threshold": rel_thr},
         "sequence_start": {"n": int(len(start)), "t_ms": np.round(tb * 1000).astype(int).tolist(),
                            "uV": np.round(erp, 3).tolist()}}, indent=2))
    print("wrote", RESULTS / f"rsvp_timing_{args.session}.json")


if __name__ == "__main__":
    main()

"""StarStim lab sessions (data/lab_sessions/E<nn>_<date>_<code>/experiment/*.xdf + PsychoPy files).

No-pulse image trials (cue 70, image trigger 1001-1299), 26 channels (the six early stimulation electrodes excluded),
channels flat in a run (stimulation electrodes of that run) and bad channels (SD > 3x median) interpolated, average
reference, baseline -0.2..0 s, 150 µV peak-to-peak rejection; then the old-lab format: FFT resample of the -0.2..1.0 s
epoch to 250 Hz, baseline, crop 0-1 s, the 25 channels shared with the old lab, re-referenced.
Epochs are placed by sample index on a smoothed LSL clock (linear fit + 5-s running median), never by raw time stamps.

load_rsvp: rapid-stream ("SSVEP") blocks without stimulation (image codes 3001-3299 = image trigger + 2000, 5 images per
second), in the THINGS lab contract: the 31 StarStim channels shared with THINGS-EEG2, channels flat or bad within the block
interpolated, average reference, baseline -0.2..0 s, 150 µV rejection, 250 Hz, 0-1 s. Time zero is the image marker plus the
session's rsvp_offset_ms (configs/sessions.yaml; estimated by scripts/check_rsvp_timing.py): in some sessions the EEG time
stamps drift against the marker clock, so the offset is a line over the recording.

Privacy: the PsychoPy files also hold questionnaire answers; only whitelisted columns are ever read
(presentation_image_trigger / presentation_image_path and ssvep_image_trigger / ssvep_image_path).
"""
import csv
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import torch
import yaml
from scipy import signal
from scipy.ndimage import median_filter

from .channels import SESSION_26, SHARED_25, STARSTIM_31, STARSTIM_32
from .paths import CONFIGS, LAB_SESSIONS, THINGS_OBJECT_IMAGES

FS, PRE, POST = 500, 100, 500                                  # -0.2..1.0 s at 500 Hz
CUE = {70: "none", 71: "stim", 72: "pulse_only", 73: "blank"}
REJECT_UV = 150.0
RSVP_CODES = (3001, 3299)                                      # rapid-stream image codes = image trigger + 2000
PULSE_EVENT = 7100                                             # stimulation pulse in the controller's stream neurostim_nic


def session_ids(kind=None, rsvp=None) -> list:
    """Session ids from configs/sessions.yaml (optionally only one kind / only sessions with a rapid-stream block)."""
    S = yaml.safe_load(open(CONFIGS / "sessions.yaml"))["sessions"]
    return [s["id"] for s in S if (kind is None or s.get("kind") == kind) and (rsvp is None or bool(s.get("rsvp")) == rsvp)]


def rsvp_offset_ms(session_id: str) -> tuple:
    """True image onset minus image marker in the session's rapid-stream block -> (ms at the start of the run's EEG stream,
    ms per minute). configs/sessions.yaml: [a, b], or one number for a constant offset; (0, 0) if not given."""
    S = yaml.safe_load(open(CONFIGS / "sessions.yaml"))["sessions"]
    v = next((s.get("rsvp_offset_ms", 0) for s in S if s["id"] == session_id), 0)
    return tuple(float(q) for q in v) if isinstance(v, list) else (float(v), 0.0)


def _folder(session_id: str) -> Path:
    hits = sorted(glob.glob(str(LAB_SESSIONS / f"{session_id}_*")))
    assert len(hits) == 1, f"{session_id}: expected one folder in {LAB_SESSIONS}, found {len(hits)}"
    return Path(hits[0])


def _interpolate(ep, bads, channels=SESSION_26):
    import mne
    info = mne.create_info(list(channels), FS, "eeg"); info.set_montage("standard_1020"); info["bads"] = list(bads)
    e = mne.EpochsArray(ep * 1e-6, info, verbose=False); e.interpolate_bads(reset_bads=True, verbose=False)
    return e.get_data() * 1e6


def load_run(path):
    """No-pulse image trials of one XDF run, whether their block contained any pulse, and the run's flat channels."""
    st, _ = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_pyexp"}, {"name": "neurostim_pyexp-EEG"}])
    S = {s["info"]["name"][0]: s for s in st}
    if "neurostim_pyexp" not in S or "neurostim_pyexp-EEG" not in S:
        return None
    codes = np.array(S["neurostim_pyexp"]["time_series"]).ravel().astype(int); mts = S["neurostim_pyexp"]["time_stamps"]
    rows, blk, cond = [], None, None
    for c, t in zip(codes, mts):
        if 131 <= c <= 146: blk = c - 130
        elif c in CUE: cond = CUE[c]
        elif 1001 <= c <= 1299 and cond is not None: rows.append(dict(block=blk, condition=cond, trigger=c, image_lsl=t)); cond = None
        elif c == 1000: cond = None
    tr = pd.DataFrame(rows)
    if tr.empty:
        return None
    pulsed_blocks = set(tr.block[tr.condition.isin(["stim", "pulse_only"])])
    x = S["neurostim_pyexp-EEG"]["time_series"].astype(np.float64); lin = S["neurostim_pyexp-EEG"]["time_stamps"]
    raw = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_pyexp-EEG"}], dejitter_timestamps=False)[0][0]["time_stamps"]
    clock = lin + median_filter(raw - lin, size=2501, mode="nearest")
    tr = tr[tr.condition == "none"].reset_index(drop=True)
    if tr.empty:
        return None
    on = np.searchsorted(clock, tr.image_lsl.values); ok = (on - PRE >= 0) & (on + POST <= len(x))
    tr, on = tr[ok].reset_index(drop=True), on[ok]
    ci = [STARSTIM_32.index(c) for c in SESSION_26]
    ep = np.stack([x[o - PRE:o + POST, ci] for o in on]).transpose(0, 2, 1)       # (n, 26, 600) µV, own reference
    flat = [c for c, i in zip(SESSION_26, ci) if (np.diff(x[:, i]) == 0).mean() > 0.5]
    if flat:
        ep = _interpolate(ep, flat)
    return tr.assign(run=Path(path).stem, pulse_free=~tr.block.isin(pulsed_blocks)), ep, flat


def image_table(folder: Path, prefix="presentation") -> dict:
    """trigger -> THINGS object image, from the PsychoPy files through a two-column whitelist ({prefix}_image_trigger / _path)."""
    tcol, pcol = f"{prefix}_image_trigger", f"{prefix}_image_path"
    table = {}
    for f in sorted(glob.glob(str(folder / "experiment/*.csv"))):
        if f.endswith(("_markers.csv", "_pulses.csv")):
            continue
        header = next(csv.reader(open(f)))
        if tcol not in header or pcol not in header:
            continue
        d = pd.read_csv(f, usecols=[tcol, pcol]).dropna()
        for t, p in zip(d[tcol].astype(int), d[pcol]):
            p = Path(str(p).replace("\\", "/")); local = THINGS_OBJECT_IMAGES / p.parent.name / p.name
            assert table.get(t, local) == local, f"{folder.name}: trigger {t} maps to two images"
            table[t] = local
    return table


def load_session(session_id: str):
    """-> dict(tr: trial table, ep: (n, 25, 250) tensor in the old-lab format, files: trigger -> image, bad, flat, missing)."""
    folder = _folder(session_id)
    parts = [p for p in (load_run(f) for f in sorted(glob.glob(str(folder / "experiment/*.xdf")))) if p is not None]
    if not parts:
        return None
    tr = pd.concat([p[0] for p in parts], ignore_index=True); ep = np.concatenate([p[1] for p in parts])
    ep = ep - ep[..., :PRE].mean(-1, keepdims=True)
    ea = ep - ep.mean(1, keepdims=True)
    sd = ea.std((0, 2)); bad = [c for c, v in zip(SESSION_26, sd) if v > 3 * np.median(sd)]
    if bad:
        ep = _interpolate(ep, bad)
    ep = ep - ep.mean(1, keepdims=True)
    keep = np.ptp(ep, axis=-1).max(1) <= REJECT_UV
    tr, ep = tr[keep].reset_index(drop=True), ep[keep]
    ep = signal.resample(ep, 300, axis=-1)                                         # -0.2..1.0 s at 250 Hz
    ep = ep - ep[..., :50].mean(-1, keepdims=True)
    ep = ep[..., 50:300][:, [SESSION_26.index(c) for c in SHARED_25]]                # 0..1 s, 25 shared channels
    ep = ep - ep.mean(1, keepdims=True)
    files = image_table(folder)
    missing = sorted({int(t) for t in tr.trigger if int(t) not in files or not files[int(t)].exists()})
    return dict(tr=tr, ep=torch.from_numpy(ep).float(), files=files, bad=bad, missing=missing,
                flat=sorted({c for p in parts for c in p[2]}))


def rsvp_runs(session_id: str):
    """Every XDF run of a session with a rapid-stream block -> (run name, x (samples, 31) µV in its own reference, image
    trigger per image marker, sample index of each image marker, sequence number per image marker (code 220 starts a
    sequence), sample index of each stimulation pulse event of the run). Marker timing as recorded: no offset is applied."""
    ci = [STARSTIM_32.index(c) for c in STARSTIM_31]
    for path in sorted(glob.glob(str(_folder(session_id) / "experiment/*.xdf"))):
        st, _ = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_pyexp"}, {"name": "neurostim_pyexp-EEG"},
                                                     {"name": "neurostim_nic"}])
        S = {s["info"]["name"][0]: s for s in st}
        if "neurostim_pyexp" not in S or "neurostim_pyexp-EEG" not in S:
            continue
        codes = np.array(S["neurostim_pyexp"]["time_series"]).ravel().astype(int)
        img = (codes >= RSVP_CODES[0]) & (codes <= RSVP_CODES[1])
        if not img.any():
            continue
        mts = np.array(S["neurostim_pyexp"]["time_stamps"])
        x = S["neurostim_pyexp-EEG"]["time_series"].astype(np.float64)[:, ci]; lin = S["neurostim_pyexp-EEG"]["time_stamps"]
        raw = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_pyexp-EEG"}], dejitter_timestamps=False)[0][0]["time_stamps"]
        clock = lin + median_filter(raw - lin, size=2501, mode="nearest")
        nic = S.get("neurostim_nic", {"time_series": [], "time_stamps": []})
        pulse = [t for v, t in zip(nic["time_series"], nic["time_stamps"]) if int(float(v[0])) == PULSE_EVENT]
        yield (Path(path).stem, x, codes[img] - 2000, np.searchsorted(clock, mts[img]), np.cumsum(codes == 220)[img],
               np.searchsorted(clock, pulse))


def load_rsvp(session_id: str, require_images=True, offset_ms=None):
    """Rapid-stream block of a session -> dict(tr: trial table (trigger, sequence), ep: (n, 31, 250) tensor in the THINGS
    lab contract, files: trigger -> image, flat, bad, n_all, offset_ms) or None if the session has no such block.
    require_images=False skips the check that every image file exists (for EEG-only analyses when the image disk is offline).
    offset_ms: time zero relative to the image marker, (ms at the start of the run's EEG stream, ms per minute) or one
    number; None = the session's rsvp_offset_ms, 0 = the marker itself."""
    folder = _folder(session_id)
    a, b = rsvp_offset_ms(session_id) if offset_ms is None else (*(float(v) for v in np.atleast_1d(offset_ms)), 0.0)[:2]
    trs, eps, flat = [], [], set()
    for run, x, trigger, on, seq, _ in rsvp_runs(session_id):
        on = on + np.round((a + b * on / FS / 60) / 1000 * FS).astype(int); ok = (on - PRE >= 0) & (on + POST <= len(x))
        block = x[on[ok].min() - PRE:on[ok].max() + POST]
        run_flat = [c for i, c in enumerate(STARSTIM_31) if (np.diff(block[:, i]) == 0).mean() > 0.5]
        ep = np.stack([x[o - PRE:o + POST] for o in on[ok]]).transpose(0, 2, 1)       # (n, 31, 600) µV
        if run_flat:
            ep = _interpolate(ep, run_flat, STARSTIM_31)
        flat |= set(run_flat)
        trs.append(pd.DataFrame(dict(trigger=trigger[ok], sequence=seq[ok], run=run))); eps.append(ep)
    if not eps:
        return None
    tr = pd.concat(trs, ignore_index=True); ep = np.concatenate(eps)
    ep = ep - ep[..., :PRE].mean(-1, keepdims=True)
    sd = (ep - ep.mean(1, keepdims=True)).std((0, 2)); bad = [c for c, v in zip(STARSTIM_31, sd) if v > 3 * np.median(sd)]
    if bad:
        ep = _interpolate(ep, bad, STARSTIM_31)
    ep = ep - ep.mean(1, keepdims=True)
    keep = np.ptp(ep, axis=-1).max(1) <= REJECT_UV
    n_all = len(tr); tr, ep = tr[keep].reset_index(drop=True), ep[keep]
    ep = signal.resample(ep, 300, axis=-1)
    ep = ep - ep[..., :50].mean(-1, keepdims=True)
    ep = ep[..., 50:300]; ep = ep - ep.mean(1, keepdims=True)                      # 0..1 s, 31 channels
    files = {t - 2000: p for t, p in image_table(folder, "ssvep").items()}
    missing = sorted({int(t) for t in tr.trigger if int(t) not in files or not files[int(t)].exists()})
    assert not (require_images and missing), f"{session_id}: rapid-stream images missing for triggers {missing}"
    return dict(tr=tr, ep=torch.from_numpy(np.ascontiguousarray(ep)).float(), files=files, flat=sorted(flat), bad=bad,
                n_all=n_all, offset_ms=[a, b])

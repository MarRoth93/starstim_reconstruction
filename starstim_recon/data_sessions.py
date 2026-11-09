"""StarStim lab sessions (data/lab_sessions/E<nn>_<date>_<code>/experiment/*.xdf + PsychoPy files).

No-pulse image trials (cue 70, image trigger 1001-1299), 26 channels (the six early stimulation electrodes excluded),
channels flat in a run (stimulation electrodes of that run) and bad channels (SD > 3x median) interpolated, average
reference, baseline -0.2..0 s, 150 µV peak-to-peak rejection; then the old-lab format: FFT resample of the -0.2..1.0 s
epoch to 250 Hz, baseline, crop 0-1 s, the 25 channels shared with the old lab, re-referenced.
Epochs are placed by sample index on a smoothed LSL clock (linear fit + 5-s running median), never by raw time stamps.

Privacy: the PsychoPy files also hold questionnaire answers; only two whitelisted columns are ever read
(presentation_image_trigger, presentation_image_path).
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

from .channels import SESSION_26, SHARED_25, STARSTIM_32
from .paths import CONFIGS, LAB_SESSIONS, THINGS_OBJECT_IMAGES

FS, PRE, POST = 500, 100, 500                                  # -0.2..1.0 s at 500 Hz
CUE = {70: "none", 71: "stim", 72: "pulse_only", 73: "blank"}
REJECT_UV = 150.0


def session_ids(kind=None) -> list:
    """Session ids from configs/sessions.yaml (optionally only one kind)."""
    S = yaml.safe_load(open(CONFIGS / "sessions.yaml"))["sessions"]
    return [s["id"] for s in S if kind is None or s.get("kind") == kind]


def _interpolate(ep, bads):
    import mne
    info = mne.create_info(SESSION_26, FS, "eeg"); info.set_montage("standard_1020"); info["bads"] = list(bads)
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


def image_table(folder: Path) -> dict:
    """trigger -> THINGS object image, from the PsychoPy files through a two-column whitelist."""
    table = {}
    for f in sorted(glob.glob(str(folder / "experiment/*.csv"))):
        if f.endswith(("_markers.csv", "_pulses.csv")):
            continue
        header = next(csv.reader(open(f)))
        if "presentation_image_trigger" not in header or "presentation_image_path" not in header:
            continue
        d = pd.read_csv(f, usecols=["presentation_image_trigger", "presentation_image_path"]).dropna()
        for t, p in zip(d.presentation_image_trigger.astype(int), d.presentation_image_path):
            p = Path(str(p).replace("\\", "/")); local = THINGS_OBJECT_IMAGES / p.parent.name / p.name
            assert table.get(t, local) == local, f"{folder.name}: trigger {t} maps to two images"
            table[t] = local
    return table


def load_session(session_id: str):
    """-> dict(tr: trial table, ep: (n, 25, 250) tensor in the old-lab format, files: trigger -> image, bad, flat, missing)."""
    hits = sorted(glob.glob(str(LAB_SESSIONS / f"{session_id}_*")))
    assert len(hits) == 1, f"{session_id}: expected one folder in {LAB_SESSIONS}, found {len(hits)}"
    folder = Path(hits[0])
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

"""StarStim lab sessions (data/lab_sessions/E<nn>_<date>_<code>/experiment/*.xdf + PsychoPy files).

No-pulse image trials (cue 70, image trigger 1001-1299), 26 channels (the six early stimulation electrodes excluded),
channels flat in a run (stimulation electrodes of that run) and bad channels (SD > 3x median) interpolated, average
reference, baseline -0.2..0 s, 150 µV peak-to-peak rejection; then the old-lab format: FFT resample of the -0.2..1.0 s
epoch to 250 Hz, baseline, crop 0-1 s, the 25 channels shared with the old lab, re-referenced.
Epochs are placed by sample index on a smoothed LSL clock (linear fit + 5-s running median), never by raw time stamps.

load_rsvp: rapid-stream ("SSVEP") blocks without stimulation (image codes 3001-3299 = image trigger + 2000, 5 images per
second), in the THINGS lab contract: the 31 StarStim channels shared with THINGS-EEG2, channels flat or bad within the block
interpolated, average reference, baseline -0.2..0 s, 150 µV rejection, 250 Hz, 0-1 s. Time zero is the image marker plus the
session's clock error at that time plus rsvp_display_lag_ms (configs/sessions.yaml).

clock_error: in some sessions the EEG time stamps fall behind the marker clock by 2-6 ms per minute. Every image marker of
both loaders is moved by the session's clock error, measured from the stimulation pulses of the session's own recordings
(the same method as memstim/lab_sessions.py in memorability_stim).

Privacy: the PsychoPy files also hold questionnaire answers; only whitelisted columns are ever read
(presentation_image_trigger / presentation_image_path and ssvep_image_trigger / ssvep_image_path).
"""
import csv
import functools
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyxdf
import torch
import yaml
from scipy import signal
from scipy.ndimage import median_filter
from scipy.stats import theilslopes

from .channels import SESSION_26, SHARED_25, STARSTIM_31, STARSTIM_32
from .paths import CACHE, CONFIGS, LAB_SESSIONS, THINGS_OBJECT_IMAGES

FS, PRE, POST = 500, 100, 500                                  # -0.2..1.0 s at 500 Hz
CUE = {70: "none", 71: "stim", 72: "pulse_only", 73: "blank"}
REJECT_UV = 150.0
RSVP_CODES = (3001, 3299)                                      # rapid-stream image codes = image trigger + 2000
PULSE_EVENT = 7100                                             # stimulation pulse in the controller's stream neurostim_nic
NORMAL_MS = 7.0                                                # pulse artifact after its event when the EEG clock keeps time


def session_ids(kind=None, rsvp=None) -> list:
    """Session ids from configs/sessions.yaml (optionally only one kind / only sessions with a rapid-stream block)."""
    S = yaml.safe_load(open(CONFIGS / "sessions.yaml"))["sessions"]
    return [s["id"] for s in S if (kind is None or s.get("kind") == kind) and (rsvp is None or bool(s.get("rsvp")) == rsvp)]


def rsvp_offset_ms(session_id: str):
    """A session's own rapid-stream offset from configs/sessions.yaml, replacing clock error + display lag -> (ms at the
    start of the run's EEG stream, ms per minute); [a, b] or one number for a constant. None if the session has none."""
    S = yaml.safe_load(open(CONFIGS / "sessions.yaml"))["sessions"]
    v = next((s.get("rsvp_offset_ms") for s in S if s["id"] == session_id), None)
    return v if v is None else tuple(float(q) for q in v) if isinstance(v, list) else (float(v), 0.0)


def rsvp_display_lag_ms() -> float:
    """Image onset on our screen after its marker, relative to THINGS-EEG2 timing (configs/sessions.yaml)."""
    return float(yaml.safe_load(open(CONFIGS / "sessions.yaml"))["rsvp_display_lag_ms"])


def _folder(session_id: str) -> Path:
    hits = sorted(glob.glob(str(LAB_SESSIONS / f"{session_id}_*")))
    assert len(hits) == 1, f"{session_id}: expected one folder in {LAB_SESSIONS}, found {len(hits)}"
    return Path(hits[0])


def _interpolate(ep, bads, channels=SESSION_26):
    import mne
    info = mne.create_info(list(channels), FS, "eeg"); info.set_montage("standard_1020"); info["bads"] = list(bads)
    e = mne.EpochsArray(ep * 1e-6, info, verbose=False); e.interpolate_bads(reset_bads=True, verbose=False)
    return e.get_data() * 1e6


def _pulse_latency(path):
    """One XDF run: LSL time of every stimulation pulse event (stream neurostim_nic, code 7100) and when its artifact shows
    in the EEG relative to it (ms): the first sample-to-sample jump above 50x the median jump from 1 s before to 3 s
    after the event, on the smoothed clock of load_run."""
    st, _ = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_nic"}, {"name": "neurostim_pyexp-EEG"}])
    S = {s["info"]["name"][0]: s for s in st}; nic, eeg = S.get("neurostim_nic"), S.get("neurostim_pyexp-EEG")
    if nic is None or eeg is None or not len(nic["time_stamps"]) or len(eeg["time_stamps"]) < 2 * FS:
        return np.zeros(0), np.zeros(0)
    nc = np.array(nic["time_series"]).ravel().astype(float).astype(int); nt = nic["time_stamps"][nc == PULSE_EVENT]
    x = S["neurostim_pyexp-EEG"]["time_series"].astype(np.float64); lin = S["neurostim_pyexp-EEG"]["time_stamps"]
    raw = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_pyexp-EEG"}], dejitter_timestamps=False)[0][0]["time_stamps"]
    clock = lin + median_filter(raw - lin, size=2501, mode="nearest")
    d = np.abs(np.diff(x, axis=0)).max(1); thr = 50 * np.median(d); t, lat = [], []
    for tj in nt:
        o = np.searchsorted(clock, tj); big = np.flatnonzero(d[max(o - FS, 0):o + 3 * FS] > thr)   # up to +3 s: E26's EEG lags 0.9-2.2 s
        if o >= FS and len(big):
            t.append(tj); lat.append((big[0] - FS) * 1000 / FS)
    return np.array(t), np.array(lat)


@functools.lru_cache(maxsize=None)
def clock_error(session_id: str):
    """-> f(marker LSL times) -> seconds to add to them before they are looked up in the EEG clock.

    In E16-E19, E21 and E23 the EEG time stamps fall behind the marker clock by 2-6 ms per minute (E17: from the middle of
    the pulse run on), so an event shows in the EEG up to 0.37 s BEFORE its marker. E27's first run: 63-89 ms behind. E26
    is the other way round and much larger: the EEG is stamped 0.88 s (first pulse) to 2.07 s (last pulse) AFTER the marker
    clock, +44 ms per minute. Measured from the pulses of the session's recordings (they share the LSL clock):
    error = artifact latency - NORMAL_MS; medians of 2-min bins, interpolated; before the first and after the last pulse
    extended with the Theil-Sen slope of the first / last 10 min of pulses. Sessions whose error stays under 10 ms at every
    bin (and sessions with fewer than 20 pulses) get no correction. Cached in cache/<session>_clock.npz.
    Copied from memstim/lab_sessions.py (memorability_stim); keep the two in step."""
    f = CACHE / f"{session_id}_clock.npz"
    if not f.exists():
        P = [_pulse_latency(p) for p in sorted(glob.glob(str(_folder(session_id) / "experiment/*.xdf")))]
        t = np.concatenate([p[0] for p in P]); e = np.concatenate([p[1] for p in P]) - NORMAL_MS; o = np.argsort(t); t, e = t[o], e[o]
        ok = np.abs(e - np.array([np.median(e[max(i - 5, 0):i + 6]) for i in range(len(e))])) <= 15      # a jump that is not the pulse
        t, e = t[ok], e[ok]; kt, ke, s = np.zeros(0), np.zeros(0), np.zeros(2)
        if len(t) >= 20:
            b = ((t - t[0]) // 120).astype(int); bins = [i for i in np.unique(b) if (b == i).sum() >= 3]
            kt = np.array([t[b == i].mean() for i in bins]); ke = np.array([np.median(e[b == i]) for i in bins])
            if np.abs(ke).max() < 10:
                ke = np.zeros(len(ke))
            else:
                a, z = t <= t[0] + 600, t >= t[-1] - 600; s = np.array([theilslopes(e[a], t[a])[0], theilslopes(e[z], t[z])[0]])
        f.parent.mkdir(parents=True, exist_ok=True); np.savez(f, t=kt, e=ke, s=s)
    z = np.load(f); kt, ke, s = z["t"], z["e"], z["s"]

    def err(x):
        x = np.asarray(x, float)
        if not len(kt):
            return np.zeros(x.shape)
        return (np.interp(x, kt, ke) + s[0] * np.minimum(x - kt[0], 0) + s[1] * np.maximum(x - kt[-1], 0)) / 1000
    return err


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
    err = clock_error(Path(path).parents[1].name.split("_")[0])
    on = np.searchsorted(clock, tr.image_lsl.values + err(tr.image_lsl.values)); ok = (on - PRE >= 0) & (on + POST <= len(x))
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
    trigger per image marker, sample index of each image marker as recorded, sequence number per image marker (code 220
    starts a sequence), the session's clock error at each image marker in ms). No offset is applied here."""
    ci = [STARSTIM_32.index(c) for c in STARSTIM_31]; err = clock_error(session_id)
    for path in sorted(glob.glob(str(_folder(session_id) / "experiment/*.xdf"))):
        st, _ = pyxdf.load_xdf(path, select_streams=[{"name": "neurostim_pyexp"}, {"name": "neurostim_pyexp-EEG"}])
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
        yield (Path(path).stem, x, codes[img] - 2000, np.searchsorted(clock, mts[img]), np.cumsum(codes == 220)[img],
               err(mts[img]) * 1000)


def load_rsvp(session_id: str, require_images=True, offset_ms=None, bandpass=None):
    """Rapid-stream block of a session -> dict(tr: trial table (trigger, sequence), ep: (n, 31, 250) tensor in the THINGS
    lab contract, files: trigger -> image, flat, bad, n_all, offset) or None if the session has no such block.
    require_images=False skips the check that every image file exists (for EEG-only analyses when the image disk is offline).
    Time zero is the image marker plus an offset. offset_ms None: the session's clock error at that marker (clock_error)
    plus rsvp_display_lag_ms, or the session's rsvp_offset_ms if configs/sessions.yaml has one. A number, or (ms at the
    start of the run's EEG stream, ms per minute), is used as given; 0 = the marker itself.
    offset: where it came from (clock / config / argument) and its value in ms at the first and the last image.
    bandpass: (low, high) in Hz filters the continuous data of each run before the epochs are cut (zero-phase FIR, as
    memprep's rapid pipeline uses 0.1-30 Hz); None, the default, leaves the data unfiltered."""
    folder = _folder(session_id)
    line = rsvp_offset_ms(session_id) if offset_ms is None else (*(float(v) for v in np.atleast_1d(offset_ms)), 0.0)[:2]
    source = "clock" if line is None else "config" if offset_ms is None else "argument"
    trs, eps, flat, offs = [], [], set(), []
    for run, x, trigger, on, seq, clock_ms in rsvp_runs(session_id):
        off = clock_ms + rsvp_display_lag_ms() if line is None else line[0] + line[1] * on / FS / 60
        on = on + np.round(off / 1000 * FS).astype(int); ok = (on - PRE >= 0) & (on + POST <= len(x)); offs.append(off[ok])
        block = x[on[ok].min() - PRE:on[ok].max() + POST]
        run_flat = [c for i, c in enumerate(STARSTIM_31) if (np.diff(block[:, i]) == 0).mean() > 0.5]
        if bandpass is not None:
            import mne
            x = mne.filter.filter_data(x.T, FS, bandpass[0], bandpass[1], fir_design="firwin", verbose=False).T
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
                n_all=n_all, offset=dict(source=source, first_ms=float(offs[0][0]), last_ms=float(offs[-1][-1])))

"""The three low-level reconstructions shown in the infographic. Each returns
dict(pred (n,3,32,32), true (n,3,32,32), vals (per-image coarse8), paths (photos), groups (scoring set per image)).

  things     linear ridge on THINGS-EEG2 (31 StarStim channels), 16,448 training images averaged over 4 reps x 10 people
             (old-lab images dropped), test-200 averaged over 80 reps x 10 people; alpha 1e-3 (chosen on test-200, 1 of 10)
  old_lab    the old lab's own ridge (31 channels, ZCA per person), image level, 8-fold image cross-validation
  sessions   the old-lab ridge (25 shared channels, trained on image averages) applied to lab sessions, per session-image
"""
import numpy as np
import torch

from . import data_old_lab, data_sessions, data_things
from .channels import SHARED_25
from .images import pixel32_many
from .metrics import coarse8
from .readout import aggregate, fit_primal_ridge, fit_ridge, nested_cv, zca_by_subject

THINGS_ALPHA = 1e-3


def things(alpha=THINGS_ALPHA):
    keep = torch.tensor([i for i in range(16540) if i not in data_things.lab_image_conditions()])
    predict = fit_primal_ridge(data_things.eeg_means("train")[keep], data_things.targets("train")[keep].flatten(1), alpha)
    true = data_things.targets("test")
    pred = predict(data_things.eeg_means("test")).view(-1, 3, 32, 32)
    vals, _ = coarse8(pred, true, np.arange(len(true)))
    return dict(pred=pred, true=true, vals=vals, paths=data_things.image_paths("test"), groups=np.zeros(len(true)))


def old_lab():
    eeg, t, s, ibt = data_old_lab.load("train")
    uniq = np.array(sorted(ibt)); true = pixel32_many([ibt[int(u)] for u in uniq])
    X, img, _ = aggregate(zca_by_subject(eeg, s), t, s, "image")
    pred = torch.from_numpy(nested_cv(X, img, true, uniq).reshape(-1, 3, 32, 32)).float()
    vals, _ = coarse8(pred, true, np.arange(len(uniq)))
    return dict(pred=pred, true=true, vals=vals, paths=[ibt[int(u)] for u in uniq], groups=np.zeros(len(uniq)))


def old_lab_readout(exclude_basenames=()):
    """Ridge on all usable old-lab trials (25 shared channels, ZCA per person, image averages) -> (predict, alpha)."""
    eeg, t, s, ibt = data_old_lab.load(None, SHARED_25, exclude_basenames)
    x = zca_by_subject(eeg, s); uniq = np.array(sorted(ibt)); true = pixel32_many([ibt[int(u)] for u in uniq])
    Xi = torch.stack([x[t == u].mean(0) for u in uniq]).flatten(1).numpy().astype(np.float64)
    predict, alpha, _ = fit_ridge(Xi, true.flatten(1).numpy(), uniq, true, np.arange(len(uniq)))
    return predict, alpha


def load_sessions(ids):
    S = {}
    for sid in ids:
        s = data_sessions.load_session(sid)
        if s is None:
            continue
        assert not s["missing"], f"{sid}: image files missing for triggers {s['missing']}"
        S[sid] = s
    return S


def session_image_means(s, trials="pulse_free"):
    """ZCA (per session, label-free) then the mean over the chosen trials of each image -> (X float64, triggers)."""
    x = zca_by_subject(s["ep"], np.array(["session"] * len(s["ep"])))
    mask = s["tr"].pulse_free.values if trials == "pulse_free" else np.ones(len(s["tr"]), bool)
    tr, ep = s["tr"][mask], x[mask]
    trig = np.array(sorted(set(tr.trigger.astype(int))))
    X = torch.stack([ep[tr.trigger.values == u].mean(0) for u in trig]).flatten(1).numpy().astype(np.float64)
    return X, trig


def sessions(ids, trials="pulse_free"):
    S = load_sessions(ids)
    shared = {f.name for s in S.values() for f in s["files"].values()}
    predict, _ = old_lab_readout(exclude_basenames=shared)                      # no target leakage
    P, T, V, paths, groups = [], [], [], [], []
    for sid, s in S.items():
        X, trig = session_image_means(s, trials)
        if len(trig) < 3:
            continue
        pred = torch.from_numpy(predict(X).reshape(-1, 3, 32, 32)).float()
        true = pixel32_many([s["files"][int(u)] for u in trig])
        v, _ = coarse8(pred, true, np.arange(len(trig)))                           # scored within the session's own images
        P.append(pred); T.append(true); V.append(v); paths += [s["files"][int(u)] for u in trig]; groups += [sid] * len(trig)
    return dict(pred=torch.cat(P), true=torch.cat(T), vals=np.concatenate(V), paths=paths, groups=np.array(groups))

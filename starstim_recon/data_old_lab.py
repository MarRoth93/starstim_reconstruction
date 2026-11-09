"""Old lab data (May-June 2026): StarStim 32, 6 people (S01-S06), 160 THINGS images shown alone for ~4 s.

data/old_lab/lab_starstim31_epochs_250hz.npz: raw microvolts, average reference over the 31 channels, baseline -0.2..0 s,
0-1 s at 250 Hz. manifests/old_lab_trials.tsv (in git, pseudonymised): one row per epoch with its image (relative to
data/things), subject code, split (train 128 images / val 32 images) and usability.
"""
import csv
from pathlib import Path

import numpy as np
import torch

from .channels import STARSTIM_31
from .paths import OLD_LAB_EPOCHS, OLD_LAB_MANIFEST, THINGS


def load(split=None, channels=STARSTIM_31, exclude_basenames=()):
    """-> (eeg (n, C, 250) float32, image target index per trial, subject code per trial, {target index: image path}).

    split: 'train', 'val' or None (both). Channels are picked by name; a subset is re-referenced to its own average.
    """
    rows = [r for r in csv.DictReader(open(OLD_LAB_MANIFEST), delimiter="\t")
            if r["usable"] == "True" and (split is None or r["split"] == split)
            and Path(r["image"]).name not in set(exclude_basenames)]
    z = np.load(OLD_LAB_EPOCHS, allow_pickle=True)
    names = z["ch_names"].astype(str).tolist()
    eeg = z["eeg"][[int(r["epoch_index"]) for r in rows]]
    if sorted(channels) != sorted(names):
        eeg = eeg[:, [names.index(c) for c in channels]]
        eeg = eeg - eeg.mean(1, keepdims=True)
    targets = np.array([int(r["target_index"]) for r in rows])
    subjects = np.array([r["subject"] for r in rows])
    image_by_target = {int(r["target_index"]): THINGS / r["image"] for r in rows}
    return torch.from_numpy(np.ascontiguousarray(eeg)).float(), targets, subjects, image_by_target

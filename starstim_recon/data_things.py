"""THINGS-EEG2 through the StarStim lab contract (31 channels, average reference, per-subject ZCA x1.1, 0-1 s, 250 Hz).

Files in data/things_eeg2_labcontract/: sub-XX_starstim31_eeg.pt (16540 images x 4 reps x 31 x 250, fp16),
sub-XX_starstim31_eeg_test.pt (200 x 80 x 31 x 250), lab_stimulus_exclusion.json, targets_pixel32[_test].pt (used only for
the canonical image order in `relative_paths`; the pixel targets themselves are recomputed with images.pixel32).
"""
import json

import torch

from .images import pixel32_many
from .paths import CACHE, THINGS, THINGS_EEG

N_SUBJECTS = 10
_IMAGE_DIR = {"train": "training_images", "test": "test_images"}


def eeg_means(split: str) -> torch.Tensor:
    """(n_images, 31 * 250) float32: each image averaged over all repetitions and all 10 subjects."""
    total = None
    for sub in range(1, N_SUBJECTS + 1):
        eeg = torch.load(THINGS_EEG / f"sub-{sub:02d}_starstim31_eeg{'_test' if split == 'test' else ''}.pt",
                         weights_only=False)["eeg"]
        m = eeg.float().mean(1)
        total = m if total is None else total + m
        del eeg
    return (total / N_SUBJECTS).flatten(1)


def lab_image_conditions() -> set:
    """THINGS training conditions that are also old-lab stimuli (dropped from any THINGS fit)."""
    return set(json.load(open(THINGS_EEG / "lab_stimulus_exclusion.json"))["train_condition_indices"])


def image_paths(split: str) -> list:
    rel = torch.load(THINGS_EEG / f"targets_pixel32{'_test' if split == 'test' else ''}.pt", weights_only=False)["relative_paths"]
    return [THINGS / "image_set" / _IMAGE_DIR[split] / r for r in rel]


def targets(split: str) -> torch.Tensor:
    """(n_images, 3, 32, 32) pixel32 targets in EEG order, cached in cache/ after the first call."""
    path = CACHE / f"things_pixel32_{split}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    t = pixel32_many(image_paths(split))
    CACHE.mkdir(parents=True, exist_ok=True); torch.save(t, path)
    return t

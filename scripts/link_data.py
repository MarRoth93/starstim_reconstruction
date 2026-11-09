#!/usr/bin/env python3
"""Create the symlinks in data/ from configs/paths.local.yaml (absolute paths on this machine), then check the layout.

    cp configs/paths.example.yaml configs/paths.local.yaml   # edit the four paths
    python scripts/link_data.py                              # link + check
    python scripts/link_data.py --check                      # check only
"""
import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon.data_sessions import session_ids  # noqa: E402
from starstim_recon.paths import CONFIGS, LAB_SESSIONS, OLD_LAB_EPOCHS, THINGS, THINGS_EEG  # noqa: E402

LINKS = {"things": THINGS, "things_eeg2_labcontract": THINGS_EEG, "old_lab_epochs": OLD_LAB_EPOCHS,
         "lab_sessions": LAB_SESSIONS}


def link():
    cfg = yaml.safe_load(open(CONFIGS / "paths.local.yaml"))
    for key, dst in LINKS.items():
        if not cfg.get(key):
            print(f"skip {key}: not set"); continue
        src = Path(cfg[key]).expanduser()
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            if Path(os.readlink(dst)) == src:
                continue
            dst.unlink()
        elif dst.exists():
            print(f"keep {dst}: a real file/folder is already there"); continue
        dst.symlink_to(src); print(f"linked {dst.relative_to(dst.parents[1])} -> {src}")


def check():
    ok = True

    def need(path, what):
        nonlocal ok
        if not path.exists():
            ok = False; print(f"MISSING {what}: {path}")
    need(THINGS / "image_set" / "test_images", "THINGS test images")
    need(THINGS / "image_set" / "training_images", "THINGS training images")
    need(THINGS / "images_THINGS" / "object_images", "THINGS object images")
    for sub in range(1, 11):
        for suffix in ("", "_test"):
            need(THINGS_EEG / f"sub-{sub:02d}_starstim31_eeg{suffix}.pt", "THINGS lab-contract EEG")
    for f in ("targets_pixel32.pt", "targets_pixel32_test.pt", "lab_stimulus_exclusion.json"):
        need(THINGS_EEG / f, "THINGS lab-contract file")
    need(OLD_LAB_EPOCHS, "old-lab epochs")
    for sid in session_ids():
        hits = list(LAB_SESSIONS.glob(f"{sid}_*"))
        if len(hits) != 1:
            ok = False; print(f"MISSING session {sid}: {len(hits)} folders match {LAB_SESSIONS}/{sid}_*")
    print("data layout OK" if ok else "data layout incomplete (see above)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--check", action="store_true"); args = ap.parse_args()
    if not args.check:
        link()
    sys.exit(0 if check() else 1)

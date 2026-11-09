#!/usr/bin/env python3
"""Build manifests/old_lab_trials.tsv (in git) from the reconstruction repo's lab split manifest (not in git: it holds names).

    python scripts/build_old_lab_manifest.py <Hierarchical_EEG2Image_Reconstruction>/lab_data/derived/live_preproc/lab_finetune_split_manifest.tsv

Keeps only what the analyses need, pseudonymises the people (S01-S06, sorted by their folder name) and stores each image
relative to the THINGS root (data/things), so no name and no machine-specific path enters git.
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from starstim_recon.paths import OLD_LAB_MANIFEST  # noqa: E402

COLUMNS = ["epoch_index", "target_index", "split", "usable", "subject", "image"]


def main(source: Path):
    rows = list(csv.DictReader(open(source), delimiter="\t"))
    code = {s: f"S{i + 1:02d}" for i, s in enumerate(sorted({r["subject_folder"] for r in rows}))}
    out = []
    for r in rows:
        path = r["exact_image_path"].replace("\\", "/")
        assert "/THINGS/" in path, path
        out.append(dict(epoch_index=r["epoch_index"], target_index=r["target_index"], split=r["split"],
                        usable=str(str(r["usable_for_finetune"]).lower() == "true"), subject=code[r["subject_folder"]],
                        image=path.split("/THINGS/", 1)[1]))
    OLD_LAB_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OLD_LAB_MANIFEST, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t"); w.writeheader(); w.writerows(out)
    print(f"wrote {OLD_LAB_MANIFEST}: {len(out)} epochs, {len(code)} people, "
          f"{sum(r['usable'] == 'True' for r in out)} usable")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

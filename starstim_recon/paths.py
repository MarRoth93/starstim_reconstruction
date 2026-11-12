"""Every path the code uses, relative to the repository root.

data/ holds symlinks made by scripts/link_data.py from configs/paths.local.yaml (or real folders placed there by hand),
so no absolute path appears anywhere in the code.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
THINGS = DATA / "things"                          # THINGS dataset root: image_set/{training,test}_images, images_THINGS/object_images
THINGS_EEG = DATA / "things_eeg2_labcontract"     # THINGS-EEG2 through the StarStim lab contract: sub-XX_starstim31_eeg[_test].pt, ...
OLD_LAB_EPOCHS = DATA / "old_lab" / "lab_starstim31_epochs_250hz.npz"   # May-June 2026 StarStim recordings, 0-1 s epochs
LAB_SESSIONS = DATA / "lab_sessions"              # StarStim session folders E<nn>_<date>_<code> (XDF + PsychoPy files)
THINGS_PIXEL_NETWORK = DATA / "things_pixel_network" / "best.pt"        # optional: THINGS-pretrained pixel network (things_network.py)

THINGS_OBJECT_IMAGES = THINGS / "images_THINGS" / "object_images"
THINGS_TEST_IMAGES = THINGS / "image_set" / "test_images"

CONFIGS = ROOT / "configs"
MANIFESTS = ROOT / "manifests"
OLD_LAB_MANIFEST = MANIFESTS / "old_lab_trials.tsv"
CACHE = ROOT / "cache"                            # regenerable intermediate files (gitignored)
RESULTS = ROOT / "results"                        # figures and result files (gitignored)

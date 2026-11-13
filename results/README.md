# Results

Every result of the project as of 2026-11-13, so they can be read without running anything. The files were written by
the scripts in `scripts/` on the lab workstation; how they were made, and what to keep in mind when reading them, is in
[`../HANDOVER.md`](../HANDOVER.md) and [`../README.md`](../README.md).

**What is reconstructed:** the coarse 8x8 colour layout of a viewed image, from StarStim 32 EEG. **Score:** coarse8, the
correlation of the reconstructed layout with the true one relative to the other candidate images (0 = chance, 1 =
perfect). THINGS-EEG2 reaches +0.44 on the same task; our best is about +0.07 to +0.09 (all eight usable recordings
averaged per image).

## Where to look first

1. [`infographic_results.png`](infographic_results.png): the overview of all rapid-stream results on one page.
2. [`sheets/`](sheets): every reconstruction as an image. Each image is one block: its true 8x8 layout, then the
   reconstruction of each decoder with that image's score below it. The reconstructions are contrast-enhanced for display.

| Sheet | Images | Decoders |
|---|---|---|
| [`sheets/all_recordings_bp0.1-30.png`](sheets/all_recordings_bp0.1-30.png) | the 262 images of the 8 usable recordings, averaged over their recordings, 0.1-30 Hz band-pass | own decoder, THINGS decoder, THINGS as prior, fine-tuned network |
| [`sheets/all_recordings.png`](sheets/all_recordings.png) | the same without the band-pass | the same |
| `sheets/rsvp_E21.png` ... `sheets/rsvp_E31.png` | the 144 images of one recording, decoded from that recording alone | own decoder, THINGS decoder, THINGS as prior, old-lab decoder |

E26 is not usable (its timing could not be verified); its sheet is there and says so. The fine-tuned network was
pretrained on 173 of the 262 images (marked * on the sheet), so its score on those is not free of leakage; on the 80
images it never saw it scores +0.066 with the band-pass, level with the linear THINGS prior (+0.069).

## The files

| Files | Content | Written by |
|---|---|---|
| `infographic_results.{png,pdf}`, `infographic_results_numbers.json` | the overview figure and every number in it | `make_results_infographic.py` |
| `infographic_lowlevel.{png,pdf}`, `infographic_numbers.json` | the earlier overview: THINGS-EEG2, old lab, slow trials E06-E20 | `make_infographic.py` |
| `rsvp_<session>.json` | scores of one recording, corrected timing; `_offset0` = time zero at the image marker, `_bp0.1-30` = band-pass | `run_rsvp_reconstruction.py` |
| `rsvp_<session>_predictions.npz` | the 32x32 reconstruction of every image per decoder, and its coarse8 | same |
| `rsvp_timing_<session>.json` | the timing check of one recording; decides whether it is usable | `check_rsvp_timing.py` |
| `rsvp_pooled.json`, `rsvp_pooled[_bp0.1-30]_predictions.npz` | all usable recordings together: scores, and the reconstructions on 31 and on 10 posterior channels | `run_rsvp_pooled.py` |
| `rsvp_finetune.json`, `rsvp_finetune[_bp0.1-30]_predictions.npz` | the THINGS-pretrained network on the same data: scores of every variant, reconstructions of the fine-tuned ensemble | `run_rsvp_finetune.py` |
| `rsvp_learning_curves.json` | the pooled decoders trained on fewer images and fewer recordings per image | `run_rsvp_learning_curves.py` |
| `transfer_sessions.json`, `things_prior.json`, `global_props.json` | R5, R6, R7 on the old lab data and the slow trials | `run_transfer_sessions.py`, `run_things_prior.py`, `run_global_props.py` |
| `explore_rsvp_E21.json` | an exploratory screen of decoder variants on E21 (not a result) | `explore_rsvp_improvements.py` |
| `sheets/*.png` | every reconstruction as an image | `make_reconstruction_sheets.py` |

To use the reconstructions in Python (numpy only, no GPU):

```python
import numpy as np
P = np.load("results/rsvp_pooled_bp0.1-30_predictions.npz")
P["image"]                 # (262,) THINGS file names, e.g. "acorn/acorn_02n.jpg"
P["things_prior"]          # (262, 3, 32, 32) RGB reconstruction in [0, 1]; "_posterior10" keys = 10 posterior channels
P["coarse8_things_prior"]  # (262,) score of every image; the mean is the reported score
```

Not included: the per-recording example figures (`rsvp_*_examples.png`), which show THINGS photos. The THINGS images are
not ours to redistribute; the sheets name each image by its THINGS file name instead.

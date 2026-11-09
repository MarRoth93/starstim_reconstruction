# data/

Nothing in this folder is committed except this file. `python scripts/link_data.py` fills it with symlinks from
`configs/paths.local.yaml`; you can also place real folders here by hand. The code only ever reads these paths:

| Path | Content | Provenance |
|---|---|---|
| `things/` | THINGS dataset root: `image_set/training_images/`, `image_set/test_images/` (THINGS-EEG2 stimuli), `images_THINGS/object_images/` (all THINGS images, used by the lab sessions) | THINGS (Hebart et al. 2019) and THINGS-EEG2 (Gifford et al. 2022) image sets |
| `things_eeg2_labcontract/` | `sub-01..10_starstim31_eeg.pt` (16540 images x 4 reps x 31 ch x 250 samples, fp16) and `_test.pt` (200 x 80 x 31 x 250); `lab_stimulus_exclusion.json` (THINGS conditions that are old-lab stimuli); `targets_pixel32[_test].pt` (only its `relative_paths`, the canonical image order, is used) | raw THINGS-EEG2 rebuilt through the StarStim lab contract (31 StarStim channels, average reference, per-subject ZCA x1.1, 0-1 s, 250 Hz) by `scripts/build_things_labcontract_caches.py` in `Hierarchical_EEG2Image_Reconstruction` (branch `marco_new_tests`, commit 745430d); exclusion list and image order by `scripts/build_things_lowlevel_caches.py` (commit e9c83fa) |
| `old_lab/lab_starstim31_epochs_250hz.npz` | old lab recordings (May-June 2026): 1200 epochs x 31 ch x 250 samples, raw µV, average reference, baseline -0.2..0 s, 0-1 s | `scripts/build_lab_live_preproc_cache.py` in `Hierarchical_EEG2Image_Reconstruction` (branch `marco_new_tests`, commits 4fb6961 / 4010cd8) from the NIC2 `.easy` files |
| `lab_sessions/` | StarStim session folders `E<nn>_<date>_<code>/` with `experiment/*.xdf` (LabRecorder) and the PsychoPy files | lab recordings (memorability_stim sessions; future reconstruction blocks) |

The trial list of the old lab data is `manifests/old_lab_trials.tsv` (in git, pseudonymised, images relative to `things/`),
built by `scripts/build_old_lab_manifest.py`.

**Privacy:** the session folders contain questionnaire answers (PsychoPy `.csv` / `.psydat`). The code reads only the
columns `presentation_image_trigger` and `presentation_image_path`; never copy those files into the repository.

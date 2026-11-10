# starstim_recon

Low-level EEG image reconstruction (the coarse 8x8 colour layout of a viewed image; no detail, no semantics) for
recordings with a StarStim 32 cap, benchmarked against THINGS-EEG2.

## Setup

```bash
# Python 3.10+ with the packages in pyproject.toml (pip install -e .)
cp configs/paths.example.yaml configs/paths.local.yaml   # set the four data locations on your machine
python scripts/link_data.py                              # creates the symlinks in data/ and checks the layout
```

All code reads data only through `data/` (see `data/README.md` for the expected content and its provenance), so the
repository works on any machine that has the THINGS images, the THINGS-EEG2 lab-contract cache, the old lab epochs and
the session folders.

## Reproduce the infographic

```bash
python scripts/make_infographic.py              # -> results/infographic_lowlevel.{png,pdf}, results/infographic_numbers.json
```

The first run computes the three reconstructions and caches them in `cache/infographic_data.pt` (several minutes);
later runs only redraw. Expected numbers (mean coarse8; 0 = chance, 1 = perfect):

| Dataset | Images | coarse8 |
|---|---|---|
| THINGS-EEG2, 31 StarStim channels, test images averaged over 80 reps x 10 people | 200 | +0.440 |
| Old lab, ridge trained on the lab data (8-fold image cross-validation) | 128 | +0.079 |
| Lab sessions E06-E20, old-lab ridge, pulse-free blocks | 283 | -0.010 |

The infographic's session panel is pinned to E06-E20 (`STIM_PANEL_SESSIONS` in `scripts/make_infographic.py`), so
registering new sessions does not change it.

## Add new data

1. Put the session folder (`E<nn>_<date>_<code>/` with `experiment/*.xdf` + PsychoPy files) into the `lab_sessions` location.
2. Add a line to `configs/sessions.yaml` (`kind: recon` for a dedicated reconstruction block without stimulation).
3. `python scripts/link_data.py --check`, then rerun the analyses / `python scripts/make_infographic.py --recompute`.
4. For a session with a rapid-stream block (`rsvp: true`): `python scripts/check_rsvp_timing.py --session E<nn>`, enter
   the suggested `rsvp_offset_ms` line in `configs/sessions.yaml`, then `python scripts/run_rsvp_reconstruction.py --session E<nn>`.

## Code

| File | Role |
|---|---|
| `starstim_recon/paths.py` | every data path, relative to the repository root |
| `starstim_recon/channels.py` | StarStim / THINGS channel sets (always index by name) |
| `starstim_recon/images.py` | `pixel32` (the one target definition for all datasets), 8x8 layout, global image properties |
| `starstim_recon/metrics.py` | coarse8, pairwise identification, permutation nulls, paired bootstrap |
| `starstim_recon/readout.py` | per-person ZCA, kernel and primal ridge, nested image-fold cross-validation |
| `starstim_recon/data_things.py`, `data_old_lab.py`, `data_sessions.py` | the three data sources |
| `starstim_recon/reconstruct.py` | the three reconstructions shown in the infographic |
| `scripts/make_infographic.py` | the infographic |
| `scripts/run_transfer_sessions.py` | old-lab ridge -> sessions, per session and pooled (R5) |
| `scripts/run_global_props.py` | 8 global image properties, per-person readouts (R7) |
| `scripts/run_things_prior.py` | THINGS ridge as a weight prior for the lab ridge (R6) |
| `scripts/run_rsvp_reconstruction.py` | reconstruction from a rapid-stream block (E21, E23): THINGS zero-shot, within-person, THINGS prior, old-lab ridge |
| `scripts/check_rsvp_timing.py` | EEG clock drift from the stimulation pulses and where the image-specific response sits relative to the image marker |
| `scripts/link_data.py`, `scripts/build_old_lab_manifest.py` | data setup |

## Rapid-stream sessions (2026-11-10)

E21 and E23 contain a block without stimulation in which 144 images are shown at 5 images per second in shuffled order.

**EEG clock drift.** In some sessions the EEG time stamps drift against the marker clock, linearly over the recording, so
that late in a session the response to an image starts well before its marker. The drift is measured from the stimulation
pulses of the same run (latency of the pulse artifact relative to the controller's pulse event; 6-10 ms without drift;
E06-E20 were measured with the same method in a one-off script, not with a script of this repository):

| Sessions | Drift (ms per minute) |
|---|---|
| E06-E15 (E09: too few pulses to tell), E20 | none (at most 0.2) |
| E16, E17, E19 | -2.3, -1.9, -2.3 |
| E18, E21, E23 | -5.8, -5.1, -3.6 |

`scripts/check_rsvp_timing.py` fits that line, compares it with the label-free estimate from the images (best match with
the THINGS-EEG2 responses to the same images, per quarter of the block; 0 ms for a THINGS subject against the others,
+20 ms for the old lab) and prints the `rsvp_offset_ms` line for `configs/sessions.yaml`. In E21 and E23 the two agree
up to a constant display lag of about 43 ms. `load_rsvp` cuts every epoch at marker + offset at that time. The cause of
the drift is unknown and nothing here is confirmed with a photodiode. The slow-trial loader (`load_session`) is not
corrected, so the session trials of E16-E19 used in R5 and R7 are mis-timed by up to about 300 ms.

| Session | Images x repeats | Within-person | THINGS zero-shot | THINGS prior | Within-person, time zero at the marker |
|---|---|---|---|---|---|
| E21 | 144 x 20 | +0.047 (p .013) | +0.018 (p .19) | +0.017 (p .20) | +0.081 (p .001) |
| E23 | 144 x 10 | +0.020 (p .18) | +0.060 (p .004) | +0.054 (p .006) | +0.014 (p .28) |

(`python scripts/run_rsvp_reconstruction.py --session E21`, with `--offset-ms 0` for the last column.) With the drift
removed the THINGS ridge transfers to E23 without any lab training data; for E21 it does not reach significance. The
marker column is higher for E21 most likely because the baseline window (-0.2..0 s) then contains the image response,
which the baseline subtraction copies into every sample.

## Results so far (2026-11-06)

| Step | Question | Result |
|---|---|---|
| R1 | Does the old lab data carry image signal? (lab ridge, old lab TRAIN split) | single trial +0.007, per person x image +0.017, image-averaged over 6 people +0.079 (p .0005) |
| R3 | Does a THINGS-pretrained pixel network transfer zero-shot when scored per image? | no: +0.020 (p .31) |
| R4 | Frozen THINGS features or fine-tuned adapters? | all below the plain lab ridge |
| R5 | Does the old-lab ridge transfer to the stimulation sessions E06-E20? | no: -0.010 (p .77) over 283 session-images (2-4 trials per image) |
| R6 | THINGS ridge as a weight prior? | worse than the plain lab ridge (+0.032 vs +0.079) |
| R7 | Global image properties, per-person readouts? | per-person ridges +0.066 mean r (p .02) within the old lab; transfer null |

R3 and R4 need the ATMS pixel networks of `Hierarchical_EEG2Image_Reconstruction`; their code stays there
(branch `dev/lab_lowlevel_transfer`). Bottleneck: too few repeats per image and too little data per person; the next step
is a dedicated reconstruction block without stimulation (many repeats of shared images, the same people across sessions).

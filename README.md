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
| Lab sessions E06-E20, old-lab ridge, pulse-free blocks | 283 | -0.011 |

The infographic's session panel is pinned to E06-E20 (`STIM_PANEL_SESSIONS` in `scripts/make_infographic.py`), so
registering new sessions does not change it.

## Add new data

1. Put the session folder (`E<nn>_<date>_<code>/` with `experiment/*.xdf` + PsychoPy files) into the `lab_sessions` location.
2. Add a line to `configs/sessions.yaml` (`kind: recon` for a dedicated reconstruction block without stimulation).
3. `python scripts/link_data.py --check`, then rerun the analyses / `python scripts/make_infographic.py --recompute`.
4. For a session with a rapid-stream block (`rsvp: true`): `python scripts/check_rsvp_timing.py --session E<nn>` checks
   the automatic timing correction, then `python scripts/run_rsvp_reconstruction.py --session E<nn>`.

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
| `scripts/check_rsvp_timing.py` | checks the automatic timing correction of a rapid-stream block against an estimate from the images |
| `scripts/link_data.py`, `scripts/build_old_lab_manifest.py` | data setup |

## Timing: EEG clock drift (2026-11-11)

In some sessions the EEG time stamps fall behind the marker clock, so that late in a session an event shows in the EEG
well before its marker. `data_sessions.clock_error` measures this for every session from the stimulation pulses of its own
recordings (latency of the pulse artifact relative to the controller's pulse event, minus the normal 7 ms; medians of
2-min bins, interpolated, extended past the first and last pulse with the slope of the first / last 10 min). Both loaders
move every image marker by it; a session whose error stays under 10 ms gets no correction, and nothing is entered by hand.
The method and its thresholds are copied from `memstim/lab_sessions.py` in memorability_stim and give identical values for
the sessions both repositories read (E14-E25). The result is cached in `cache/<session>_clock.npz`.

| Sessions | Clock error, first -> last pulse bin (ms) |
|---|---|
| E06-E15, E20, E24, E25 | no correction (E09: too few pulses to measure) |
| E16 | -39 -> -85 |
| E17 | +3 -> -39 |
| E18 | -190 -> -307 |
| E19 | -49 -> -109 |
| E21 | -111 -> -243 |
| E23 | -97 -> -189 |

The cause of the drift is unknown. For the slow image trials the correction leaves the session transfer (R5) where it
was: -0.010 before, -0.011 after, with E06-E15 and E20 unchanged to four decimals.

## Rapid-stream sessions (2026-11-11)

E21 and E23 contain a block without stimulation in which 144 images are shown at 5 images per second in shuffled order.
Time zero of an epoch is the image marker plus the clock error at that time plus `rsvp_display_lag_ms`
(`configs/sessions.yaml`, 46 ms): the delay of the image on our screen relative to THINGS-EEG2 timing, which the pulses
cannot measure. `scripts/check_rsvp_timing.py` checks the result against an estimate from the images themselves (best
match with the THINGS-EEG2 responses to the same images, per quarter of the block; 0 ms for a THINGS subject against the
others, +20 ms for the old lab). In E21 and E23 the two agree in every quarter, and the remainder is 46 ms in both. Not
confirmed with a photodiode. A session without stimulation pulses cannot be corrected automatically; if it drifts, the
check prints an `rsvp_offset_ms` line to enter for that session.

| Session | Images x repeats | Within-person | THINGS zero-shot | THINGS prior | Within-person, time zero at the marker |
|---|---|---|---|---|---|
| E21 | 144 x 20 | +0.042 (p .024) | +0.012 (p .30) | +0.014 (p .25) | +0.081 (p .001) |
| E23 | 144 x 10 | +0.021 (p .18) | +0.052 (p .006) | +0.044 (p .024) | +0.014 (p .28) |

(`python scripts/run_rsvp_reconstruction.py --session E21`, with `--offset-ms 0` for the last column.) With the drift
removed the THINGS ridge transfers to E23 without any lab training data; for E21 it does not reach significance. These
scores move by about 0.01-0.02 when the offset changes by a few ms. The marker column is higher for E21 most likely
because the baseline window (-0.2..0 s) then contains the image response, which the baseline subtraction copies into
every sample.

## Results so far (2026-11-06)

| Step | Question | Result |
|---|---|---|
| R1 | Does the old lab data carry image signal? (lab ridge, old lab TRAIN split) | single trial +0.007, per person x image +0.017, image-averaged over 6 people +0.079 (p .0005) |
| R3 | Does a THINGS-pretrained pixel network transfer zero-shot when scored per image? | no: +0.020 (p .31) |
| R4 | Frozen THINGS features or fine-tuned adapters? | all below the plain lab ridge |
| R5 | Does the old-lab ridge transfer to the stimulation sessions E06-E20? | no: -0.011 (p .80) over 283 session-images (2-4 trials per image); -0.010 (p .77) before the clock correction |
| R6 | THINGS ridge as a weight prior? | worse than the plain lab ridge (+0.032 vs +0.079) |
| R7 | Global image properties, per-person readouts? | per-person ridges +0.066 mean r (p .02) within the old lab; transfer null |

R3 and R4 need the ATMS pixel networks of `Hierarchical_EEG2Image_Reconstruction`; their code stays there
(branch `dev/lab_lowlevel_transfer`). Bottleneck: too few repeats per image and too little data per person; the next step
is a dedicated reconstruction block without stimulation (many repeats of shared images, the same people across sessions).

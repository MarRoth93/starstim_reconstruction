# starstim_recon

Low-level EEG image reconstruction (the coarse 8x8 colour layout of a viewed image; no detail, no semantics) for
recordings with a StarStim 32 cap, benchmarked against THINGS-EEG2.

**New to the project, person or agent? Read [`HANDOVER.md`](HANDOVER.md) first:** what was tried, what worked and what did
not, the caveats behind the numbers, and where to continue. **To look at the results without running anything:**
[`results/README.md`](results/README.md), with every reconstruction as an image in [`results/sheets/`](results/sheets).

## Setup

```bash
# Python 3.10+ with the packages in pyproject.toml (pip install -e .)
cp configs/paths.example.yaml configs/paths.local.yaml   # set the four data locations on your machine (a fifth is optional)
python scripts/link_data.py                              # creates the symlinks in data/ and checks the layout
```

All code reads data only through `data/` (see `data/README.md` for the expected content and its provenance), so the
repository works on any machine that has the THINGS images, the THINGS-EEG2 lab-contract cache, the old lab epochs and
the session folders. Only `scripts/run_rsvp_finetune.py` needs a fifth, optional one: the checkpoint of the THINGS-pretrained
pixel network (`things_pixel_network`); `scripts/link_data.py --check` reports it when it is not linked and does not fail.

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
| `starstim_recon/reconstruct.py` | the three reconstructions shown in the infographic; `rsvp_pooled`, the rapid-stream recordings averaged per image |
| `starstim_recon/things_network.py` | the THINGS-pretrained pixel network (model code copied from `Hierarchical_EEG2Image_Reconstruction`) and the images it was trained on |
| `scripts/make_infographic.py` | the infographic |
| `scripts/run_transfer_sessions.py` | old-lab ridge -> sessions, per session and pooled (R5) |
| `scripts/run_global_props.py` | 8 global image properties, per-person readouts (R7) |
| `scripts/run_things_prior.py` | THINGS ridge as a weight prior for the lab ridge (R6) |
| `scripts/run_rsvp_reconstruction.py` | reconstruction from a rapid-stream block (E21, E23): THINGS zero-shot, within-person, THINGS prior, old-lab ridge |
| `scripts/check_rsvp_timing.py` | checks the automatic timing correction of a rapid-stream block against an estimate from the images |
| `scripts/run_rsvp_pooled.py` | one decoder on all usable rapid-stream recordings, averaged per image; with and without the band-pass and the posterior channels |
| `scripts/run_rsvp_finetune.py` | the THINGS-pretrained pixel network on the same pooled recordings: as it is, linear probes, fine-tuning; every score also split by the network's pretraining set |
| `scripts/run_rsvp_learning_curves.py` | the pooled own decoder and THINGS prior trained on fewer images and on fewer recordings per image, with a shuffled-EEG null |
| `scripts/make_results_infographic.py` | infographic of the current rapid-stream results and the timing correction, drawn from the result files |
| `scripts/make_reconstruction_sheets.py` | every reconstruction as an image: one sheet per recording and per filter for the pooled recordings (`results/sheets/`) |
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
| E26 | +919 -> +2055 (EEG stamped after the markers) |
| E27 | -63 -> +3 (first recording 63-89 ms behind, second on time) |
| E28 | -112 -> -260 |
| E29 | -3 -> -33 |
| E31 | +3 -> -16 |

The cause of the drift is unknown. For the slow image trials the correction leaves the session transfer (R5) where it
was: -0.010 before, -0.011 after, with E06-E15 and E20 unchanged to four decimals.

## Rapid-stream sessions (2026-11-11)

E21, E23-E29 and E31 contain a block without stimulation in which 144 images are shown at 5 images per second in shuffled
order. Time zero of an epoch is the image marker plus the clock error at that time plus `rsvp_display_lag_ms`
(`configs/sessions.yaml`, 46 ms): the delay of the image on our screen relative to THINGS-EEG2 timing, which the pulses
cannot measure. `scripts/check_rsvp_timing.py` removes the clock error and estimates from the images themselves what is
left (best match with the THINGS-EEG2 responses to the same images, per quarter of the block; 0 ms for a THINGS subject
against the others, +20 ms for the old lab). That remainder is constant within every usable recording (E21 +40, E23 +40 to
+60, E24 +60, E25 +40, E27 +20, E28 +60, E29 +40, E31 +40 ms) and counts as fitting within 40 ms of the constant. Not confirmed with a photodiode.
A session without stimulation pulses cannot be corrected automatically; if it drifts, the check prints an
`rsvp_offset_ms` line to enter for that session.

| Session | Images x repeats | Within-person | THINGS zero-shot | THINGS prior | Within-person, time zero at the marker |
|---|---|---|---|---|---|
| E21 | 144 x 20 | +0.042 (p .024) | +0.012 (p .30) | +0.014 (p .25) | +0.081 (p .001) |
| E23 | 144 x 10 | +0.021 (p .18) | +0.052 (p .006) | +0.044 (p .024) | +0.014 (p .28) |
| E24 | 144 x 20 | +0.051 (p .009) | +0.043 (p .022) | +0.035 (p .048) | +0.033 (p .051) |
| E25 | 144 x 5 | +0.050 (p .011) | -0.003 (p .57) | +0.005 (p .42) | +0.049 (p .013) |
| E27 | 144 x 10 | +0.031 (p .083) | -0.039 (p .97) | +0.000 (p .51) | +0.034 (p .055) |
| E28 | 144 x 20 | -0.014 (p .75) | +0.013 (p .27) | +0.001 (p .48) | -0.014 (p .74) |
| E29 | 144 x 5 | -0.025 (p .89) | +0.036 (p .044) | +0.031 (p .063) | -0.023 (p .88) |
| E31 | 144 x 10 | +0.053 (p .006) | +0.004 (p .42) | +0.022 (p .14) | +0.039 (p .033) |

E26 is not usable: its EEG is stamped 2.5-3.2 s after the markers during the block, the stimulation electrodes and three
occipital channels are flat or noisy, 845 of 2880 trials survive, and the check finds no image-specific signal.
(`python scripts/run_rsvp_reconstruction.py --session E21`, with `--offset-ms 0` for the last column;
`python scripts/make_results_infographic.py` draws the overview.) With the drift removed the THINGS ridge scores above
chance in E23, E24 and E29 without any lab training data, not in E21, E25, E27, E28 and E31. E28 and E29 do not decode with
their own decoder, although E28 has 20 repeats and a strong image-specific signal (split-half reliability .66). These scores move by about 0.01-0.02 when
the offset changes by a few ms. The marker column is higher for E21 most likely because the baseline window (-0.2..0 s)
then contains the image response, which the baseline subtraction copies into every sample.

E31 (recorded 2026-11-12, added 2026-11-13) shows the image set of E23, E28 and E29 again, so it adds repeats, not
images. E32 (2026-11-13) has no rapid-stream block: only slow image trials during stimulation (128 images, 96 of them
shown once) and OASIS images, so it is not registered.

## All usable recordings together (2026-11-12, with E31 since 2026-11-13)

The usable recordings (`data_sessions.rsvp_usable`: E21, E23, E24, E25, E27, E28, E29, E31; 5 people, 13,018 kept trials)
show only 262 distinct images: two fixed sets of 144 with 26 in common. `scripts/run_rsvp_pooled.py` whitens each
recording, averages the recordings of every image and scores one decoder over the 262 images (design fixed before the
first run). The averaged recordings come from `reconstruct.rsvp_pooled` and are cached in `cache/rsvp_pooled_<filter>.pt`;
the cache is rebuilt when the usable sessions or the timing settings in `configs/sessions.yaml` change, and with
`--recompute`, which any other change to the loader needs. coarse8 over all images, without -> with the 0.1-30 Hz
band-pass on the continuous data
(`load_rsvp(..., bandpass=(0.1, 30))`, off by default):

| Channels | Own decoder | THINGS zero-shot | THINGS prior |
|---|---|---|---|
| 31 | +0.048 -> +0.051 | +0.031 -> +0.051 | +0.042 -> +0.067 |
| 10 posterior (`channels.POSTERIOR_10`) | +0.065 -> +0.057 | +0.039 -> +0.068 | +0.070 -> +0.094 |

All twelve are above chance (p .0005 to .024). Most do not differ reliably from each other (standard error of a
difference about 0.015). The THINGS prior on the posterior channels is the highest and is reliably above the posterior
THINGS decoder without the filter (+0.031, 95% interval +0.013 to +0.050) and above the posterior own decoder with it
(+0.037, +0.009 to +0.063); it is the best of twelve, picked after the fact, so it needs confirming on new images.
Within the two image sets (144 candidates each, as in the per-recording scores) the filtered THINGS prior on 31 channels
reaches +0.066 and +0.069. For comparison, the per-recording means are +0.026 (own decoder), +0.015 (THINGS zero-shot)
and +0.019 (THINGS prior). Averaging the recordings is the main gain; simply stacking them into one decoder gives none
(+0.020, one-off test with seven recordings). The band-pass alone, per recording, raises the THINGS readouts (zero-shot
+0.013 -> +0.027 on five recordings) and lowers the own decoder (+0.039 -> +0.029). New recordings add repeats, not
images; the learning curves below show that the decoders gain from both.

**Adding E31** (a fourth recording for the 144 images of E23, E28 and E29) raised all twelve readouts and the fine-tuned
network. Per image, against the seven-recording results (git history):

| Readout, with the band-pass | 7 recordings | 8 recordings | Gain (95% interval) |
|---|---|---|---|
| THINGS prior, 10 posterior | +0.062 | +0.094 | +0.031 (+0.014 to +0.049) |
| Own decoder, 10 posterior | +0.040 | +0.057 | +0.018 (+0.000 to +0.035) |
| THINGS prior, 31 channels | +0.061 | +0.067 | +0.007 (-0.005 to +0.019) |
| Own decoder, 31 channels | +0.049 | +0.051 | +0.002 (-0.013 to +0.017) |
| Fine-tuned network, 31 channels | +0.079 | +0.086 | +0.006 (-0.020 to +0.033) |

Only the posterior THINGS prior gained reliably (without the filter +0.050 -> +0.070, +0.020, +0.006 to +0.033). Most of
the gain is on the 144 images that got the fourth recording (+0.043 for the posterior prior); the images whose data did
not change gained a little through the better-trained decoder (+0.018).

## The THINGS-pretrained network on the pooled recordings (2026-11-12, with E31 since 2026-11-13)

`scripts/run_rsvp_finetune.py` puts the pixel network of R3 / R4 (`starstim_recon/things_network.py`, checkpoint
`data/things_pixel_network/best.pt`) on the same averaged recordings (31 channels) and the same 8 image folds. Variants
and regime were fixed before the first run (MSE, AdamW, 40 epochs, batch 32, last epoch, learning rates 1e-3 adapter,
1e-4 temporal layer, 1e-5 all other layers); the five seeds and the split by pretraining set were added after it.
coarse8, without -> with the 0.1-30 Hz band-pass:

| Readout | All 262 images | 173 images the network was pretrained on | 80 images it never saw |
|---|---|---|---|
| Own decoder (reference) | +0.048 -> +0.051 | +0.054 -> +0.059 | +0.045 -> +0.048 |
| THINGS prior, linear (reference) | +0.042 -> +0.067 | +0.042 -> +0.073 | +0.041 -> +0.069 |
| Network as it is | +0.017 -> +0.029 | +0.022 -> +0.038 | +0.010 -> +0.015 |
| Ridge on its hidden features | +0.005 -> -0.012 | -0.008 -> -0.024 | +0.045 -> +0.027 |
| Ridge on its temporal features | +0.067 -> +0.086 | +0.089 -> +0.085 | +0.037 -> +0.101 |
| Fine-tuned: adapter slot | +0.017 -> +0.044 | +0.016 -> +0.041 | +0.019 -> +0.044 |
| Fine-tuned: + temporal layer | +0.063 -> +0.066 | +0.053 -> +0.075 | +0.092 -> +0.056 |
| Fine-tuned: + all other layers, 5 seeds averaged | +0.064 -> +0.086 | +0.057 -> +0.099 | +0.090 -> +0.066 |
| Fine-tuned: all layers, one row per recording x image | +0.059 -> +0.065 | +0.058 -> +0.074 | +0.074 -> +0.060 |

Fine-tuning all layers gives the highest score on the pooled recordings on 31 channels (+0.086 with the filter,
p .0005; single seeds +0.083 to +0.087, without the filter +0.062 to +0.066). It is not reliably better than the own
decoder (+0.035, 95% interval -0.011 to +0.081) or the linear THINGS prior (+0.018, -0.024 to +0.060). With seven
recordings the ensemble scored +0.056 -> +0.079.

**Leakage.** The network was pretrained on 15,626 THINGS-EEG2 training images with their pixel targets; only the 92
old-lab images were left out. 173 of the 262 rapid-stream images were therefore in its training set and 9 in its
validation split, so it had already been trained to reconstruct them from other people's EEG. The linear THINGS readouts do
not have this, they are fitted without any image shown in the lab. On the 80 images the network never saw (candidates =
these 80), the fine-tuned network scores +0.090 (p .006) without and +0.066 (p .048) with the filter. With the filter
that is level with the linear prior (+0.069) on the same images, and its gain over the own decoder is larger on the
pretraining images (+0.040 there, +0.018 on the never-seen ones); without the filter it is the other way round (+0.003
and +0.045). On 80 images these differences have 95% intervals of about +-0.08, so the split neither
shows nor excludes an inflated score. The never-seen column is the one free of leakage. A clean score on all images needs
the network pretrained again without the 262 lab images; that has not been done.

## What more data would buy: learning curves (2026-11-12, with E31 since 2026-11-13)

`scripts/run_rsvp_learning_curves.py` trains the pooled own decoder and the linear THINGS prior on less data and scores
them on the same test images at their full average (same folds, 31 channels; design fixed before the first run). The
training images of every fold are thinned at random (20 draws), and every training image is averaged over fewer of its
recordings (one recording is about 12 kept trials per image, all 4.40 are about 51). coarse8 with the 0.1-30 Hz band-pass:

| Images in training (all recordings) | 25 | 50 | 100 | 150 | 200 | all (229) |
|---|---|---|---|---|---|---|
| Own decoder | -0.008 | +0.001 | +0.022 | +0.041 | +0.046 | +0.051 |
| Own decoder, EEG shuffled | -0.012 | -0.018 | -0.020 | -0.019 | -0.014 | -0.012 |
| THINGS prior | +0.040 | +0.043 | +0.045 | +0.056 | +0.061 | +0.067 |
| THINGS prior, EEG shuffled | -0.006 | -0.005 | -0.006 | -0.006 | -0.004 | -0.006 |

| Recordings per training image (all images) | 1 | 2 | 3 | all (4.40) |
|---|---|---|---|---|
| Own decoder | +0.017 | +0.038 | +0.045 | +0.051 |
| THINGS prior | +0.045 | +0.054 | +0.058 | +0.067 |

Both decoders gain from more images and from more recordings. From 100 to all images the own decoder gains +0.029 (95%
interval +0.013 to +0.045) and the THINGS prior +0.022 (+0.007 to +0.037); from one to all recordings per training image
+0.035 (+0.015 to +0.053) and +0.023 (+0.006 to +0.039). The noise in the test image counts as much: scored from one
recording instead of all, a test image loses 0.024 (own decoder) and 0.038 (THINGS prior). At the same number of image x
recording rows in training, 200 images with one recording against 100 images with two, the THINGS prior is level
(0.000, -0.005 to +0.005) and the own decoder is +0.011 higher with the 200 images (+0.002 to +0.021), of which about
0.006 is the difference between the two nulls. So the pooled decoders are limited by the amount of data, by distinct
images and by trials per image about equally. The THINGS prior is still rising at the last step and passes the THINGS
decoder alone (+0.051, no lab data) from 150 training images on; the own decoder slows above 150 images (+0.041, +0.046,
+0.051, while its null rises by 0.007), so whether it would keep gaining beyond 262 images is open. Without the filter
the picture is the same in outline
(`results/rsvp_learning_curves.json`).

**The null is not 0.** With the EEG shuffled against the images (40 shuffles per point) the cross-validated own decoder
scores -0.012 to -0.023 and the THINGS prior about -0.005: every out-of-fold prediction contains the mean target of its
training fold, which is anti-correlated with the held-out images. coarse8 of a decoder fitted in image folds is therefore
slightly too low, the own decoder's more than the THINGS prior's, and the permutation p-values (which shuffle labels
after the fit) are, if anything, conservative for it. The zero-shot THINGS readouts are not affected. The reported
scores are left as they are.

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

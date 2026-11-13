# Handover: what was tried, what came out, where to continue

State of 2026-11-13. For coworkers and their coding agents: read this before changing anything. It is the map; `README.md`
has the setup, every command and the full tables, `data/README.md` the data and where they come from.

## The project in five lines

- **Goal:** reconstruct the coarse 8x8 colour layout of a viewed image from StarStim 32 EEG (no detail, no semantics).
- **Score:** coarse8 = correlation of the reconstructed layout with the true one, relative to the other candidate images
  (0 = chance, 1 = perfect). One definition, `starstim_recon/metrics.py`.
- **Benchmark:** THINGS-EEG2 reduced to our 31 channels reaches +0.44 (10 people, 80 repeats per test image).
- **Best on our data:** +0.067 with a linear readout and +0.086 with a fine-tuned network (see the leakage caveat below),
  on eight recordings averaged per image and all 31 channels. That is 15-20% of the benchmark. The linear THINGS prior on
  the 10 posterior channels reaches +0.094, the best of twelve readouts picked after the fact. A single recording reaches
  about +0.03.
- **What limits it:** the amount of data, distinct images and trials per image about equally. The decoders tried so far
  cannot be told apart on the 262 images we have.

## The data

| Data | What it is | Used for |
|---|---|---|
| THINGS-EEG2, lab contract | 10 people, 16,540 training and 200 test images, rebuilt with our 31 channels and preprocessing | benchmark, THINGS readouts, pretraining |
| Old lab (May-June 2026) | 6 people, 128 images shown slowly, about 7 trials per image | R1-R7 |
| Sessions E06-E20 | memorability_stim sessions: slow image trials during a stimulation protocol, pulse-free blocks only, 2-4 trials per image | R5 |
| Rapid-stream blocks E21, E23-E29, E31 | 144 images at 5 per second, no stimulation, 5-20 repeats; 9 recordings from 5 people | everything since 2026-11-09 |

Eight rapid-stream recordings are usable (E26 is not: its timing cannot be verified and channels are flat). They show only
two fixed image sets of 144 with 26 in common, 262 distinct images in total. `data_sessions.rsvp_usable` is the one
definition of "usable". E32 (2026-11-13) is not registered: it has no rapid-stream block, only slow image trials during
stimulation (96 of 128 images shown once) and OASIS images.

## What worked

| Idea | Result | Where |
|---|---|---|
| Rapid-stream block instead of slow trials | the first lab data that decode per recording: own decoder above chance in 4 of 8 recordings (mean +0.026) | `scripts/run_rsvp_reconstruction.py` |
| Correcting the EEG clock | in 11 of 23 measurable sessions the EEG time stamps drift against the markers by up to 300 ms (one session by seconds). Measured from the stimulation pulses and corrected automatically. Only with it does a THINGS-trained decoder work on our data at all (E23 -0.03 -> +0.05) | `data_sessions.clock_error`, `scripts/check_rsvp_timing.py` |
| Averaging the recordings of every image | the main gain: the own decoder goes from +0.026 per recording to +0.048 to +0.065 on the eight recordings together. The old lab data showed the same (single trial +0.007, averaged over 6 people +0.079) | `scripts/run_rsvp_pooled.py` |
| THINGS ridge as a prior for the lab ridge (the lab ridge shrunk toward the THINGS decoder) | a safe default and the first sign of a gain from lab data, not yet proven. On all 31 channels +0.067 with the band-pass, not reliably above the THINGS decoder alone (+0.051; difference +0.017, 95% interval -0.005 to +0.039). On the 10 posterior channels +0.094, and without the filter reliably above THINGS alone (+0.070 against +0.039, difference +0.031, +0.013 to +0.050), but that setting was picked after the fact. On single recordings it tracks the THINGS decoder (mean +0.019 against +0.015) and rescues the one where THINGS fails (E27, +0.039 [+0.012, +0.065]). With few training images it is below THINGS alone and passes it from about 150 images (learning curves) | `scripts/run_rsvp_pooled.py`, `scripts/run_rsvp_learning_curves.py` |
| 0.1-30 Hz band-pass on the continuous data | raises every THINGS-based readout (pooled zero-shot +0.031 -> +0.051). Mixed for the own decoder: lower per recording (+0.039 -> +0.029, five recordings), pooled higher on 31 channels (+0.048 -> +0.051) and lower on the posterior ones (+0.065 -> +0.057). Off by default | `load_rsvp(..., bandpass=(0.1, 30))` |
| Fine-tuning all layers of the THINGS-pretrained pixel network on the pooled recordings | +0.086, the highest score on 31 channels, but not reliably above the linear readouts and not free of leakage | `scripts/run_rsvp_finetune.py` |
| Another recording of an existing image set (E31, 2026-11-13) | every pooled readout rose; reliably only the THINGS prior on the posterior channels (+0.062 -> +0.094 with the band-pass, gain +0.031, +0.014 to +0.049), mostly on the 144 images that got the fourth recording. On 31 channels +0.002 to +0.007, not reliable | `scripts/run_rsvp_pooled.py`; README, pooled section |

## What did not work

| Idea | Result | Where |
|---|---|---|
| THINGS-pretrained pixel network zero-shot on the old lab (R3) | +0.020 (p .31) | README table; code in `Hierarchical_EEG2Image_Reconstruction` |
| Its frozen features, or fine-tuned adapters, on the old lab (R4) | all below the plain lab ridge | same |
| Old-lab ridge applied to the sessions E06-E20 (R5) | -0.011 (p .80), with 2-4 slow trials per image | `scripts/run_transfer_sessions.py` |
| THINGS ridge as a prior on the old lab (R6) | +0.032 against +0.079 for the plain ridge | `scripts/run_things_prior.py` |
| Global image properties instead of the layout (R7) | weak within the old lab (mean r +0.066), no transfer | `scripts/run_global_props.py` |
| Old-lab ridge on a rapid-stream block | above chance in none of the eight recordings (-0.022 to +0.020, mean +0.007) | `scripts/run_rsvp_reconstruction.py` |
| THINGS decoder on a single recording, without lab training | above chance in only 3 of 8 recordings (mean +0.015); only E24 decodes both ways | same |
| Stacking the recordings into one decoder instead of averaging them | +0.020, no gain (one-off test) | README, pooled section |
| Only the 10 posterior channels, for the own decoder | no reliable difference from 31 channels (+0.006 to +0.017); for the THINGS prior see the E31 row above | `scripts/run_rsvp_pooled.py` |
| The network as it is on the pooled recordings; a ridge on its hidden features; fine-tuning only the adapter | +0.029, -0.012, +0.044 (all with the band-pass) | `scripts/run_rsvp_finetune.py` |
| Fine-tuning on one row per recording x image instead of the averages | +0.065, below the fine-tuning on the averages (+0.086) | same |
| Shorter time windows, 50 Hz pooling, training on single trials, low-dimensional targets, a spatial alignment to THINGS, a time shift of the THINGS decoder | none reliably beat the plain ridge on E21. Exploratory: one session, and run before the clock correction, so with time zero at the marker | `scripts/explore_rsvp_improvements.py` |

## What you must know before trusting or comparing numbers

1. **The pooled readouts can hardly be told apart.** They lie between +0.03 and +0.09 with 95% intervals of about
   +-0.03 to +-0.05 on their differences. Only the THINGS prior on the posterior channels is reliably ahead of some of
   the others, and it was picked after the fact. Another decoder variant on these 262 images will not settle anything.
2. **The fine-tuned network's +0.086 is not free of leakage.** It was pretrained on 173 of our 262 images with their pixel
   targets. On the 80 images it never saw it scores +0.066 with the band-pass, level with the linear prior (+0.069) there.
   The split neither shows nor excludes an inflated score. The linear THINGS readouts are clean: they are fitted without
   any image shown in the lab (`exclude_basenames`). A clean network needs retraining without the lab images; not done.
3. **A cross-validated decoder without signal scores below 0, not 0.** With the EEG shuffled against the images the own
   decoder scores -0.012 to -0.023, the THINGS prior about -0.005. The reported scores of the decoders fitted in image
   folds are therefore slightly too low and their p-values, if anything, conservative. Decided for now: the scores stay as they are
   (README, learning-curve section).
4. **The timing is inferred, not measured.** Clock error from the pulses plus a display lag of 46 ms estimated from the
   images; no photodiode yet. Scores move by 0.01-0.02 when the offset changes by a few ms. The cause of the drift is
   unknown.
5. **E21's +0.081 at marker timing is not a better result.** Most likely the baseline window then contains the image response.
   With corrected timing E21 scores +0.042.
6. **E28 and E29 do not decode with their own decoder**, although E28 has 20 repeats and a strong image-specific signal
   (split-half reliability .66). Unexplained.
7. **Per-recording and pooled scores use different candidate sets** (144 images of one recording, 262 of all). The pooled
   script also reports the scores within each image set of 144.
8. **Which runs were planned:** the scripts' docstrings say what was fixed before the first run and what was added after
   it. `explore_rsvp_improvements.py` and the stacking test are screens, not results.

## Where to continue

Suggestions in the order I would take them; none of them is decided.

1. **Record more, and differently.** The learning curves (`scripts/run_rsvp_learning_curves.py`) show that the pooled
   decoders still gain from more images and from more recordings per image, and that a test image scored from one
   recording instead of all of its recordings (4.40 on average) loses 0.02 to 0.04. So: a new image set in every session
   instead of the two fixed sets, images that are not THINGS-EEG2 training images (then nothing can leak and the clean
   test set grows), 20 repeats rather than 5, and the same people across sessions. The curves cannot say what happens
   beyond 262 images; the own decoder's curve slows above 150. On the new images, test THINGS as prior against the
   THINGS decoder alone (prior minus THINGS, paired bootstrap over images), with the test fixed before the run: the
   curves suggest the prior's lead grows with more lab data, but on the present 262 images it is reliable only on the
   posterior channels without the filter, a setting picked after the fact. E31 showed that another recording of a known
   image set still helps; new image sets are what the curves cannot judge yet.
2. **Confirm the timing with a photodiode**, once. It would replace the estimated display lag.
3. **Retrain the pixel network without the lab images**, but only when there are enough new test images to see a
   difference; on the present 262 it would change no conclusion.
4. **Decide about the score floor** (point 3 above) before the next round of comparisons between cross-validated and
   zero-shot readouts.

## Working in this repository

**Start.** `README.md`, section Setup: link the data with `scripts/link_data.py`, then `--check`. `results/` is in git
(index: `results/README.md`), so the result files and figures of the last runs come with a clone; `cache/` is not. The
numbers in the README come from the runs of 2026-11-09 to 2026-11-13 on the lab workstation. After a rerun, commit the
changed result files with the code that produced them. To rebuild everything:

```bash
python scripts/make_infographic.py                                        # the three reference numbers
for s in E21 E23 E24 E25 E26 E27 E28 E29 E31; do                          # every rapid-stream session of configs/sessions.yaml
  python scripts/check_rsvp_timing.py --session $s
  python scripts/run_rsvp_reconstruction.py --session $s
  python scripts/run_rsvp_reconstruction.py --session $s --offset-ms 0
done
python scripts/run_rsvp_pooled.py
python scripts/run_rsvp_finetune.py                                       # needs data/things_pixel_network/best.pt and a GPU
python scripts/run_rsvp_learning_curves.py
python scripts/make_results_infographic.py                                # results/infographic_results.png
python scripts/make_reconstruction_sheets.py                              # results/sheets/*.png
```

The results infographic needs the output of every line above, including the fine-tuning, so it cannot be drawn without
the network checkpoint even though the data check treats that file as optional.

**A new recording.** README, section "Add new data"; then rerun the five last lines above. The pooled input is cached in
`cache/`; it is rebuilt when the session list or the timing settings change, anything else needs `--recompute`.

**Rules the code relies on.**

- Privacy: the session folders hold questionnaire answers. Read only the whitelisted image columns
  (`data_sessions.py`, module docstring). Sessions are named by E-number everywhere; no participant codes in outputs,
  figures or commits.
- One definition each, reused everywhere: the target (`images.pixel32`), the score (`metrics.coarse8`), a usable
  recording (`data_sessions.rsvp_usable`), the pooled input (`reconstruct.rsvp_pooled`). Do not rebuild them elsewhere.
- Index EEG channels by name (`channels.py`); THINGS, the hardware and the loaders use different orders.
- Any readout fitted on THINGS must leave out the images shown in the lab (`data_things.train_indices`).
- All data are read through `data/`; no absolute path in the code.
- Fix the design of an analysis before its first run and write into the docstring what was added afterwards. Mark
  exploratory work as such.
- The infographic draws only from result files and stops with the command to run when one is missing or stale.

**Related repositories.** `Hierarchical_EEG2Image_Reconstruction` (branch `dev/lab_lowlevel_transfer`): the THINGS
lab-contract caches, the pixel network and its training, R3 and R4. `memorability_stim`: the session recordings and the
clock-error method this repository copies (`memstim/lab_sessions.py`).

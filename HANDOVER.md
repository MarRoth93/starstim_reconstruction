# Handover: what was tried, what came out, where to continue

State of 2026-11-13. For coworkers and their coding agents: read this before changing anything. It is the map; `README.md`
has the setup, every command and the full tables, `data/README.md` the data and where they come from.

## The project in five lines

- **Goal:** reconstruct the coarse 8x8 colour layout of a viewed image from StarStim 32 EEG (no detail, no semantics).
- **Score:** coarse8 = correlation of the reconstructed layout with the true one, relative to the other candidate images
  (0 = chance, 1 = perfect). One definition, `starstim_recon/metrics.py`.
- **Benchmark:** THINGS-EEG2 reduced to our 31 channels reaches +0.44 (10 people, 80 repeats per test image).
- **Best on our data:** +0.06 with a linear readout and +0.08 with a fine-tuned network (see the leakage caveat below), on
  seven recordings averaged per image. That is 14-18% of the benchmark. A single recording reaches about +0.02.
- **What limits it:** the amount of data, distinct images and trials per image about equally. The decoders tried so far
  cannot be told apart on the 262 images we have.

## The data

| Data | What it is | Used for |
|---|---|---|
| THINGS-EEG2, lab contract | 10 people, 16,540 training and 200 test images, rebuilt with our 31 channels and preprocessing | benchmark, THINGS readouts, pretraining |
| Old lab (May-June 2026) | 6 people, 128 images shown slowly, about 7 trials per image | R1-R7 |
| Sessions E06-E20 | memorability_stim sessions: slow image trials during a stimulation protocol, pulse-free blocks only, 2-4 trials per image | R5 |
| Rapid-stream blocks E21, E23-E29 | 144 images at 5 per second, no stimulation, 5-20 repeats; 8 recordings from 5 people | everything since 2026-11-09 |

Seven rapid-stream recordings are usable (E26 is not: its timing cannot be verified and channels are flat). They show only
two fixed image sets of 144 with 26 in common, 262 distinct images in total. `data_sessions.rsvp_usable` is the one
definition of "usable".

## What worked

| Idea | Result | Where |
|---|---|---|
| Rapid-stream block instead of slow trials | the first lab data that decode per recording: own decoder above chance in 3 of 7 recordings (mean +0.022) | `scripts/run_rsvp_reconstruction.py` |
| Correcting the EEG clock | in 10 of 22 sessions the EEG time stamps drift against the markers by up to 300 ms (one session by seconds). Measured from the stimulation pulses and corrected automatically. Only with it does a THINGS-trained decoder work on our data at all (E23 -0.03 -> +0.05) | `data_sessions.clock_error`, `scripts/check_rsvp_timing.py` |
| Averaging the recordings of every image | the main gain: +0.022 per recording -> +0.041 to +0.061 on the seven recordings together. The old lab data showed the same (single trial +0.007, averaged over 6 people +0.079) | `scripts/run_rsvp_pooled.py` |
| THINGS ridge as a prior for the lab ridge (the lab ridge shrunk toward the THINGS decoder) | a safe default, not yet a proven gain. On the pooled recordings +0.061 with the band-pass, the highest leak-free number, but not reliably above the THINGS decoder alone (+0.047; difference +0.014, 95% interval -0.007 to +0.035). On single recordings it tracks the THINGS decoder (mean +0.019 against +0.016) and rescues the one where THINGS fails (E27, +0.039 [+0.012, +0.065]). With few training images it is below THINGS alone and passes it from about 150 images (learning curves). The THINGS decoder alone, with no lab training at all, is about as good as anything trained on our data | `scripts/run_rsvp_pooled.py`, `scripts/run_rsvp_learning_curves.py` |
| 0.1-30 Hz band-pass on the continuous data | raises every THINGS-based readout (pooled zero-shot +0.029 -> +0.047). Mixed for the own decoder: lower per recording (+0.039 -> +0.029), higher pooled (+0.041 -> +0.049). Off by default | `load_rsvp(..., bandpass=(0.1, 30))` |
| Fine-tuning all layers of the THINGS-pretrained pixel network on the pooled recordings | +0.079, the highest score, but not reliably above the linear readouts and not free of leakage | `scripts/run_rsvp_finetune.py` |

## What did not work

| Idea | Result | Where |
|---|---|---|
| THINGS-pretrained pixel network zero-shot on the old lab (R3) | +0.020 (p .31) | README table; code in `Hierarchical_EEG2Image_Reconstruction` |
| Its frozen features, or fine-tuned adapters, on the old lab (R4) | all below the plain lab ridge | same |
| Old-lab ridge applied to the sessions E06-E20 (R5) | -0.011 (p .80), with 2-4 slow trials per image | `scripts/run_transfer_sessions.py` |
| THINGS ridge as a prior on the old lab (R6) | +0.032 against +0.079 for the plain ridge | `scripts/run_things_prior.py` |
| Global image properties instead of the layout (R7) | weak within the old lab (mean r +0.066), no transfer | `scripts/run_global_props.py` |
| Old-lab ridge on a rapid-stream block | above chance in none of the seven recordings (-0.022 to +0.020, mean +0.005) | `scripts/run_rsvp_reconstruction.py` |
| THINGS decoder on a single recording, without lab training | above chance in only 3 of 7 recordings (mean +0.016); only E24 decodes both ways | same |
| Stacking the recordings into one decoder instead of averaging them | +0.020, no gain (one-off test) | README, pooled section |
| Only the 10 posterior channels | no reliable difference from 31 channels | `scripts/run_rsvp_pooled.py` |
| The network as it is on the pooled recordings; a ridge on its hidden features; fine-tuning only the adapter | +0.028, -0.017, +0.030 (all with the band-pass) | `scripts/run_rsvp_finetune.py` |
| Fine-tuning on one row per recording x image instead of the averages | +0.049, no better than the own decoder | same |
| Shorter time windows, 50 Hz pooling, training on single trials, low-dimensional targets, a spatial alignment to THINGS, a time shift of the THINGS decoder | none reliably beat the plain ridge on E21. Exploratory: one session, and run before the clock correction, so with time zero at the marker | `scripts/explore_rsvp_improvements.py` |

## What you must know before trusting or comparing numbers

1. **The pooled readouts cannot be told apart.** All of them lie between +0.04 and +0.08 with 95% intervals of about
   +-0.045 on their differences. Another decoder variant on these 262 images will not settle anything.
2. **The fine-tuned network's +0.079 is not free of leakage.** It was pretrained on 173 of our 262 images with their pixel
   targets. On the 80 images it never saw it scores +0.061 with the band-pass, level with the own decoder (+0.057) there.
   The split neither shows nor excludes an inflated score. The linear THINGS readouts are clean: they are fitted without
   any image shown in the lab (`exclude_basenames`). A clean network needs retraining without the lab images; not done.
3. **A cross-validated decoder without signal scores below 0, not 0.** With the EEG shuffled against the images the own
   decoder scores -0.013 to -0.022, the THINGS prior about -0.005. The reported scores of the decoders fitted in image
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
   recording instead of all of its recordings (3.85 on average) loses 0.03. So: a new image set in every session instead
   of the two fixed sets, images that are not THINGS-EEG2 training images (then nothing can leak and the clean test set
   grows), 20 repeats rather than 5, and the same people across sessions. The curves cannot say what happens beyond 262
   images; the own decoder's curve slows above 150. On the new images, test THINGS as prior against the THINGS decoder
   alone (prior minus THINGS, paired bootstrap over images), with the test fixed before the run: the curves suggest the
   prior's lead grows with more lab images, but on the present 262 it is not reliable.
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
for s in E21 E23 E24 E25 E26 E27 E28 E29; do                              # every rapid-stream session of configs/sessions.yaml
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

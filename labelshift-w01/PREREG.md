# Pre-registration A: labels as shipped against corrected labels on Paris4 w01

Written on 2026-11-04 before any training run of A, and fixed before the first GPU run. A change made after
the first GPU run of A starts makes A exploratory; any such change goes into the report with its reason.

## 1. Question

Does training villa's `full_3d` ink trainer on Paris4 w01 with the labels as shipped (AS) give a lower
held-out ink ROC AUC than training on the same labels moved into the tifxyz frame with the HF65 w01
correction (C0)?

Why w01: in the October study on w00 (`../labelshift/README.md`), AS was not measurably worse than C0, and a
cost was detected only for a synthetic 128 px shift (C32y). The HF65 offset on w01 is much larger than on
w00. Over w01's supervised area it runs from +49.3 to +257.3 px in y and from -158.0 to -42.0 px in x, up to
288 px in length (`logs/check_arms_w01.log`). On w00 the October PREREG gives -1.6 to +17.4 px in y and -10.0
to +14.5 px in x.

## 2. Data and held-out boxes

| segment | CT volume (public S3) | labels |
|---|---|---|
| `ink/phercparis4/w01_20230702185753_r15` | `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` | `w01_20230702185753_r15_{inklabels,supervision_mask}` |

Inputs from the public bucket: `meta.json`, `x.tif`, `y.tif`, `z.tif` (the tifxyz) and the two label TIFFs,
about 500 MB in all. Path, bytes and sha256 of each are in `logs/fetch_manifest_w01.tsv`, and
`pod_run.sh prepare` checks every download against `logs/raw_sha256.txt`.

The label TIFFs are what the trainer reads. Every chunk of levels 2 to 5 of both label zarrs in the bucket,
and every chunk of inklabels levels 0 and 1, equals the TIFF sampled at that level's stride, and the 64
slices other than slice 32 are empty (`logs/check_tif_vs_zarr_w01.log`). Supervision mask levels 0 and 1 were
not downloaded, so they were not compared.

Frames: canvas (y, x) is a pixel of the shipped label image (65419 x 56889). A tifxyz cell is a
full-resolution cell of the segment's tifxyz (65698 x 56669), which is what `full_3d` reads labels
against. The HF65 w01 correction (`scripts/hf65/apply_fit.py`, model "readme", copied from
stantheman0128/vesuvius-hf65-alignment): canvas content at (y, x) belongs at cell (y + dy, x + dx), with
dy = 11.51 + 0.004263 * y and dx = 11.82 - 0.003876 * x. Its check is `../correct/RESULTS.md`: on nine w01
regions, six used by the fit and three not, the residual after correction is +0.10 to +0.15 px on both
axes, and the NCC between the HF65 image and the CT peaks at zero shift (0.894 to 0.930). Of the three regions not used by the
fit, one lies outside the supervision mask and two lie inside it.

Held-out boxes. October's rule for w00 was "three labelled components holding roughly 10 to 20% of the
ink, mapped to tifxyz, padded 64 px", and it did not say which three. For A, `scripts/pick_heldout.py`
fixes the choice. It was written and its sha256 recorded (`logs/heldout_rule_timestamp.txt`, 17:38:47
+0800) before any w01 component list existed:

- A component is a connected area of level-4 supervision after a 48 px dilation
  (`scripts/list_components.py`, as in October).
- Eligible: the box mapped into the tifxyz frame and padded by 64 px lies inside the tifxyz grid; the
  canvas box padded by 64 px (where AS and the shipped labels put the component) lies inside the guard
  box defined below; and the component holds at least 2% of the segment's ink.
- Candidates: three eligible components whose mapped, padded boxes do not overlap and whose ink shares sum
  to 10% to 20%.
- Choice: the candidate that removes the least other supervision from training. Ties go to the ink share
  closest to 15%, then to the smallest ids.

Result (`logs/pick_heldout_w01.log`): 26 components, 23 eligible, 1232 candidates. The components lie far
apart, so many candidates remove no other supervision and the choice fell to the tie-breaks. Held out:
components 5, 17 and 19, 15.0% of w01's ink.

| id | canvas box [y0, y1, x0, x1] | ink share | offset (dy, dx) at its centre |
|---|---|---|---|
| 5 | 13296, 15616, 15712, 17440 | 3.0% | +73.1, -52.4 |
| 17 | 35856, 38512, 29456, 33168 | 5.9% | +170.0, -109.5 |
| 19 | 38448, 41616, 34128, 37600 | 6.1% | +182.2, -127.2 |

`regions.json` holds these boxes. It was saved before any label build (`logs/regions_timestamp.txt`,
17:54:25 +0800, with its sha256). Each box is mapped into the tifxyz frame with the correction and padded
by 64 px; the union of the three is HELD. HELD padded by 512 px is GUARD. No arm has training supervision
inside GUARD.

The centres of two fit regions used by the correction lie inside held-out components: r4 (14336, 16384)
in component 5 and r5 (40960, 36352) in component 19. The fit registered the HF65 image against the CT
(`../correct/regcorr.py`) and used no ink labels or ink predictions, so the held-out ink stays unseen by
training. It does mean the C0 reference labels in those two components sit where the correction was
measured.

## 3. Arms

| arm | labels and supervision in the training area |
|---|---|
| C0 | canvas labels resampled into the tifxyz frame with the correction (nearest neighbour) |
| AS | as shipped: tifxyz cell (y, x) carries canvas pixel (y, x) |

In both arms the ink labels inside HELD are C0's, the validation mask is C0 supervision inside HELD, and
the supervision mask is the arm's own supervision outside GUARD. The trainer's own validation therefore
scores both arms against C0. The AS copy of each held-out component lies inside GUARD (eligibility above),
so neither arm trains on a held-out component.

Built by `scripts/build_labels.py --arms AS,C0`; level-0 nonzero pixel counts (`labels/manifest_full.json`):

| arm | inklabels | supervision_mask | validation_mask |
|---|---|---|---|
| AS | 34,373,332 | 114,335,744 | 18,030,085 |
| C0 | 34,356,961 | 114,376,420 | 18,030,085 |

How far apart the arms are in the training area (outside GUARD, level 0): 124,991,940 cells are supervised
in AS or C0, the supervision differs at 17.0% of them, and where both arms are supervised the ink label
differs at 23,125,309 cells, 22.3% (`logs/check_arms_w01.log`). The trainer finds 1350 training patches
for C0 and 1374 for AS, and 334 validation patches for both (`logs/dryrun_counts.log`).

Two checks that AS and C0 differ only by the correction:

- `scripts/check_arms_w01.py` was written apart from the builder. It recomputes both arms from the label
  TIFFs on every 4096 px tile of the tifxyz grid, using the closed-form inverse of the fit where the
  builder uses a fixed-point inverse and separable index maps, and compares all three arrays of both arms
  pixel for pixel, every level against level 0, and the counts against the manifest. Every comparison
  differs at 0 px, every count matches, and every arm, array and level is nonzero
  (`logs/check_arms_w01.log`, ALL_OK).
- `scripts/xcorr_arms_w01.py` measures, for each of the 23 training components, the shift that best aligns
  the C0 ink with the AS ink. It matches the fit's offset at the component centre within 3.9 px for all 23,
  and at that shift 98.5% to 99.5% of the AS ink overlaps C0 ink (`logs/xcorr_arms_w01.log`). The fit's
  scale terms change the offset by up to about 10 px between the centre and the edge of the largest box
  (component 2, 4384 x 5440 px), which a single shift cannot follow.

## 4. Training configuration

`scripts/mkconfig.py gpu` writes each run's config from villa's
`vesuvius/src/vesuvius/ink_detection/configs/dino_guided_teacher.json` at 37d1897fb, the same way it did
for October's Amendment F and G runs (`../labelshift/results/F/runs/C0_s27/config.json`). Kept from those
runs: mode `full_3d`, the teacher's model, loss, optimiser and learning rate, `diffusers_cosine_warmup`, EMA,
bf16, batch 2, patch 128^3, volume_scale 2, `MAXSTEPS_MODE=anneal` (max_steps equal to STEPS), warmup and
EMA start scaled by STEPS / 60001, val_every STEPS / 16, one checkpoint at the last step.

Changed from those runs:

| setting | October F and G | A | why |
|---|---|---|---|
| datasets | w00 and w018, arm label folders | w01 only, arm label folders | decision 1 of the brief |
| dataloader_workers | 16 (pod) | 8 | the GPU machine's measured setting; at its old 15 GB WSL limit 8 workers already hit the OOM killer, and the limit is now 24 GB |

The worker count sets how batches are loaded and can change the order of random draws, so A's runs are
not draw-for-draw comparable with October's. Within A every run uses 8.

STEPS is 12000: Amendment E's L2 setting with the anneal schedule, which the October F and G runs used
(`../labelshift/results/F/f_choice.txt`; their config.json files have num_iterations and max_steps 12000,
warmup_steps 1000, EMA start 200, val_every 750). That keeps A next to the October comparison. On w00 the
L2 pilot also learned more than the 6000-step L1 pilot (AUC 0.7425 against 0.6943, one run each), which
makes the gate in section 5 more likely to pass. `pod_run.sh prepare`, which writes the run configs, and
`desk_launch.sh` stop unless STEPS is 12000, and `desk_launch.sh` checks that every run config carries the same
STEPS and worker count.

Trainer code: villa at 37d1897fb, the October commit. On the GPU machine it is the existing sparse checkout in
`~/labelshift/villa`, used read-only.

## 5. Run order and gate

1. C0 seed 27, then AS seed 27.
2. Gate: C0 seed 27 primary AUC (section 6) at least 0.60. If it is below, A ends with the outcome
   "w01 does not learn in this setup"; seeds 28 to 30 are not run and AS seed 27 is reported without a
   comparison.
3. If the gate passes: C0 and AS for seed 28, then seed 29, then seed 30.

## 6. Evaluation

Each run is scored at its last checkpoint with the EMA weights by `scripts/eval_heldout.py`, on every
held-out patch once, in order, without augmentation, through villa's own InkDataset.

- Primary: w01 held-out voxel ROC AUC against the 3D-projected C0 labels where the validation mask
  projects. The eval config points at the C0 label folder, so both arms are scored on the same
  334 patches. Written to `eval.json`.
- Secondary, reported only: the same AUC against the labels as shipped in the held-out boxes. The eval
  folder EAS (`scripts/build_eval_labels.py`) is C0 except inside the canvas boxes padded by 64 px, where
  cell (y, x) carries the shipped canvas pixel (y, x) for both ink and validation mask. Scored on EAS's
  own 343 held-out patches. Written to `eval_AS.json`.

`scripts/check_eval_labels.py` found E0 byte-identical to C0 (sha256 of all three arrays) and EAS equal to
the shipped labels inside its three boxes, with its validation mask zero outside them
(`logs/check_eval_labels_w01.log`, ALL_OK).

## 7. Decision rule

With four seeds per arm: d = mean(C0) - mean(AS) over the primary AUC. s^2 is the pooled seed variance
(squared deviations from each arm's mean, summed over both arms, over 6 degrees of freedom). The 95%
interval is d +- t * s * sqrt(1/4 + 1/4) with t = 2.447 (Student t, 6 degrees of freedom).

- Lower end above 0: "cost measured", reported with d and the interval.
- Upper end below 0: AS scored above C0. Reported as not explained by this design.
- Otherwise: "no cost measured", and the upper end is reported as the largest cost the data allow.
- A run that failed or is missing: no decision. The interval over the runs present is shown as
  descriptive only.
- Secondary: the same means and difference on the as-shipped AUC, listed without a verdict.

For scale: with October's pooled w00 seed noise (s = 0.0056 after Amendment H, 0.0148 after Amendment I),
the half-width of this interval would be 0.010 to 0.026 AUC.

`scripts/decide_w01.py` applies sections 5 and 7 to the `runs/` folder and writes `results.md`.

## 8. Budget and runtime

A runs on one GPU machine (one RTX 3070 Ti, WSL Ubuntu-24.04) through `desk_launch.sh`, outside the hours
the machine is in use (Monday, Wednesday and Friday, 10:00 to 19:00 +0800). A run starts only if its
estimate ends before the next of those windows. Training ran at 2.6 s per step at the end of a 200-step
smoke test on that machine with a cold cache on 2026-11-04 (October's settings, 8 workers); a warm cache
may be faster. The two evaluations per run are not measured there yet. The estimate is (STEPS x 2.6 s +
40 minutes) plus 10%, and after the first run the script uses the longest measured run plus 5% instead.

| STEPS | per run | start Wednesday 19:00: seed 27 pair done | all 8 runs done |
|---|---|---|---|
| 12000 | 10.3 h | Thursday 16:30 | Sunday 22:20 |

The table includes a one-hour prepare and the waits for those windows. If the gate fails, A stops after the
seed 27 pair.

## 9. Not covered

Other segments and scrolls. Training on w01 together with other segments, as October did with w00 and
w018. Synthetic shifts on w01. Training at level 0, or the teacher's full 60k-step schedule. Whether the
correction holds as well away from the nine checked regions; its common +0.1 px residual was not traced.
Three held-out components of one segment are a small sample of w01's ink, and the AUC difference speaks
for those three areas.

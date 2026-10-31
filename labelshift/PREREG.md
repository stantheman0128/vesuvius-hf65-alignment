# Pre-registration: what label misalignment costs full_3d ink training

Follow-up to ScrollPrize/villa#2370 (HF65 canvas vs tifxyz offset) and #2369. Written before any
training run of this experiment, the CPU smoke tests included. Smoke-test numbers are plumbing checks
and never enter a claim. Changing anything below after the GPU run starts makes the run exploratory;
any such change gets listed in REPORT.md with its reason.

## 1. Question

How much does training on labels that are offset from the CT surface cost held-out ink detection in
villa's `full_3d` trainer, and where do the labels as shipped (canvas frame, what
`dino_guided_teacher.json` trains on today) fall on that cost curve?

## 2. Data

| key | segment | CT volume (public S3, read at level 2) | labels |
|---|---|---|---|
| w00 | `ink/phercparis4/w00_20231016151002` | `PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` | `w00_20231016151002_{inklabels,supervision_mask}` |
| w018 | `ink/1667/w018_20240304144031_2um` | `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr` | `w018_20240304144031_2um_{inklabels,supervision_mask}` (the files the trainer discovers) |

Label pixels are read from the label TIFFs; `logs/check_tif_vs_zarr*.log` shows every chunk of the
trainer's zarrs that was compared equals the TIFF (w00 all levels, w018 levels 2 to 5, level 2 being
the one training reads).

Frames: canvas (y, x) is a pixel of the shipped label image. A tifxyz cell is a full-resolution cell
of the segment's tifxyz, which is what `full_3d` reads labels against. The HF65 correction
(`scripts/hf65/apply_fit.py` from stantheman0128/vesuvius-hf65-alignment a7b7468, model "readme",
the one tested in its `correct/RESULTS.md`): canvas content at (y, x) belongs at cell
(y + dy, x + dx), with dy = -20.70 + 0.000993*y and dx = 11.09 - 0.000663*x on w00, and
dy = dx = -19.64 on w018.

Held-out tiles (`regions.json`, component boxes from `logs/components_*.log`):
- w018: components 5, 7 and 14, the three supervised areas fully inside the shipped
  `w018_20240304144031_validation_mask_v2` (11.8% of w018 ink). The trainer never reads that file
  because its prefix lacks `_2um`, so it is used here only to pick the tiles.
- w00: components 10, 12 and 14 (20.8% of w00 ink). w00 ships no validation mask.
Each box is mapped into the tifxyz frame and padded by 64 px. Supervision within 512 px of a
held-out box is removed from training in every arm.

## 3. Arms

One shift cell = 4 level-0 px = one voxel at the training level (volume_scale 2).

| arm | labels and supervision in the training area |
|---|---|
| C0 | resampled into the tifxyz frame with the correction (nearest neighbour) |
| C2y, C4y, C8y | C0 moved down by 2, 4, 8 cells (8, 16, 32 level-0 px) |
| AS | as shipped: cell (y, x) carries canvas pixel (y, x) |
| C2x, C4x, C8x | optional: C0 moved right by 2, 4, 8 cells; run only if `ARMS` includes them at launch |

In every arm, labels inside the held-out boxes are C0 labels and the validation mask is C0
supervision inside the held-out boxes, so the trainer's own validation also scores against C0.

What AS amounts to: on w018 the shipped labels sit +19.64 px in y and in x from C0 (4.9 cells per
axis, 27.8 px diagonal). On w00 the offset depends on position: -1.6 to +17.4 px in y and -10.0 to
+14.5 px in x over the supervised area.

Unit decision: the brief says "2, 4, 8 tifxyz cells". Read as level-0 cells, the largest shift
(8 px) would be two voxels below the training resolution and smaller than the AS offsets it is meant
to bracket, so the cells are taken at the training level. This is a choice made in prep; the
constant is `shift_cell_level0_px` in `regions.json`.

## 4. Training configuration

`scripts/mkconfig.py gpu` writes each run's config from villa's
`vesuvius/src/vesuvius/ink_detection/configs/dino_guided_teacher.json` at 37d1897fb (origin/main on
2026-10-30). Kept from the teacher: mode `full_3d`, model `unet` autoconfigured, loss
(LabelSmoothedDCAndBCE, smoothing 0.1), SGD with Nesterov momentum 0.99, lr 0.01, weight decay 3e-5,
`diffusers_cosine_warmup`, EMA (decay 0.9995, validated, saved), bf16, batch 2,
`percentile_minmax` normalisation, `projection_half_thickness` 3, patch overlap 0.5, minimum
labelled coverage 0.02, patch finding at scale 4, val_steps 10, 4 prefetch.

| setting | teacher | here | why |
|---|---|---|---|
| datasets | 8 scrolls, historical local labels | w00 + w018 only, arm label folders | the arms only differ on these two |
| volume_scale | 0 | 2 | cost; the tutorial's native-3D recipe also trains at level 2 |
| patch_size | 256^3 | 128^3 | cost, and the supervised components are 1.2k to 6k px across |
| num_iterations | 60001 | STEPS (default 6000) | budget, section 8 |
| max_steps, warmup_steps, ema.start_step | 250000, 5000, 1000 | scaled by STEPS/60001 (25000, 500, 100 at 6000) | keep the teacher's schedule shape |
| val_every, save | 1000, 5000 and 60001 | STEPS/16, final step only | |
| seed | 27 | 27 and 28 | two seeds per arm |
| dataloader_workers | 4 | 8 (pod CPU dependent, env `WORKERS`) | throughput only |
| volume cache | none | on-disk cache, 200 GB cap | each chunk is read from S3 once |

Trainer code: a `git archive` of villa 37d1897fb (`trees/main37d1897`). Its `ink_detection` and
`tifxyz` code equals the #2277 PR base 7154682 apart from `tifxyz/store.py` (added by #2358, not
imported by training).

## 5. Run order

Seed 27 for C0, AS, C8y, C4y, C2y (then C8x, C4x, C2x if enabled), then seed 28 in the same order.
C0 seed 27 is the pilot for gate G0.

## 6. Evaluation

Every run is scored at its final checkpoint by `scripts/eval_heldout.py` with the EMA weights, on
every held-out patch of the C0 dataset (villa's InkDataset, no augmentation, same patches for every
arm), against the 3D-projected C0 labels where the validation mask projects.

- Primary metric: pooled held-out voxel ROC AUC (both segments).
- Secondary, reported but not used by the rules below: per-segment AUC, best F0.5 and its threshold,
  F0.5 and balanced accuracy at 0.5, and the trainer's own `val_balanced_accuracy` (last line of
  `validation_metrics.jsonl`).

## 7. Decision rules

Notation: m(arm) is the mean primary metric over the two seeds; s1, s2 the seed values.

- G0, pilot gate: if C0 seed 27 has pooled AUC below 0.70, stop. Outcome: inconclusive, this
  setting does not learn ink well enough to measure a label effect. No other arm is run.
- G1, completeness: the curve needs C0, C2y, C4y, C8y and AS with both seeds. An arm with a failed or
  missing seed is reported as missing and left out of R1 to R3.
- Noise: sigma = sqrt(mean over complete arms of (s1 - s2)^2 / 2); delta = max(2 * sigma, 0.01).
- R1, does a shift cost anything: COST if m(C0) - m(C8y) > delta. NO COST if
  |m(C0) - m(C8y)| <= delta, stated as "no measurable cost of a 32 px shift in this setting".
  If m(C8y) - m(C0) > delta, report it as unexplained and do not interpret the other arms.
- R2, shape (only under COST): the curve is called monotone if each of m(C2y), m(C4y), m(C8y) is at
  most delta above the arm before it. Reported either way, with all four means.
- R3, shipped labels: d_AS = m(C0) - m(AS). If |d_AS| <= delta: "the cost of the shipped offset is
  not distinguishable from zero here". Otherwise AS is placed on the piecewise-linear curve through
  (0, m(C0)), (8, m(C2y)), (16, m(C4y)), (32, m(C8y)) px: its equivalent y shift is the first
  crossing of m(AS); below m(C8y) it is reported as "more than 32 px equivalent". This is a y-only
  yardstick; the shipped offset is two-axis on w018 and position-dependent on w00, and the report
  says so.
- R4, what may go into a villa comment: the R1 outcome, d_AS with delta and both seed values of every
  arm, and the R3 equivalent shift. Secondary metrics and per-segment numbers may be shown but not
  used to support a claim the primary metric does not make. Optional x arms are reported as a
  separate curve with the same rules.

## 8. Budget and abort

`pod_run.sh smoke` runs 200 steps of C0 seed 27 on the pod and prints s/step and the projected GPU
hours for the chosen arms. `pod_run.sh run` refuses to start when that projection exceeds
`CAP_HOURS` (default 45). Stan approves the GPU run and its cap; nothing here rents anything.

## 9. Not covered

Other segments and scrolls, training at level 0, the teacher's full 60k-step schedule, mixing with
the other six scrolls, diagonal or position-dependent synthetic shifts, and whether the labels sit
exactly on the canvas pixels (assumed, as in the HF65 work; a sub-pixel error there is below one
training voxel). GPU nondeterminism is part of the seed noise.

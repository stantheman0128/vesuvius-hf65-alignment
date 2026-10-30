# Per-segment canvas to tifxyz shifts for the ink training segments

villa#2370 and the ink bucket audit in `audit/` found that on 26 of the 32 training segments in `hf://buckets/scrollprize/datasets/ink/`, the HF65 canvas does not line up with the tifxyz in the same folder. The ink labels and supervision masks are drawn on the canvas grid, so they are off too. Canvas pixel (y, x) shows the CT at tifxyz cell (y + dy, x + dx). villa's full_3d training pairs label pixel (y, x) with tifxyz cell (y, x), so on these segments every label sits dy, dx px away from the CT it is trained against.

This folder gives one correction per segment and a way to apply it:

| file | what it is |
|---|---|
| `shifts.json`, `shifts.csv` | one row per offset segment, 26 rows, with the conventions spelled out in `shifts.json` "convention" |
| `apply_shifts.py` | looks up a segment's row; canvas pixel to tifxyz cell and back; resamples a label tile into the tifxyz frame. The maths is `correct/apply_fit.py`, imported unchanged |
| `selftest_shifts.py` | synthetic tests of `apply_shifts.py`, no data needed |
| `build_shifts.py` | rebuilds `shifts.json` and `shifts.csv` from the measurements in this repo |
| `validate.py`, `cachelayer.py` | the held-out validation below |
| `validation/` | its output: `results.jsonl` (one row per run), `validation.csv`, `batch.log`, `regions_far_<key>.json` |

## Conventions

- Units are level-0 canvas pixels. Slopes are pixels per pixel.
- Canvas (y, x) is a level-0 pixel of `<segment>.zarr`, pixel centres at integers. The labels and masks in the same folder share this grid.
- tifxyz (ty, tx) is a full-resolution cell of the tifxyz in the same folder, in villa's convention: full-res cell (r, c) is stored cell (r * scale_y, c * scale_x), bilinear (`vesuvius/src/vesuvius/tifxyz/types.py:153-154`).
- Canvas content at (y, x) lies at tifxyz cell (y + dy, x + dx), with `dy = dy_a + dy_per_y * y + dy_per_x * x` and `dx = dx_a + dx_per_y * y + dx_per_x * x`, both evaluated at canvas coordinates. This is the sign and form of `correct/apply_fit.py`.
- For training: the label value that belongs to tifxyz cell (ty, tx) is the label at `inv(ty, tx)`. Equivalently, label pixel (y, x) should be paired with tifxyz cell `fwd(y, x)`.

## The 26 rows

| group | segments | model | parameters |
|---|---|---|---|
| one_cell | 1667 w013, w018, w023, w028, w029, w031; 0009b auto_grown_20250919055754487_inp_hr_2um; 0139 w016, w017, w028, w029, w030 | constant | dy -19.71 to -19.54, dx -19.70 to -19.56 |
| second_shift | 0139 w035, w039, w040, w041, w043, w044 | constant | dy +30.59 to +30.84, dx -49.32 to -49.12 |
| paris4_size_ratio | phercparis4 w00, w01, w02, w03, w05, w06, w07, w09 | linear, dy depends on y only and dx on x only | intercepts -42.4 to +11.5 (dy), +11.1 to +23.2 (dx); slopes as in `shifts.csv` |

Where the parameters come from (`params_from` and `fit_regions` in each row):

- 22 rows come from the audit's own registrations, `audit/results/phase2/reg_<key>.jsonl`. A constant row is the mean over that segment's measured regions. A Paris4 row takes its slope per axis from the metadata, tifxyz full size / canvas size - 1, and its intercept as the mean over the measured regions of (d - slope * region centre). These intercepts reproduce the audit's Paris4 table (w02 -20.7, +11.9; w03 -20.8, +11.8; w06 -21.2, +11.9; w07 -42.2, +23.2; w09 -20.7, +12.0).
- 4 rows (phercparis4 w00, w01, w05 and 1667 w018) are `correct/apply_fit.py` FITS copied unchanged, so this file and the published correction agree on those segments. Their slopes are fitted, not taken from the metadata. `selftest_shifts.py` checks that the two give the same model.
- 12 of the 26 rows rest on a single measured region. For those the fit residual column is empty, because one region cannot show a residual. The held-out region below is their only second measurement.
- `fit_resid_max_dy` and `fit_resid_max_dx` are the largest |measured - model| over a row's fit regions. They are at most 0.06 px on the constant rows with two or more regions, and at most 0.47 px on the Paris4 rows.

Not in the file: the five aligned training segments (0500p2 -1, 841 w00, 841 auto_grown_20260220144552896, 841 auto_grown_20260220174252405, MAN5_outer_3) are listed under `aligned_no_correction` and get the identity map. 814_46527_2um_try2 is listed under `not_measured`: its canvas came from a CT scan that is not on public S3, so it was never registered and asking for it is an error.

## How to apply it

From Python, with `correct/shifts` on the path:

```python
import apply_shifts as A
m = A.model("w013_20240304141531_2um")     # key, folder, scroll/folder or ink/scroll/folder
ty, tx = m.fwd(y, x)                         # canvas or label pixel -> tifxyz full-res cell
y, x = m.inv(ty, tx)                         # tifxyz cell -> canvas pixel
out, ok, (ty0, tx0) = A.resample_tile("w013_20240304141531_2um", label_tile, origin=(Y0, X0), level=2, order=0)
```

`resample_tile` takes a 2D tile at pyramid level K whose pixel (0, 0) is level-K canvas pixel (Y0, X0) and returns the same content on the tifxyz level-K lattice: `out[r, c]` is tifxyz level-K cell (ty0 + r, tx0 + c), and `ok` marks the pixels whose source fell inside the tile. Use `order=0` (nearest) for labels and masks and `order=1` for images. Pass a tile a few pixels larger than you need, because the shifted edge has no source.

From the command line (PowerShell or bash, with the repo's Python environment):

```
python -I correct/shifts/apply_shifts.py list
python -I correct/shifts/apply_shifts.py fwd 0139/w035_2026031718 5000 5000
python -I correct/shifts/apply_shifts.py inv phercparis4/w07_1619 18404.82 45571.45
python -I correct/shifts/apply_shifts.py resample 1667/w013_20240304141531_2um tile.npy tile_tifxyz.npy --origin 2000 3000 --level 2 --order 0
```

Two cautions:

- The folder name `w00` exists twice (phercparis4 key `w00` and `841/w00`). A bare `w00` is rejected as ambiguous; use the scroll/folder form.
- The rows are level-0 offsets. If you read labels at a coarser pyramid level, the level-K pixel centre convention of that zarr matters at the sub-pixel scale (`--pyr-shift`, as in `apply_fit.py`). The audit measured pyramid models for the canvases only (`audit/results/phase2/pyr_<key>.json`), not for the label zarrs.

## Held-out validation (2026-10-31)

Every row was checked on one 1024 px region of its segment that the row was not fitted to, using the audit's own registration code (`validate.py`, run as in `run_validation.sh`). The five aligned segments ran as controls with the identity map, and w013 also ran on one of its fit regions. Each region gets three measurements:

- before: the audit's registration with no correction, which is the offset the row has to predict;
- after (method A): the same registration with the CT sampled through `apply_shifts.model`, so the peak is whatever the correction leaves behind;
- method B: the canvas tile moved into the tifxyz frame with `apply_fit.resample`, which is what happens to a label tile.

What came out (table below, raw rows in `validation/results.jsonl`):

- Before correction the 26 held-out regions are off by 6.6 to 147 px on the larger axis. After it, the largest residual on either axis is 0.55 px (Paris4 w06, method A; 0.53 px with method B). No peak sat on the edge of its search window.
- one_cell, 12 regions: after A is +0.03 to +0.14 px on both axes.
- second_shift, 6 regions: after A is -0.28 to -0.06 px.
- paris4_size_ratio, 8 regions: after A is -0.06 to +0.55 px in dy and -0.12 to +0.38 px in dx. The largest prediction error (before minus the model at the region centre) is 0.69 px, also on w06.
- The NCC peak went up on every corrected region, from 0.81 to 0.93 before to 0.86 to 0.94 after.
- Controls: the identity map leaves the CT samples bit-identical and method B returns the canvas tile unchanged on all five. They still read +0.37 to +0.49 px in dy and +0.39 to +0.52 px in dx (mean +0.44 on both), with nothing moved.
- New downloads: 372,958,389 bytes. Reads already in the audit cache cost nothing.

## How accurate the rows are

The registration reads +0.4 to +0.5 px on segments that should read zero (the controls above), and the rows keep the measured values instead of subtracting that. On 0139 w040 there is an independent check. The canvas against the official S3 render of the same mesh peaks at (+30.002, -49.991), and the maintainers' S3 `ink-labels/2.399um-volume-20260102150214/20260918/` labels for w040 are the bucket labels moved by exactly (30, -50). The w040 row is (+30.66, -49.30), about 0.7 px from that on both axes. So treat the rows as good to about 1 px, which is a quarter of a level-2 voxel.

For the six second_shift segments the cause analysis on villa#2370 gives the integer (+30, -50). It is confirmed directly only on w040, because the S3 meshes for w041 and w044 differ from the bucket ones.

## Related tool in villa

`vesuvius/src/vesuvius/tifxyz_label_transfer`, added in villa#1943 on 2026-10-01, already handles a constant canvas offset. The quality assurance section of its README measures the offset from CT and applies it only through `--label-canvas-offset DY DX`, with pherc0139-w039 as the example case. Its `docs/theory.md` says label pixel (i, j) refers to source canvas position (i + dy, j + dx), the same sign as this file, so a constant row's (dy_a, dx_a) can go straight into that flag. The Paris4 rows are linear and cannot be expressed as one constant offset. The ink_detection full_3d loader that villa#2370 is about applies no offset.

## Validation table

| segment | group | region | sample | before dy, dx | model at centre | after residual dy, dx | after NCC | method B dy, dx |
|---|---|---|---|---|---|---|---|---|
| 1667/w013_20240304141531_2um | one_cell | r0 (17408, 12288) | in-sample | -19.70, -19.69 | -19.68, -19.67 | +0.05, +0.09 | 0.884 | +0.12, +0.11 |
| 1667/w013_20240304141531_2um | one_cell | r2 (29184, 18432) | held-out | -19.67, -19.60 | -19.68, -19.67 | +0.08, +0.14 | 0.920 | +0.15, +0.20 |
| 1667/w018_20240304144031_2um | one_cell | new1 (23040, 19968) | held-out | -19.67, -19.63 | -19.64, -19.64 | +0.05, +0.08 | 0.913 | +0.11, +0.16 |
| 1667/w023_20240304161941_2um | one_cell | r1 (29184, 46080) | held-out | -19.66, -19.64 | -19.64, -19.62 | +0.07, +0.08 | 0.917 | +0.13, +0.15 |
| 1667/w028_20251208130119156_2um | one_cell | far0 (36864, 6144) | held-out | -19.62, -19.62 | -19.65, -19.61 | +0.08, +0.03 | 0.923 | +0.14, +0.10 |
| 1667/w029_20251212185248662_2um | one_cell | r1 (8192, 8192) | held-out | -19.73, -19.68 | -19.71, -19.66 | +0.06, +0.05 | 0.918 | +0.13, +0.11 |
| 1667/w031_2025122323_2um | one_cell | r1 (22528, 23552) | held-out | -19.72, -19.73 | -19.69, -19.70 | +0.03, +0.03 | 0.928 | +0.08, +0.08 |
| 0009b/auto_grown_20250919055754487_inp_hr_2um | one_cell | r2 (23040, 7680) | held-out | -19.54, -19.57 | -19.54, -19.56 | +0.08, +0.09 | 0.938 | +0.17, +0.17 |
| 0139/w016_2025010815 | one_cell | r2 (22016, 13824) | held-out | -19.60, -19.60 | -19.60, -19.63 | +0.09, +0.12 | 0.932 | +0.16, +0.17 |
| 0139/w017_2025010815 | one_cell | r1 (10752, 12800) | held-out | -19.62, -19.61 | -19.62, -19.60 | +0.07, +0.06 | 0.929 | +0.14, +0.14 |
| 0139/w028_20260115221 | one_cell | far0 (1024, 1536) | held-out | -19.72, -19.62 | -19.69, -19.64 | +0.05, +0.11 | 0.933 | +0.10, +0.16 |
| 0139/w029_202601261946 | one_cell | r1 (3072, 15872) | held-out | -19.63, -19.64 | -19.62, -19.63 | +0.07, +0.08 | 0.930 | +0.14, +0.15 |
| 0139/w030_202601301912 | one_cell | r1 (22016, 11264) | held-out | -19.60, -19.60 | -19.61, -19.61 | +0.09, +0.09 | 0.930 | +0.16, +0.13 |
| 0139/w035_2026031718 | second_shift | r1 (6144, 8192) | held-out | +30.73, -49.25 | +30.84, -49.12 | -0.24, -0.26 | 0.885 | -0.34, -0.31 |
| 0139/w039_2026030210 | second_shift | r1 (17920, 10752) | held-out | +30.73, -49.23 | +30.73, -49.26 | -0.19, -0.16 | 0.873 | -0.32, -0.23 |
| 0139/w040_2025083102 | second_shift | r2 (9216, 8704) | held-out | +30.64, -49.21 | +30.66, -49.30 | -0.16, -0.10 | 0.867 | -0.26, -0.16 |
| 0139/w041_2026010816 | second_shift | r1 (20992, 17408) | held-out | +30.76, -49.38 | +30.59, -49.32 | -0.06, -0.18 | 0.864 | -0.13, -0.30 |
| 0139/w043_2026011217 | second_shift | far0 (1024, 1536) | held-out | +30.67, -49.32 | +30.73, -49.17 | -0.19, -0.28 | 0.886 | -0.30, -0.37 |
| 0139/w044_2026011522 | second_shift | r1 (17920, 8704) | held-out | +30.77, -49.16 | +30.69, -49.30 | -0.10, -0.11 | 0.874 | -0.21, -0.17 |
| phercparis4/w00_20231016151002 | paris4_size_ratio | new1 (21504, 7680) | held-out | +0.79, +6.58 | +0.65, +6.00 | +0.18, +0.38 | 0.910 | +0.28, +0.38 |
| phercparis4/w01_20230702185753_r15 | paris4_size_ratio | new1 (31744, 15872) | held-out | +147.06, -49.41 | +146.83, -49.70 | +0.12, +0.13 | 0.924 | +0.11, +0.12 |
| phercparis4/w02_20231031143852 | paris4_size_ratio | r2 (32768, 43008) | held-out | +11.05, -10.92 | +10.89, -10.58 | -0.04, -0.09 | 0.913 | -0.12, -0.02 |
| phercparis4/w03_20231106155351 | paris4_size_ratio | r2 (3584, 35840) | held-out | -17.06, -8.29 | -17.16, -8.66 | -0.06, +0.20 | 0.904 | -0.14, +0.13 |
| phercparis4/w05_4424 | paris4_size_ratio | new1 (8192, 26112) | held-out | -35.96, +11.22 | -35.93, +11.08 | -0.02, -0.01 | 0.909 | -0.00, -0.11 |
| phercparis4/w06_1321 | paris4_size_ratio | r2 (4608, 34304) | held-out | -15.64, -9.73 | -16.33, -10.08 | +0.55, +0.19 | 0.896 | +0.53, +0.19 |
| phercparis4/w07_1619 | paris4_size_ratio | r2 (45568, 78336) | held-out | -4.74, -11.07 | -4.99, -10.72 | -0.00, -0.12 | 0.900 | -0.10, -0.09 |
| phercparis4/w09_20230929220926 | paris4_size_ratio | r2 (26112, 62464) | held-out | -1.47, -22.89 | -1.87, -22.66 | +0.16, -0.01 | 0.916 | +0.16, +0.06 |
| 0500p2/-1 | aligned_control | r0 (14848, 6144) | control | +0.49, +0.52 | +0.00, +0.00 | +0.49, +0.52 | 0.939 | +0.50, +0.54 |
| 841/w00 | aligned_control | r0 (9728, 12800) | control | +0.46, +0.44 | +0.00, +0.00 | +0.46, +0.44 | 0.931 | +0.47, +0.44 |
| 841/auto_grown_20260220144552896 | aligned_control | r0 (4096, 11264) | control | +0.47, +0.44 | +0.00, +0.00 | +0.47, +0.44 | 0.903 | +0.48, +0.45 |
| 841/auto_grown_20260220174252405 | aligned_control | r0 (6144, 10752) | control | +0.43, +0.39 | +0.00, +0.00 | +0.43, +0.39 | 0.926 | +0.44, +0.40 |
| man5/MAN5_outer_3 | aligned_control | r0 (7680, 12800) | control | +0.37, +0.41 | +0.00, +0.00 | +0.37, +0.41 | 0.906 | +0.38, +0.41 |


# Correcting the HF65 canvas offset (villa#2370)

On the four segments in villa#2370, sampling the CT at the tifxyz cells given by the README fits brings HF65 slice 32 and the CT within 0.4 px of each other. That holds on all 24 regions the fits were made from and on 8 regions they never saw. Before the correction the offsets were up to 254 px. In this document, offset and residual values are level-0 canvas px. Every value here comes from runs whose logs are in `logs/`.

## What was applied

Canvas content at level-0 pixel (y, x) lies at tifxyz full-res cell (y + dy, x + dx). Full-res cell (r, c) is stored cell (r*sy, c*sx), as in villa `vesuvius/src/vesuvius/tifxyz/types.py:153-154`. The README fits as first published, with only the dominant slope per axis, as `correct/apply_fit.py` holds them (model "readme"):

| segment | dy | dx |
|---|---|---|
| phercparis4 w00_20231016151002 | -20.70 + 0.000993*y | +11.09 - 0.000663*x |
| phercparis4 w01_20230702185753_r15 | +11.51 + 0.004263*y | +11.82 - 0.003876*x |
| phercparis4 w05_4424 | -42.39 + 0.000789*y | +22.93 - 0.000454*x |
| 1667 w018_20240304144031_2um | -19.64 | -19.64 |

Before using the fits I checked their coordinates against `measure/summarize.py` and `results_*.jsonl`. They are in level-0 canvas px, evaluated at the region centre (y0 + size/2), and they come after the w00 pyramid correction. The README prints only the dominant slope per axis. Its intercepts come from the full fit `a + b*y + c*x`, and `apply_fit.py check` reproduces them and the slopes to the printed digits. The README gives no number for w018 beyond "constant". -19.64 is the mean of the six published regions on both axes. The README's own held-out prediction for w01 at (46016, 27008), +207.68 / -92.86, comes out of `apply_fit.py fwd` unchanged.

`apply_fit.py inv` inverts the map by fixed-point iteration, `y <- ty - dy(y, x)`. With 6 iterations the largest difference from the closed-form inverse, over each segment's whole tifxyz extent plus 300 px, is 1.5e-11 px. Three iterations already give 2.3e-5 px. See `logs/apply_fit_check.log`.

## How the correction was tested

`correct/regcorr.py` is `measure/reg.py` with one change. In "after" mode the CT is sampled at tifxyz cell fwd(canvas lattice point) instead of at the canvas lattice point, and the search window is centred on zero. Everything else is unchanged: the level-2 lattice, the w00 pyramid shift of 3 rows (from `sources.pyr_shift`), CT level 2 with trilinear sampling, high-pass sigma 4, NCC search radius 32 level-2 px and the parabolic sub-pixel peak. "Before" mode is `reg.py` as published. On all 24 published regions it reproduced `results_<key>.jsonl` exactly, with a difference of 0 in dy, dx, high-pass NCC and raw NCC (`logs/before_vs_published.log`). That includes the six w00 regions, so the w00 pyramid correction is still in effect.

A second, separate check ("method B") needs no new downloads. It takes the HF65 slice of each before-run, moves it into the tifxyz frame with `apply_fit.resample` (inverse map, bilinear), and runs the same NCC against the CT samples that run already holds. Method A tests the forward map. Method B tests the inverse map and the resampler that a user would apply to labels.

## Before and after

"in" regions are the six per segment that the fits were made from, so a small after-residual there is partly circular. "OUT" regions were not used by the fits:

- w01 at45504_26496 is the held-out region from the README's independent check, centred at (46016, 27008).
- new0 and new1 are new regions. `correct/newregions.py` picked them as picks 7 and 8 of the same farthest-point selection that `measure/regions.py` uses. It first recomputed picks 1 to 6 and got exactly the published `regions_<key>.json` for all four segments.

NCC is the high-pass NCC. "NCC at 0" is the NCC with no shift between canvas and tifxyz. It is n/a where the before search window, centred on the size-ratio prediction as in the published w01 runs, does not contain zero shift.

| segment | region | sample | centre (y, x) | before dy, dx | before peak NCC | before NCC at 0 | after residual dy, dx | after peak NCC | after NCC at 0 | method B residual |
|---|---|---|---|---|---|---|---|---|---|---|
| w00 | r0 | in | 11264, 21504 | -9.32, -2.94 | 0.859 | 0.309 | +0.03, +0.25 | 0.916 | 0.916 | -0.02, +0.37 |
| w00 | r1 | in | 17920, 2560 | -2.96, +9.86 | 0.828 | 0.341 | +0.16, +0.40 | 0.904 | 0.904 | +0.27, +0.51 |
| w00 | r2 | in | 7168, 37376 | -13.65, -13.45 | 0.842 | -0.024 | -0.09, +0.13 | 0.912 | 0.912 | -0.13, +0.12 |
| w00 | r3 | in | 16896, 31232 | -3.86, -9.27 | 0.871 | 0.320 | +0.10, +0.18 | 0.899 | 0.899 | +0.12, +0.16 |
| w00 | r4 | in | 4096, 27648 | -16.58, -7.19 | 0.882 | -0.013 | -0.06, +0.16 | 0.899 | 0.899 | -0.11, +0.25 |
| w00 | r5 | in | 14336, 14848 | -6.51, +1.49 | 0.821 | 0.454 | +0.06, +0.26 | 0.899 | 0.899 | +0.12, +0.35 |
| w00 | new0 | OUT | 4096, 20480 | -16.59, -2.11 | 0.815 | 0.111 | -0.08, +0.30 | 0.877 | 0.877 | -0.15, +0.40 |
| w00 | new1 | OUT | 21504, 7680 | +0.79, +6.58 | 0.869 | 0.583 | +0.18, +0.38 | 0.910 | 0.910 | +0.25, +0.39 |
| w01_r15 | r0 | in | 28160, 27648 | +131.70, -95.33 | 0.868 | 0.014 | +0.10, +0.14 | 0.911 | 0.911 | +0.10, +0.14 |
| w01_r15 | r1 | in | 56832, 18432 | +253.88, -59.66 | 0.794 | n/a | +0.14, +0.13 | 0.905 | 0.905 | +0.14, +0.11 |
| w01_r15 | r2 | in | 9728, 43008 | +53.22, -155.05 | 0.824 | n/a | +0.10, +0.15 | 0.904 | 0.904 | +0.09, +0.14 |
| w01_r15 | r3 | in | 56320, 37888 | +251.82, -134.95 | 0.860 | n/a | +0.15, +0.13 | 0.926 | 0.926 | +0.16, +0.13 |
| w01_r15 | r4 | in | 14336, 16384 | +72.74, -51.54 | 0.859 | 0.012 | +0.14, +0.13 | 0.908 | 0.908 | +0.15, +0.13 |
| w01_r15 | r5 | in | 40960, 36352 | +186.41, -128.63 | 0.821 | n/a | +0.13, +0.14 | 0.930 | 0.930 | +0.14, +0.13 |
| w01_r15 | at45504_26496 | OUT | 46016, 27008 | +207.88, -92.56 | 0.873 | n/a | +0.15, +0.13 | 0.930 | 0.930 | +0.17, +0.11 |
| w01_r15 | new0 | OUT | 13824, 29184 | +70.79, -100.96 | 0.822 | -0.015 | +0.14, +0.11 | 0.894 | 0.894 | +0.17, +0.09 |
| w01_r15 | new1 | OUT | 31744, 15872 | +147.06, -49.41 | 0.851 | n/a | +0.12, +0.13 | 0.924 | 0.924 | +0.11, +0.12 |
| w05_4424 | r0 | in | 18432, 39936 | -27.86, +4.57 | 0.918 | 0.007 | +0.00, -0.08 | 0.924 | 0.924 | +0.04, -0.00 |
| w05_4424 | r1 | in | 48128, 66560 | -4.39, -7.48 | 0.911 | 0.365 | -0.05, -0.08 | 0.921 | 0.921 | -0.08, -0.02 |
| w05_4424 | r2 | in | 46592, 13824 | -5.56, +16.48 | 0.865 | 0.175 | -0.04, -0.01 | 0.909 | 0.909 | -0.08, +0.06 |
| w05_4424 | r3 | in | 10752, 72192 | -33.77, -9.81 | 0.825 | -0.044 | +0.01, -0.09 | 0.920 | 0.920 | -0.01, -0.13 |
| w05_4424 | r4 | in | 20480, 14336 | -26.25, +16.41 | 0.870 | -0.112 | +0.04, +0.03 | 0.909 | 0.909 | +0.07, +0.08 |
| w05_4424 | r5 | in | 28672, 58368 | -19.79, -3.68 | 0.913 | 0.085 | +0.01, -0.03 | 0.917 | 0.917 | +0.04, +0.01 |
| w05_4424 | new0 | OUT | 36352, 28672 | -13.61, +9.70 | 0.845 | 0.053 | -0.01, -0.10 | 0.923 | 0.923 | -0.03, -0.11 |
| w05_4424 | new1 | OUT | 8192, 26112 | -35.96, +11.22 | 0.895 | -0.043 | -0.02, -0.01 | 0.909 | 0.909 | -0.00, -0.11 |
| w018 | r0 | in | 17408, 36864 | -19.67, -19.69 | 0.914 | -0.054 | +0.06, +0.05 | 0.920 | 0.920 | +0.12, +0.11 |
| w018 | r1 | in | 26112, 95744 | -19.64, -19.65 | 0.914 | -0.056 | +0.09, +0.08 | 0.921 | 0.921 | +0.14, +0.14 |
| w018 | r2 | in | 26112, 2048 | -19.69, -19.62 | 0.925 | -0.038 | +0.03, +0.09 | 0.931 | 0.931 | +0.10, +0.16 |
| w018 | r3 | in | 5632, 68608 | -19.63, -19.67 | 0.905 | -0.065 | +0.09, +0.07 | 0.911 | 0.911 | +0.16, +0.13 |
| w018 | r4 | in | 28160, 72704 | -19.62, -19.61 | 0.915 | -0.049 | +0.10, +0.09 | 0.922 | 0.922 | +0.15, +0.16 |
| w018 | r5 | in | 6144, 5632 | -19.60, -19.58 | 0.920 | -0.027 | +0.11, +0.12 | 0.927 | 0.927 | +0.17, +0.18 |
| w018 | new0 | OUT | 6656, 90112 | -19.61, -19.62 | 0.900 | -0.083 | +0.12, +0.11 | 0.907 | 0.907 | +0.19, +0.17 |

Summary (`logs/table.log`):

| group | regions | after max abs dy, dx | after mean abs dy, dx | method B max abs dy, dx |
|---|---|---|---|---|
| in-sample | 24 | 0.16, 0.40 | 0.08, 0.13 | 0.27, 0.51 |
| out-of-sample | 8 | 0.18, 0.38 | 0.10, 0.16 | 0.25, 0.40 |

After correction, the integer NCC peak is at zero shift in all 32 regions and never on the edge of the search window (`logs/after_edge_check.log`), so the NCC at zero shift equals the peak NCC. Before correction the NCC at zero shift was between -0.11 and 0.58. It is high only where the offset itself is a few px, as in w00 new1 at (+0.8, +6.6).

Two patterns stay in the residuals:

- w00 dx is +0.13 to +0.40 after correction. The README form drops the cross term of the w00 dx fit, which is +0.0000224*y. Method B with the full 3-term fit (`apply_fit.py` model "affine3", recomputed from the published jsonl) brings w00 new1 dx from +0.39 to -0.08 and r1 from +0.51 to +0.11. Over the whole w00 canvas the README form and the 3-term fit differ by up to 0.72 px in dx (`logs/apply_fit_check.log`).
- All 9 w01 after-residuals are +0.10 to +0.15 px on both axes, and all 7 w018 ones are +0.03 to +0.12. That holds for the out-of-sample regions too. I did not trace where this common sub-pixel bias comes from.

## Images

- `png/w01_before_after.png`, 121 KB: w01 new0, out of sample. The before offset there is +70.8 / -101.0 px.
- `png/w018_before_after.png`, 121 KB: w018 new0, out of sample. The before offset there is -19.6 / -19.6 px.

Each image is a checkerboard of 128 px squares that alternate the high-passed HF65 slice 32 and the high-passed CT on the tifxyz. The left panel samples the CT at tifxyz cell (y, x), the right panel at fwd(y, x). On the left, structures break at every square edge. On the right they run straight across.

## Downloads and checks on the tooling

New data came to 255.84 MB in 35,981 HTTP range or file requests, all through `measure/common.fetch`. It is cached under `$HF65_DATA/cache`, and every request was appended to the cache's `ledger.tsv`, which went from 43,440 to 79,421 lines. The summary is in `logs/ledger_summary.txt`; the per-run request ledgers are not included here.

`correct/rangecache.py` serves a requested range from bytes that are already cached under other ranges and downloads only the gaps. Without it, each after-run would have downloaded its whole CT box again. To check it, I ran the held-out w01 after-run a second time with plain `common.fetch`. 7.5 MB of that region's CT bytes had been stitched from cache. The CT samples, the mask and the NCC map came out identical (`logs/batch_more.log`, "covercheck2").

The before-runs were also bit-identical to the published results through the same layer.

`correct/selftest_apply_fit.py` tests the resample CLI (`logs/selftest_apply_fit.log`):

- On two real level-2 tiles its output matches method B's array to 7.6e-6.
- Single labelled pixels land at round(fwd(y, x)) on all four segments.
- A canvas to tifxyz to canvas round trip with nearest sampling agrees on 99.89% of interior pixels.

The four parallel after-runs failed with HTTP 429 from the Hugging Face tifxyz files. Everything after that ran one process at a time, and the failed regions were rerun.

## What this does not show

- Only 8 out-of-sample regions, at most 3 per segment, and only one per segment for w018. Other Paris4 and 1667 segments were not looked at.
- The held-out w01 region lies outside the supervision mask: its level-5 mask fraction is 0.000 (`logs/newregions.log`). The new regions lie inside it, with mask fractions 0.903 to 1.000.
- Labels: the correction is measured on the HF65 image. That the ink labels share the HF65 canvas grid is assumed here, as in the README.
- The cause of the offsets, and of the +0.1 px common residual.
- villa's own loader end to end. The index convention is the bundle's reimplementation.
- Image pyramid levels other than level 0 and level 2. For w00, someone using HF65 level 1, 2 or 3 directly also needs the pyramid shift: `apply_fit.py resample --pyr-shift`.

## Other things noticed, not chased

- The first README printed the fits without their cross terms. In that form the in-sample residual is up to 0.47 px (w00 r1 dx), while the README said 0.35 px or less, which holds for the 3-term fit. The README now lists all three terms.
- The README's independent-check region at (46016, 27008) is outside the w01 supervision mask, while "areas outside the supervision masks" was listed under "Not checked". The README now says so.

## Rerun

Python 3.14 with numpy, scipy, httpx and numcodecs. The overlays and the MAN5 overlay also need Pillow. Paths are relative to this folder. `HF65_DATA` is a cache with the bundle's layout. Run one process at a time, because parallel runs hit HTTP 429.

```
export HF65_DATA=/path/to/hf65_data
python -I correct/apply_fit.py check                       # fit table vs jsonl, inversion error
python -I correct/newregions.py w01 2                      # picks 7 and 8, writes correct/regions_new_w01.json
python -I correct/regcorr.py w01 before 0 1 2 3 4 5 new0 new1 45504,26496
python -I correct/regcorr.py w01 after  0 1 2 3 4 5 new0 new1 45504,26496
python -I correct/table.py                                 # the table above
python -I correct/overlay.py w01 new0 png/w01_before_after.png
python -I correct/selftest_apply_fit.py
python -I correct/apply_fit.py resample w01 tile.npy tile_tifxyz.npy --origin Y0 X0 --order 0
```

The same commands work for w00, w05 and c018, the key for w018. Logs: in-sample rows come from `logs/before_<key>.log` and `logs/after_<key>.log`, and the rest come from `correct/batch_oos.sh` (`logs/batch_oos.log`) and `correct/batch_more.sh` (`logs/batch_more.log`).

## MAN5_outer_3 label frame (villa#2369)

In `ink/man5/MAN5_outer_3`, the three `MAN5_outer_3_*` label zarrs are 24061 x 29841 at level 0. The segment zarr and `front_inklabels.zarr` are 24120 x 29900. `man5/place.py` puts a `MAN5_outer_3_*` label on the segment grid with a given level-0 offset: label pixel (y, x) goes to grid pixel (y + dy, x + dx), padded with zeros and cropped to the grid. Level 0 takes a window; levels 1 and 2 cover the whole slice. A fractional level-k shift, such as 29 px at level 1, is applied by linear interpolation.

The two files are different annotations. At level 1, slice 32, `front_inklabels` has 2,341,009 ink pixels and `MAN5_outer_3_inklabels` has 1,083,591, so front has 2.16 times as much ink. Even at the best shift, only 41% of front ink is covered by MAN5 ink. Shifting fixes which grid the MAN5 labels sit on. It says nothing about which file should be used for training.

Numbers for slice 32 (`logs/man5_eval.log`). "IoU in box" counts front ink only inside the MAN5 ink bounding box widened by 80 level-0 px, as the offset check in villa#2369 does. My rerun of that check gives 0.444 with no shift and 0.485 at its best (`logs/man5_offsetpy_rerun.log`). "MAN5 on front" is the share of MAN5 ink pixels that land on front ink.

| level-0 offset | why | L1 IoU | L1 IoU in box | L1 MAN5 on front | L2 IoU | L2 IoU in box | L2 MAN5 on front |
|---|---|---|---|---|---|---|---|
| 0, 0 | as stored | 0.3618 | 0.4425 | 0.8396 | 0.3616 | 0.4422 | 0.8389 |
| 29, 29 | centred 59 px crop | 0.3923 | 0.4823 | 0.8905 | 0.3921 | 0.4819 | 0.8898 |
| 29, 30 | centred 59 px crop | 0.3924 | 0.4824 | 0.8906 | 0.3922 | 0.4820 | 0.8899 |
| 30, 29 | centred 59 px crop | 0.3924 | 0.4825 | 0.8907 | 0.3922 | 0.4820 | 0.8899 |
| 30, 30 | centred 59 px crop | 0.3925 | 0.4826 | 0.8908 | 0.3922 | 0.4821 | 0.8900 |
| 30, 34 | offset.py best, level-1 (15, 17) | 0.3925 | 0.4826 | 0.8908 | 0.3923 | 0.4822 | 0.8901 |
| 30, 32 | best whole-slice IoU at level 1 | 0.3925 | 0.4826 | 0.8909 | | | |
| 32, 32 | best whole-slice IoU at level 2 | | | | 0.3924 | 0.4823 | 0.8903 |

Two different procedures produced these numbers:

- For integer level-k shifts the counts are identical to counting `place.py` output.
- For fractional shifts they interpolate a table of integer shifts. That equals counting on level-0-upsampled labels; a brute-force test agreed to 0 pixels (`logs/man5_interp_selftest.log`).

The local copy of levels 1 and 2 is complete. Every level-5 ink pixel has level-1 and level-2 ink under it, and 12 locally absent chunks next to present ones all return 404 from the bucket (`logs/man5_localcheck.log`).

What the data supports is the step from 0 to about 30 px. It raises whole-slice IoU from 0.362 to 0.392 and the share of MAN5 ink on front ink from 0.840 to 0.891. The exact value is not pinned:

- At level 1, whole-slice IoU stays within 0.3917 to 0.3925 for dy from 24 to 36 px and within 0.3921 to 0.3925 for dx from 26 to 38 px.
- In one 2048 px level-0 window (grid y 6462 to 8510, x 12512 to 14560), IoU peaks at (35, 57).
- Six 1024 px level-1 tiles were scored (`logs/man5_tiles.log`). The four whose peak is inside the search range peak at level-0 (18, 20), (58, 32), (8, 54) and (32, 28). Where it could be measured, IoU falls by only about 0.002 at 4 level-1 px from the peak.

A blobby, differently drawn second annotation is too blunt to tell 29 from 34. A sharper test would align the edge of `MAN5_outer_3_supervision_mask` with the edge of the segment image. I did not run it.

`png/man5_before_after.png`, 23 KB, shows level 1 at offset (0, 0) and at (30, 30). Blue is MAN5 ink only, orange is front ink only and black is both. The blue fringe along the upper edge mostly disappears after the shift.

Rerun: `python -I man5/evaluate.py`, `python -I man5/tiles.py`, `python -I man5/overlay_man5.py 30 30 png/man5_before_after.png`, `python -I man5/place.py inklabels 1 30 30 out.npz`. `MAN5_LOCAL` points at the local tree. Anything missing locally is read over HTTP through the same cache and ledger. The MAN5 work read 0.13 MB of new data.

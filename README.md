# HF65 canvas vs tifxyz offset (2026-10-30)

Measurements behind a ScrollPrize/villa issue: on the segments below, pixel (y, x) of the 65-slice HF surface volume ("HF65 canvas") in `scrollprize/datasets` `ink/<scroll>/<segment>/` does not show the CT at tifxyz full-res cell (y, x) of the tifxyz in the same folder.

## Result

Offset convention: canvas content at (y, x) lies at tifxyz cell (y + dy, x + dx), in level-0 px. Six 1024 px regions per segment, inside the supervision mask.

| segment | dy range | dx range | fit (per axis, d = a + b*y + c*x) | peak NCC | best other peak |
|---|---|---|---|---|---|
| phercparis4 w00_20231016151002 | -16.6 to -3.0 | -13.5 to +9.9 | dy -20.70 + 0.000993*y, dx +11.09 - 0.000663*x | 0.82 to 0.88 | 0.12 or less |
| phercparis4 w01_20230702185753_r15 | +53.2 to +253.9 | -155.0 to -51.5 | dy +11.51 + 0.004263*y, dx +11.82 - 0.003876*x | 0.79 to 0.87 | 0.15 or less |
| phercparis4 w05_4424 | -33.8 to -4.4 | -9.8 to +16.5 | dy -42.39 + 0.000789*y, dx +22.93 - 0.000454*x | 0.82 to 0.92 | 0.17 or less |
| 1667 w018_20240304144031_2um | -19.7 to -19.6 | -19.7 to -19.6 | constant (sd 0.03) | 0.91 to 0.93 | 0.10 or less |

Fit residuals are 0.35 px or less. The Paris4 slopes equal (tifxyz full-res size / canvas size - 1) on each axis, which comes from the metadata alone:

| segment | rows: tifxyz / canvas | cols: tifxyz / canvas |
|---|---|---|
| w00 | 32281 / 32249 (0.000992) | 51345 / 51380 (-0.000681) |
| w01 | 65698 / 65419 (0.004265) | 56669 / 56889 (-0.003867) |
| w05 | 51440 / 51400 (0.000778) | 76120 / 76155 (-0.000460) |
| w018 | 42632 / 42632 (0) | 98362 / 98362 (0) |

On w018 the sizes agree and the offset is constant. Its tifxyz scale is 0.04997, so one stored cell is 20.01 px; that fits the measured -19.6 plus the renderer's +0.35, but the cause is not established.

Method control: S3 surface volumes rendered from their own meshes (`PHerc1667/segments/20240304144031-w018_20240304144031_flatboi` and `PHercParis4/segments/20231016151002`) give +0.3 to +0.4 px with the same code, the renderer's half-pixel convention.

## How it was measured

CT volumes: `s3://vesuvius-challenge-open-data/PHercParis4/volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr` (its metadata names raw data from the 20260311 2.4 um Paris4 scan, the same scan the HF65 `source_zarr` path `/volumes/esrf/20260311/2.4um_PHerc-Paris4_masked.zarr/` names) and `PHerc1667/volumes/20251217075048-2.399um-0.2m-78keV-masked.zarr`, level 2, trilinear. tifxyz index convention as in villa `vesuvius/src/vesuvius/tifxyz/types.py:153-154` (full-res cell (r, c) is stored cell (r*sy, c*sx), bilinear). The CT is sampled at the tifxyz points on the canvas level-2 pixel lattice and compared with high-passed slice 32 by masked NCC over integer shifts, with a parabolic sub-pixel fit. Slice 32 was the best-matching slice in every region.

The w00 HF65 image pyramid is shifted: level 1, 2 and 3 are offset by -1, -3 and -7 rows against level 0 (exact block-mean match), columns are not. `measure/` corrects for it; without the correction the w00 dy values move by 3 px. w01 level 2 and w018 level 1 are not shifted.

## Independent check

`check/` is a second implementation written separately (own TIFF and zarr readers, own sampler, own NCC, high-pass sigma 3 instead of 4). It reproduced w018 (2 regions), the S3 control, w00 (2 regions) and w01 (2 regions, searched within +-320 px of zero) to about 0.15 px, and one w01 region that `measure/` never used: the fit predicts +207.68 / -92.86 at (46016, 27008), and the check measured +208.03 / -92.55 (NCC 0.880). Its output is `results/check_myresults.jsonl` and `results/check_batch1.log`.

## Rerun

Python 3.14 with numpy, scipy and httpx. All data comes from public URLs by HTTP range reads and is cached under `HF65_DATA` (default `./hf65_data`); the first full run reads about 500 MB.

```
export HF65_DATA=$PWD/hf65_data
python -I measure/reg.py c018 0          # one w018 region, prints level-0 (dy,dx) = (-19.7,-19.7)
python -I measure/reg.py w01             # all regions of a segment
python -I measure/summarize.py w00 w01 w05 c018
```

`measure/regions_<key>.json` holds the regions and `results/results_<key>.jsonl` the outputs used above. Running `measure/reg.py c018 0` from this folder against the cache gave the same numbers to the last digit.

## Not checked

The other Paris4 and 1667 segments, areas outside the supervision masks, where the Paris4 intercepts come from, villa's own loader end to end (the index convention was reimplemented), and whether the labels sit exactly on the canvas pixels (assumed).

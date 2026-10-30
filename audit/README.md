# Ink bucket audit (2026-10-30)

The top-level README measures the HF65 canvas against the tifxyz on the four segments of [villa#2370](https://github.com/ScrollPrize/villa/issues/2370). This folder runs the same measurement on every training segment folder in `hf://buckets/scrollprize/datasets/ink/` and adds a metadata check of all 93 segment folders there, including the 61 under `ink/unused/`.

## Result

There are 32 training segment folders, meaning every segment folder outside `ink/unused/`. At villa 37d1897fb the `dino_guided_teacher` config (mode `full_3d`, no segment allowlist) names all eight scroll dirs, and `dino_guided_v1`, `v2`, `v3` and `v3_fullsup` name `phercparis4`. Matching the config dataset names to bucket dirs is an inference.

Of the 32, 26 have a canvas that does not line up with the tifxyz in the same folder, 5 line up and 1 was not measured. The 26 are the 24 segments registered as offset in this audit, including the two controls 1667 w018 and Paris4 w00, plus Paris4 w01 and w05, which were measured for #2370 (top-level README) and not registered again here. So the per-region files in `results/phase2/` cover 29 distinct segments: 24 offset and 5 aligned.

Offset convention as in the top-level README: canvas content at (y, x) lies at tifxyz full-res cell (y + dy, x + dx), in level-0 px.

| group | segments | dy, dx (level-0 px) | regions | peak NCC | best other peak |
|---|---|---|---|---|---|
| one-cell shift (12) | 1667 w013, w018, w023, w028, w029, w031; 0009b `auto_grown_20250919055754487_inp_hr_2um`; 0139 w016, w017, w028, w029, w030 | -19.7 to -19.5 on both axes | 16 | 0.880 to 0.934 | 0.194 or less |
| second shift (6) | 0139 w035, w039, w040, w041, w043, w044 | +30.6 to +30.8, -49.3 to -49.1 | 7 | 0.811 to 0.841 | 0.105 or less |
| Paris4 size-ratio drift (8) | phercparis4 w00, w01, w02, w03, w05, w06, w07, w09 | depends on position: up to 35.5 px measured here (w07), 254 px on w01 (#2370) | 11 | 0.811 to 0.917 | 0.124 or less |
| aligned (5) | 0500p2 `-1`; 841 `w00`, `auto_grown_20260220144552896`, `auto_grown_20260220174252405`; man5 `MAN5_outer_3` | +0.4 to +0.5 on both axes | 5 | 0.903 to 0.939 | 0.097 or less |
| not measured (1) | 814 `814_46527_2um_try2` | | 0 | | |

The per-segment table is `results/segments.csv` (summary counts in `results/summary.json`), and every registered region is listed in `results/report_tables.md`. Peak NCC is the audit's NCC on high-passed images. Its lower range on the second-shift group is not a sign of weaker evidence there: the independent check below finds raw NCC 0.952 to 0.964 on those segments at the measured shift, against 0.959 to 0.975 on the other segments it checked.

**One-cell shift.** The offset is constant over the segment and equals the w018 value in #2370. These tifxyz have scale close to 0.05, so one stored cell is about 20 level-0 px, and -19.6 is -20 plus the renderer's +0.35. The tifxyz full-res size, the canvas size and the label zarr sizes are identical on all 12, so nothing in the metadata shows the shift and villa's `validate_segments.py` passes them. On the three segments with two regions far apart (1667 w013, 0009b, 0139 w016) the two values agree within 0.1 px. The two 1667 w028 regions overlap (centres 512 px apart), so they are not a separate check. w018 is the #2370 case; the other 11 are new.

**Second shift.** A constant (+30.7, -49.3) px on six 0139 segments, where the three sizes also agree and the validator passes. The cause is unknown. The canvas `.zattrs` of these folders record no source volume, so they were registered against `PHerc0139/volumes/20260102150214-2.399um-0.2m-78keV-masked.zarr`. Region 0 of w040 was also run against the other public export of that scan, `20260930100252-2.399um-0.2m-78keV-masked.zarr` (key `t040b`). It gave the same peak, NCC and second peak to every printed digit, so the two exports are probably the same bytes, and that run is not counted as separate evidence.

**Paris4 drift.** All eight Paris4 training segments have a tifxyz full-res size that differs from the canvas size, by 23 to 279 px per axis. w01 and w05 were measured in the top-level README and not registered again here. On w02, w03, w06, w07 and w09 two regions each fit d = a + (full/canvas - 1) x position with a constant intercept: about (-21, +12) px at tifxyz scale 0.1 and (-42, +23) px on w07 at scale 0.05, which matches the w05 fit in the top-level README. Extrapolated to the canvas edges the offset reaches 42 px (w07 rows); the largest value measured in a region is 35.5 px (w07 region 1, rows). Per-segment intercepts and extrapolated maxima are in `results/phase2/paris4_maxoffset.json`.

**Aligned.** These five sit at the renderer's +0.35 px within the 1.5 px tolerance. MAN5_outer_3 is among them: its canvas lines up with the tifxyz, and the #2369 problem is in its label zarrs.

**Not measured.** The canvas of `814_46527_2um_try2` was rendered from a 2.403 um scan that is not on the public S3 bucket. The 2.4 um PHerc0814 volume that is there (20251130) is a different scan.

## Method

`code/reg2.py` generalises `measure/reg.py`, `measure/regions.py` and `measure/pyrshift.py` of this repo to any key in `code/targets.json`, which gives the segment folder, the supervision mask used to pick regions and the S3 CT volume. For each segment it first finds the canvas pyramid level and slice that equal level-0 slice 32 (r > 0.99), together with the lattice offset that follows from it. It then picks up to four 1024 px regions inside the supervision mask where the tifxyz is valid. For each region it samples the CT (level 2, trilinear) at the tifxyz cells on the canvas level-2 lattice and finds the shift that maximises masked NCC between that sample and high-passed slice 32, over ±128 level-0 px, with a parabolic sub-pixel fit. Paris4 segments are run with `--pred`, which centres the search window on the size-ratio prediction.

A region is inconclusive if the peak NCC is below 0.4, the peak is on the window edge, or the best other peak is above half the peak. None of the 39 regions was inconclusive. It is aligned if dy and dx are both within 1.5 px of +0.35, and offset otherwise.

`code/common.py` and `code/lib.py` are copies of the `measure/` files with these changes: HF range reads reuse the CDN redirect and wait out the `RateLimit` window, canvas reads cover every z chunk (some canvases have 64-slice chunks over 65 slices), downloads use 4 threads, and the local mirror is never read.

For 1667 w029 the `_2um` supervision mask that the loader reads has 0.1% of pixels set at level 5, and the region picker found no candidates (key `c029`, `results/phase2/regions_c029.json`). Key `c029v` uses the `w029_20251212185248662_supervision_mask_v2.zarr` mask only to pick regions; the canvas, tifxyz and CT are the same.

Some regions were run twice. The repeated rows in `results/phase2/reg_*.jsonl` are identical apart from run time and bytes downloaded, and the tables use the last row.

## Controls

The aligned segments come out at +0.4 to +0.5 px, the renderer's half-pixel convention, so the method does not add a shift of its own on segments whose canvas matches the tifxyz. 1667 w018 region 0 gives (-19.7, -19.7) against the published constant (-19.64, -19.64). Paris4 w00 region 0 gives (-9.3, -2.9) where the published w00 fit predicts (-9.5, -2.9) at the same centre (`results/phase2/paris4_check.json`).

## Label assets the loader ignores (flags d and f)

`code/labels_check.py` runs villa's own `parse_label_asset_path` and `discover_segment_labels` from `vesuvius/src/vesuvius/ink_detection/data/segment.py` at 37d1897fb on the bucket listing, with stub segment and config objects. Flag d means a label asset whose prefix differs from the folder name, which the loader skips. Flag f means such an ignored asset is a newer version than the one the loader reads, or is the folder's only validation mask.

9 training folders have flag d and 6 have flag f. The six f folders are 1667 w018, w023, w028, w029 and w031 (all `_2um`) and 0009b. In each, the loader reads the `_2um` v1 inklabels and supervision mask, while `_v2` inklabels, supervision mask and validation mask under the name without `_2um` are ignored, so the loader also finds no validation mask. [villa PR #2140](https://github.com/ScrollPrize/villa/pull/2140), opened by another contributor and still open, adds a warning for a validation mask that is present but not selected. It is listed here as context and is not a new finding of this audit. Per-folder details are in `results/labels_check.json`.

The metadata part of villa's `validate_segments.py`, reproduced in `code/phase1.py`, fails one training folder, MAN5_outer_3 (#2369). It does not compare the tifxyz with the canvas, and it accepts label files with any prefix, so it sees none of the shifts above and neither flag d nor f.

## Independent check

`independent/` is a second implementation written separately. It has its own tifxyz reader, CT sampler and canvas pyramid check, and it registers by phase correlation on the gradient magnitude instead of NCC on high-passed images; it shares only the HTTP fetch and the canvas chunk decoder with `code/`. On 1667 w013, 0139 w035 and w040 and Paris4 w07 it agrees with the audit within 0.4 px on every region both measured, and it adds two regions the audit never used (w013 at y0, x0 = 28672, 17920 and w035 at 19200, 18432). Within w013, w035 and w040 the shift stays constant across regions 20,000 px or more apart, to within 0.1, 0.4 and 0.0 px. On w07 its two regions differ by 8.4 and 23.4 px and both match the audit's size-ratio model within 0.4 px. The controls 0500p2 and MAN5 come out at +0.2 to +0.4 px.

For 0139 w040 it also compared the bucket canvas with the S3 render of the same segment, `PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr`. That render lines up with the CT sampled at the tifxyz within about 0.6 px, and the bucket canvas sits about (+29.9, -49.6) px from it. The bucket tifxyz is the S3 mesh on 20260102150214: the `meta.json` bbox agrees to 0.001, the `x.tif` dimensions are equal, and a compared `x.tif` row differs by at most 0.004. Sampling the March 2026 volume (20260319133554) at the S3 mesh made for it fits worse on every score. On w040, then, the S3 render agrees with the tifxyz and the bucket canvas does not.

Notes, commands and per-region rows: `independent/NOTES.md`, `independent/out/results.jsonl`. Overlays (canvas, CT at the tifxyz, red/cyan overlay before and after the measured shift):

- [control_p500_r0.png](independent/control_p500_r0.png), aligned control 0500p2
- [control_man5_r0.png](independent/control_man5_r0.png), aligned control MAN5_outer_3
- [onecell_c013_r1.png](independent/onecell_c013_r1.png), one-cell shift on 1667 w013
- [second_t040_r0.png](independent/second_t040_r0.png), second shift on 0139 w040
- [paris4_p4w07_r1.png](independent/paris4_p4w07_r1.png), Paris4 drift on w07

## Not checked

- The validator's binary-value check on label content.
- The 814 segment, whose CT is not public.
- The mapping from config dataset names to bucket dirs, which is inferred.
- What causes the one-cell shift or the (+30.7, -49.3) shift.
- Whether the ignored `_v2` label assets differ in content from the ones the loader reads.
- Segments with one region rest on matching their group's value, not on two regions agreeing.
- Areas outside the supervision masks, and the 61 folders under `ink/unused/` (metadata only).
- villa's loader end to end.

The listing is the bucket as of 2026-10-30. A rerun of the listing lists the bucket as it is then.

## Reproduce

Python 3.14 with numpy, scipy, httpx and numcodecs, plus Pillow for the overlay PNGs (the audit used 3.14.6, numpy 2.4.6, scipy 1.18.0, httpx 0.28.1, numcodecs 0.16.6, Pillow 13.0.0). All data comes from public URLs by HTTP range reads. Downloads are cached under `HF65_DATA` (default `./hf65_data`, ignored by git) and every real download is appended to `$HF65_DATA/ledger.tsv` with its byte count, so a rerun against a filled cache downloads nothing. Run from the repo root:

```
export HF65_DATA=$PWD/hf65_data
mkdir -p "$HF65_DATA"
```

One region, the w018 control (about 35 MB on an empty cache: 14.3 MB for `pyr`, 2.7 MB for `regions`, 18.0 MB for `reg`). `HF65_RESULTS` sends the outputs to a new folder, so the committed `results/phase2/` stays as it is:

```
export HF65_RESULTS=$PWD/rerun_phase2
python -I audit/code/reg2.py pyr c018
python -I audit/code/reg2.py regions c018 4
python -I audit/code/reg2.py reg c018 0      # prints level-0 (dy,dx)=(-19.7,-19.7)
```

All regions (about 1.1 GB on an empty cache; one registration step took 3.9 to 35.9 MB in the audit). `batch_all.sh` runs `pyr`, `regions` and `reg` for each key, skips steps whose output exists, and stops once the ledger passes `LIMIT_MB`:

```
export LIMIT_MB=1300
B="bash audit/code/batch_all.sh"
$B 0 c018 p4w00 p4w02:pred p4w03:pred p4w06:pred p4w07:pred p4w09:pred b0009 c013 c023 c028 c029v c031 t016 t017 t028 t029 t030 t035 t039 t040 t041 t043 t044 p500 n841 n841a n841b man5 t040b
$B 1 b0009 c013 c028 t016 t040 p4w02:pred p4w03:pred p4w06:pred p4w07:pred p4w09:pred
```

Without `HF65_RESULTS` these commands skip every key, because the results are already in `results/phase2/`.

Listing and phase 1 (listing 0.5 MB, phase 1 and the canvas `.zattrs` about 1.2 MB). These rewrite the files in `audit/results/`, so `git diff` shows what changed in the bucket since 2026-10-30:

```
python -I audit/code/list_bucket.py
python -I audit/code/phase1.py
python -I audit/code/zattrs.py
keys() { python -I -c "import json, sys; print(' '.join(json.load(open(sys.argv[1]))))" "$1"; }
python -I audit/code/volinfo.py $(keys audit/results/volinfo.json)
python -I audit/code/volmeta.py $(keys audit/results/volmeta.json)
```

Flags d, e and f need villa's `segment.py` at 37d1897fb, which is not copied into this repo. Pass a villa checkout that contains that commit, or the file extracted from it. The script checks its sha256 (`7d8466baaa6057907ad1fd20d4620079e30130a5885aa7c2fa6e336fcff1d849`):

```
git clone https://github.com/ScrollPrize/villa.git
python -I audit/code/labels_check.py villa
# or
git -C villa show 37d1897fb:vesuvius/src/vesuvius/ink_detection/data/segment.py > segment_37d1897fb.py
python -I audit/code/labels_check.py segment_37d1897fb.py
```

Tables (no downloads):

```
python -I audit/code/build_table.py      # results/segments.csv, results/summary.json
python -I audit/code/paris4_check.py     # results/phase2/paris4_check.json
python -I audit/code/paris4_max.py       # results/phase2/paris4_maxoffset.json
python -I audit/code/report_tables.py > audit/results/report_tables.md
```

The independent check's commands are in `independent/NOTES.md`; with `HF65_RESULTS` set they write to that folder instead of `independent/out/`.

Rebuilding `segments.csv`, `summary.json`, `labels_check.json` and the two Paris4 files from the committed inputs gives the committed files. A rebuilt `report_tables.md` lacks the provenance note at its top, and its byte totals at the end come from your own ledgers.

## Files

- `code/targets.json`: phase-2 keys, each with segment folder, region-picking mask and CT volume.
- `results/listing.jsonl`: the bucket listing.
- `results/phase1.json`: per-folder metadata and the validator reproduction.
- `results/canvas_zattrs.json`, `results/volinfo.json`, `results/volmeta.json`: canvas attributes and CT volume metadata used to choose the volume for each key.
- `results/labels_check.json`: flags d, e and f.
- `results/segments.csv`, `results/summary.json`, `results/report_tables.md`: the tables.
- `independent/`: the independent check (scripts, notes, per-region rows, overlay PNGs).
- `results/phase2/`: `pyr_<key>.json` (pyramid model), `regions_<key>.json`, `reg_<key>.jsonl` (one row per registered region), the Paris4 fits and the batch logs. File paths in the log tracebacks were shortened to `audit/code/...`, and their line numbers refer to the scripts before the path handling was changed for this repo.

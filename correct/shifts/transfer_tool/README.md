# --label-canvas-offset values for the 26 offset segments

`offsets_transfer_frame.csv` has one row per row of `../shifts.csv`. Each row gives the `--label-canvas-offset DY DX` to pass to the existing `transfer single` of villa's `vesuvius/src/vesuvius/tifxyz_label_transfer`, with the segment's own tifxyz as both source and target and the segment's ink label as `--label`. The values are the `../shifts.csv` model converted into the tool's frame. No row needs more than one constant: the largest gap between the constant and the full model anywhere on the canvas is 0.42 px, on phercparis4 w00.

| file | what it is |
|---|---|
| `offsets_transfer_frame.csv` | the table, 26 rows |
| `table.py` | rebuilds the table and `table_checks.json` |
| `table_checks.json` | per row and axis: the corner, dense grid and kernel checks described below, and the input consistency checks |
| `villa_boot.py` | loads villa's `tifxyz_label_transfer` and a prebuilt `vc_reader` from source trees, used by `table.py` and `run_transfer.py` |
| `meta/` | the 26 meta.json files and `fetch_check.json`, written by `fetch_meta.py` |
| `pixel_notes.json` | the `checked_on_pixels` column, written by `make_pixel_notes.py` from `w00/` and `w01/` |
| `run_transfer.py` | runs villa's `transfer` CLI from a source tree |
| `w00/`, `w01/` | the pixel checks below: `compare_agreement.json`, `compare_shifts.json`, `compare_shifts_iou.json` and the scripts that made them |

## Conversion formula

Notation per axis: L is the label size, which equals the canvas size on all 26 segments. S is the tifxyz stored grid size and s is the meta.json scale for that axis. meta.json lists the scale as x then y. F = full_extent(S, s) is the tool's full-resolution size. d(y, x) is the `../shifts.csv` offset at label pixel (y, x).

The tool reads label pixel floor(p * L / S - o) for a point at stored grid position p, where o is the flag value times L / F. The `../shifts.csv` model says that label pixel y belongs at full-resolution cell y + d, which is stored position (y + d) * s. Setting the two equal gives the offset the tool needs at label pixel y:

    o(y, x) = (s * L / S) * (y + d(y, x)) - y - 0.5      in label px
    flag    = o * F / L                                  in full-resolution px

The same formula with x in place of y gives DX. The field o is affine in (y, x), so over the canvas its extremes are at the four corners. The constant with the smallest worst-case error is the midpoint of the corner range, which is also the value at the canvas centre. The gap columns are half that range, converted to full-resolution px. `table.py` checks the corner argument against a dense grid and against a direct pass through the kernel equations over every output pixel. All three agree to within 2e-11 px.

## The half-pixel convention

The tool and the full_3d loader place a full-resolution pixel at different points on the surface. In the tool, output pixel i has its centre at stored position (i + 0.5) * S / F. In the loader and in the `../shifts.csv` convention, full-resolution cell i is stored position i * s. Since S / F is close to s, a tool output pixel i holds content that belongs at loader cell i + 0.5, give or take the rounding inside F. On w01 that lag runs from +0.50 at row 0 to +0.11 at the last row, because S / s is 65697.6 while F is 65698. Across columns it stays at +0.50.

The DY and DX columns are exact in the tool's own frame, which is what the formula above asks for. If the transferred label is going to be read by the full_3d loader, the loader_DY and loader_DX columns put the content of loader cell i into output pixel i instead. They come from the same derivation with the target moved to cell i:

    o_loader(y, x) = (y + d(y, x) + 0.5) * L / N - y - 0.5

Here N is the output size the tool picks, which equals F on all 26 segments. The loader columns are larger than the tool columns by 0.5 px on the constant rows and by 0.27 to 0.73 px on the Paris4 rows. On Paris4 the rounding inside F adds or removes up to a quarter pixel over half the canvas.

Two consequences:

- On the 18 constant rows, loader_DY and loader_DX equal `../shifts.csv` dy_a and dx_a, and DY and DX are dy_a - 0.5 and dx_a - 0.5 to within 0.0002 px. So a constant row's (dy_a, dx_a) is the right flag value in the loader frame and half a pixel off in the tool's frame. `../README.md` was corrected on 2026-11-13 and its section "Related tool in villa" now states the same thing as this file.
- On the five Paris4 rows whose slopes come from the metadata, w02, w03, w06, w07 and w09, the loader-frame field is constant to within 0.002 px, because their slope is exactly F / L - 1. Their tool-frame gap of up to 0.23 px comes only from the rounding inside F.

## Results in short

- 18 constant rows: gap 0.000 px on both axes in either frame.
- Paris4, 8 rows: tool-frame gap 0.000 to 0.42 px, loader-frame gap 0.000 to 0.47 px. The three rows with fitted slopes, w00, w01 and w05, carry the largest gaps in both frames, since a fitted slope does not match the size ratio exactly.
- No row has a gap over 1 px, so none is flagged.
- The tool's output size equals full_extent on all 26 segments. The size rule in `infer_output_shape` snaps to full_extent whenever the label is within 0.5% of it, and the largest Paris4 mismatch is 0.42%, on w01 rows.

## Where each number comes from

From metadata and the `../shifts.csv` fits only, with no pixels involved:

- Every DY, DX, loader_DY, loader_DX and gap value in the CSV.
- Label and canvas sizes and the stored grid sizes come from the audit's `../../../audit/results/phase1.json`, which read the canvas zarr headers and the x.tif headers. The loader's ink label zarr has the canvas size on all 26 rows.
- Scales come from meta.json fetched live through this repo's own fetch code (`../../../audit/code/common.py`, called by `fetch_meta.py`), 26 files and 12,811 bytes in all. All 26 match `phase1.json` exactly (`meta/fetch_check.json`).
- full_extent and the tool's output size come from the tool's own functions, `vc_reader.tifxyz.full_extent` and `core.infer_output_shape`, called unchanged.

Checked on pixels:

- phercparis4 w00: full end-to-end run on local data, this row's values. Numbers below, files in `w00/`.
- phercparis4 w01: an earlier end-to-end run with the affine3 refit constant 11.3181 11.1463. The w01 row in the CSV uses the `../shifts.csv` model instead, which is the published readme fit, so its value 11.1442 11.0693 was not itself run. The w01 run checked the same formula with a different model. Files in `w01/`.

## w00 end-to-end check

Inputs were the segment's tifxyz and ink label TIFF, from a local copy made during earlier label work. Its meta.json is byte-identical to the live one. A check made during that work found the label TIFF equal to the loader's level 0 label zarr on every listed chunk. Three `transfer single` runs used the same UV cache: DY DX from the table, loader_DY loader_DX from the table, and no offset. All three produced a 32281 x 51345 output.

The reference is the `../shifts.csv` w00 row, which is the apply_fit "readme" w00 model unchanged, applied with nearest-neighbour sampling. The comparison uses a closed-form copy of apply_fit.resample. On three 256-row blocks it gave zero differing pixels against the original. The compared region is the supervision mask moved by the same model, 95,963,921 pixels, and the tool marked every one of them valid in all three runs.

Each cell gives the share of supervised pixels with the same label, then the ink IoU:

| run | model on the tool grid | model on the loader grid |
|---|---|---|
| DY DX = -21.3771 11.0061 | 99.980%, 0.9991 | 99.875%, 0.9945 |
| loader_DY loader_DX = -20.6889 11.5579 | 99.872%, 0.9944 | 99.986%, 0.9994 |
| no offset | not computed | 97.216%, 0.8851 |
| shipped label read one-to-one | not computed | 98.611%, 0.9409 |

Each offset run agrees best with the model sampled on its own frame's grid. That is the half-pixel finding again, now on a second segment.

Residual shifts in five 2048 px windows spread over the inked area:

- Loader-frame run against the model on the loader grid: rows within 0.09 px of zero. Columns read -0.23 and -0.47 px in the two left-hand windows, where the loader-frame constant is predicted to be off by -0.37 and -0.41 px. Measured minus predicted is at most 0.15 px.
- Tool-frame run against the model on the tool grid, taken as the difference between the shifts the two show against the loader-grid model: at most 0.29 px by cross-correlation and 0.49 px by ink overlap, against a predicted maximum of 0.37 px. On w01 both estimators showed about 0.4 px of error on nearest-neighbour labels, so this is as close as they can resolve.
- No-offset run: -19.3 to -20.7 px in rows and +10.7 to +11.7 px in columns. The prediction, which is close to the model's intercept, is -20.7 px in rows and +11.2 to +11.7 px in columns.

Cost on 4 cores: the first run, which also built the UV cache, took 2 min 24 s and peaked at 7.2 GiB. The other two took 1 min 27 s and 1 min 3 s at 5.4 GiB. The comparison took 2 min 8 s at 6.1 GiB.

## How to rerun

The table needs only this repo, a villa source tree and a vc_reader build. Run from this folder:

```
python -I table.py VILLA_SRC --vc-reader VC_READER --out OUT_DIR
cmp OUT_DIR/offsets_transfer_frame.csv offsets_transfer_frame.csv
cmp OUT_DIR/table_checks.json table_checks.json
```

VILLA_SRC is villa's `vesuvius/src`; the files here were reproduced at villa commit 0080fa393 with Python 3.14 and numpy 2.5.2. VC_READER is villa's `vc_reader` folder after a scikit-build build, so that it holds `src/vc_reader` and the compiled extensions in `build/cp314-cp314-<platform>` (pass `--vc-reader-build DIR` if the build folder is elsewhere). Leave out `--vc-reader` if vc_reader is installed in the environment. Without `--out` the two files here are overwritten.

The inputs of the table can be rebuilt too. These overwrite the files in this folder, so `git diff` shows any change:

```
python -I fetch_meta.py
python -I make_pixel_notes.py
```

`fetch_meta.py` downloads the 26 meta.json files (about 13 kB) and needs httpx. Its cache goes to `$HF65_DATA`, or `hf65_data/` here if that is unset.

The pixel checks need the segment folder (its tifxyz plus the ink label and supervision mask TIFFs named in `w00/compare_lib.py` and `w01/compare_lib.py`) and a Linux shell with `taskset` and GNU time:

```
PY=python VILLA_SRC=... VC_READER=... W00_DIR=... RUN_DIR=... bash w00/run_w00.sh
PY=python VILLA_SRC=... VC_READER=... W01_DIR=... RUN_DIR=... bash w01/run_w01.sh
```

Each script runs `transfer single` through `run_transfer.py`, then `compare.py`, `cc.py` and `cc2.py`. The transfer outputs, the UV cache and the three `compare_*.json` files go to RUN_DIR, so the published files in `w00/` and `w01/` are not touched.

## Not verified

- No pixel check on the other 24 segments. 1667 w018 also has its inputs locally and was not run. Its local label TIFF was only compared with a label zarr at the coarse levels. The rest are not local and were not downloaded.
- The loader-frame values were run on w00 only.
- Whether the `../shifts.csv` models themselves are right is outside this table. `../README.md` rates them at about 1 px.
- The tool reads a label zarr by taking its middle slice. Passing the label TIFF avoids that question, and the w00 and w01 checks used the TIFFs.

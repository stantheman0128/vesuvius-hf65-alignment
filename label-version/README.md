# Which label version full_3d training picks in the PHerc1667 `_2um` folders (2026-10-31)

Scripts and per-folder outputs behind my comment on [villa PR #2140](https://github.com/ScrollPrize/villa/pull/2140#issuecomment-6534603173).

With `dino_guided_teacher.json` as shipped, which sets no `label_version`, villa main at `9a92af33e` picks
`<folder>_inklabels.zarr` and `<folder>_supervision_mask.zarr` in each of the six `ink/1667/*_2um` training folders
and no validation mask. Five of the six folders also hold v2 labels (`*_inklabels_v2`, `*_supervision_mask_v2`,
`*_validation_mask_v2`), which go unused. This folder measures how much of the v2 labelling that leaves out, and
what it does to the training patches of w029.

## Results

Middle plane at level 5. "Picked" is what villa selects; "v2" is the folder's own v2 file. In every folder the
picked `_2um` mask is pixel-identical to the folder's unversioned v1 mask, so the files are not damaged. They are
the older label set.

| folder | picked supervision as a share of v2 | picked ink as a share of v2 | picked supervision inside `validation_mask_v2` (px) |
|---|---|---|---|
| w013 | no v2 labels | no v2 labels | no v2 labels |
| w018 | 51.0% | 80.3% | 15,907 of 125,426 |
| w023 | 21.4% | 35.1% | 0 of 19,989 |
| w028 | 19.3% | 30.1% | 0 of 3,768 |
| w029 | 5.8% | 9.4% | 1,446 of 1,446 |
| w031 | 43.6% | 59.1% | 2,906 of 21,834 |

All of w029's picked supervision lies inside its v2 validation area, so with the teacher config w029 trains only on
the area that v2 sets aside for validation.

w029's patches from villa's `find_segment_patches` with the teacher settings, on the level-4 middle plane:

| labels | training patches | validation patches |
|---|---|---|
| as picked (`_2um`, v1) | 82 | 0 |
| v2 supervision and ink, with `validation_mask_v2` | 656 | 454 |
| v2 supervision and ink, no validation mask | 822 | 0 |

On villa main at `c0e3b19dc7` (2026-10-31), `segment.py` and `dino_guided_teacher.json` are unchanged since
`9a92af33e`. `patch_finding_default.py` changed once since then (villa#2197), and `patches.py` pointed at the new file
prints the same three lines as `out/patches_w029.jsonl`.

## Rerun

Python 3 with numpy, numcodecs and httpx. Every read goes to the public bucket `hf://buckets/scrollprize/datasets`,
about 0.55 MB in all, and `scripts/hfio.py` stops at 200 MB. Run from this folder:

```
export W029_DL=$PWD/dl W029_OUT=$PWD/out
mkdir -p dl out/arr
python -I scripts/step_list.py
python -I scripts/selection.py out/listing.json out/selection.json
bash scripts/run_w029.sh && bash scripts/run_others.sh && bash scripts/run_val.sh
python -I scripts/compare.py out/arr 4 > out/compare_L4.json
python -I scripts/compare.py out/arr 5 > out/compare_L5.json
python -I scripts/crossseg.py out/arr
python -I scripts/patches.py out/arr/w029_2um_supervision_mask_L4.npz out/arr/w029_2um_inklabels_L4.npz - >> out/patches_w029.jsonl
python -I scripts/patches.py out/arr/w029_supervision_mask_v2_L4.npz out/arr/w029_inklabels_v2_L4.npz out/arr/w029_validation_mask_v2_L4.npz >> out/patches_w029.jsonl
python -I scripts/patches.py out/arr/w029_supervision_mask_v2_L4.npz out/arr/w029_inklabels_v2_L4.npz - >> out/patches_w029.jsonl
```

The `run_*.sh` scripts take the folder and the interpreter from `W` and `PY` (defaults: this folder and `python3`).
On 2026-10-31 they ran with local paths in their place, which is the only change from the versions that wrote the
outputs here. `run_others.sh` prints two Python tracebacks for w013, which has no v2 zarrs; they are expected, and no
array is written for them.

Running the commands above in a new folder on 2026-11-01 read 0.552 MB and gave the same `listing.json`,
`selection.json`, `readlevel.jsonl`, `compare_L4.json`, `compare_L5.json`, `crossseg_L5.json` and `patches_w029.jsonl`,
byte for byte.

## Files

- `scripts/step_list.py`: lists the six folders and every label zarr of w029, and fetches the zarr metadata.
- `scripts/selection.py`: runs villa's own `discover_segment_labels`, taken out of `villa_segment_9a92af33e.py` with
  `ast`, over that listing with the teacher settings (no `label_version`, labelled segments required).
- `scripts/readlevel.py`, `run_w029.sh`, `run_others.sh`, `run_val.sh`: read one pyramid level of a label zarr chunk
  by chunk and keep the middle plane.
- `scripts/compare.py`: pixel relations between the seven w029 label arrays at one level.
- `scripts/crossseg.py`: the per-folder table above.
- `scripts/patches.py`: runs villa's `find_segment_patches`, taken out of
  `villa_patch_finding_default_9a92af33e.py` with `ast`, with patch size 256, overlap 0.5, minimum labelled coverage
  0.02 and patch finding scale 4 from the teacher config.
- `scripts/villa_segment_9a92af33e.py` and `scripts/villa_patch_finding_default_9a92af33e.py`: unchanged copies of
  villa's `segment.py` and `patch_finding_default.py` at `9a92af33e` (MIT license, copyright Vesuvius Challenge).
- `out/`: the outputs of the commands above. `crossseg_and_patches.txt` is the printed output of `crossseg.py` and
  `patches.py`. The level-4 and level-5 arrays (`out/arr/*.npz`) are not included; `run_*.sh` rebuilds them.

## Limits

- `dino_guided_teacher.json` asks for "the corresponding historical local label root", so training on v1 may be
  intended. Whether the teacher should pick v2 is the config owner's call.
- The patch counts leave out patches that `_native_sample` drops at load time and the patch cache.
- Canvas fractions include the empty canvas around the surface. The TIFF copies in the folders were not compared
  with the zarrs.

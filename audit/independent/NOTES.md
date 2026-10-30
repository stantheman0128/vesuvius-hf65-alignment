# Independent check of the audit's phase 2 (2026-10-30)

The verifier's notes, copied into this repo. Only file paths and the run commands were changed to this repo's layout. The verifier changed nothing in `audit/code/` or `audit/results/`; its files are the ones in `audit/independent/`.

## Method (independent of reg2.py)

- Reads: `common.fetch` through `scripts/vio.py`, which logs my own cache misses to `$HF65_DATA/independent_downloads.tsv`. `lib.Image` is reused only to decode canvas and label zarr chunks.
- `scripts/vcore.py`, all my own code:
  - ByteStore: assembles any byte range from cached ledger ranges and fetches only the gaps.
  - TIFF strip reader for x/y/z.tif. Bilinear tifxyz interpolation at full-res cells, villa convention (stored = cell * scale, no half offset).
  - CT trilinear sampling at level 2, with a mean-pyramid centre (level-2 voxel i sits at level-0 4i + 1.5). The strided convention was also run on the controls, see below.
  - Registration: phase-only correlation on the Gaussian gradient magnitude, Hann-windowed and masked. Sub-pixel by a matrix-DFT upsampled surface (1/20 lattice px). Second readout: raw-intensity NCC after shifting.
  - Self-test: shifts (+7.3, -4.6), (+0.35, +0.35), (-19.6, -19.6) and 0 recovered to within 0.05 px.
- Canvas pyramid model, my own: phase correlation of each level-L slice against a block mean of level-0 slice 32, at the audit's cached `pyr` block. It gives cL = level-0 position of level-L pixel centres. Results: c013 (-0.5, -0.55); p500 (+0.5, +0.5); man5 (-0.1, +0.2); 0139 (0.0, +2.7 to +2.8); Paris4 (-1.8, +1.6).
- Sign: d = (dy, dx) in level-0 px. Canvas content at lattice y is found at y + d in CT-at-tifxyz. This is the same sign as the audit.
- Quadrants: the same registration on each 128 (L2) or 256 (L1) lattice quadrant of the region.

## Commands

```
export HF65_DATA=$PWD/hf65_data                # from the repo root, as in audit/README.md
V="python -I audit/independent/scripts"        # then: $V/vtest.py and so on
$V/vtest.py                                   # self-test
$V/vmeasure.py <tag> <key> <y0> <x0> 1024 <L> [--vol P] [--mesh P] [--strided-ct]
#   p500_r0 p500 14336 5632 L1; man5_r0 man5 7168 12288 L2; c013_r0 16896 11776 L1; c013_r1 3072 10752 L1;
#   c013_r2 28672 17920 L1 (new region); t035_r0 3584 5632 L2; t035_far 19200 18432 L2 (new region, picked by vfar.py);
#   t040_r0 2560 26112 L2; t040_r1 22016 11264 L2; p4w07_r0 17920 45056 L2; p4w07_r1 7680 99328 L2;
#   t040_r0_march: t040 r0 with --vol PHerc0139/volumes/20260319133554-2.403um-0.2m-77keV-masked.zarr
#       --mesh PHerc0139/segments/20250831000000-w040_2025083102/mesh/20250831000000-on-20260319133554-2.403um.tifxyz
#   p500_r0_strct, man5_r0_strct: --strided-ct
$V/vfar.py t035 3584 5632                     # far fully valid block on t035
$V/vw040s3.py; $V/vs3canvas.py                # S3 meshes, surface volumes, x.tif bytes for w040/w044
$V/vs3render.py t040_r0 PHerc0139/segments/20250831000000-w040_2025083102/surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr 50,54,58
$V/vpng.py <tag> <png>
```

Raw rows: `out/results.jsonl`. Arrays: `out/<tag>.npz` (not committed; a rerun writes them). Set `HF65_RESULTS` to send a rerun's rows, arrays and PNGs to another folder.

## Results

| tag | region (y0, x0), L | my d (level-0 px) | audit d | phase-corr peak / 2nd | raw NCC at 0 / at d | quadrants (dy, dx) |
|---|---|---|---|---|---|---|
| p500_r0 | 14336, 5632, L1 | +0.4, +0.4 | +0.49, +0.52 | 0.390 / 0.018 | 0.974 / 0.975 | all +0.4, +0.4 to +0.6 |
| man5_r0 | 7168, 12288, L2 | +0.4, +0.2 | +0.37, +0.41 | 0.688 / 0.026 | 0.961 / 0.961 | +0.4, 0.0 to +0.4 |
| c013_r0 | 16896, 11776, L1 | -19.7, -19.6 | -19.70, -19.69 | 0.252 / 0.016 | 0.538 / 0.960 | -19.6 to -19.8, -19.4 to -19.6 |
| c013_r1 | 3072, 10752, L1 | -19.7, -19.5 | -19.65, -19.66 | 0.379 / 0.014 | 0.391 / 0.959 | -19.6 to -19.8, -19.4 to -19.6 |
| c013_r2 (new) | 28672, 17920, L1 | -19.6, -19.6 | not measured | 0.354 / 0.014 | 0.451 / 0.965 | -19.4 to -19.6 |
| t035_r0 | 3584, 5632, L2 | +31.0, -48.8 | +30.84, -49.12 | 0.465 / 0.027 | 0.226 / 0.964 | +30.4 to +30.8, -48.8 to -49.6 |
| t035_far (new) | 19200, 18432, L2 | +30.6, -49.2 | not measured | 0.427 / 0.018 | 0.278 / 0.955 | +30.4 to +30.8, -49.2 |
| t040_r0 | 2560, 26112, L2 | +30.6, -49.0 | +30.69, -49.28 | 0.447 / 0.021 | 0.254 / 0.952 | +30.4 to +30.8, -48.8 to -49.2 |
| t040_r1 | 22016, 11264, L2 | +30.6, -49.0 | +30.63, -49.32 | 0.391 / 0.025 | 0.270 / 0.963 | +30.4 to +30.8, -48.8 to -49.2 |
| p4w07_r0 | 17920, 45056, L2 | -26.8, +3.4 | -27.18, +3.45 | 0.552 / 0.026 | 0.507 / 0.968 | -26.4 to -27.2, +3.2 to +3.6 |
| p4w07_r1 | 7680, 99328, L2 | -35.2, -20.0 | -35.50, -20.00 | 0.642 / 0.017 | 0.283 / 0.969 | -34.8 to -35.6, -20.0 |
| p500_r0_strct | strided CT pyramid | -1.2, +1.9 | n/a | 0.286 / 0.019 | 0.924 / 0.937 | |
| man5_r0_strct | strided CT pyramid | -0.4, +2.0 | n/a | 0.332 / 0.021 | 0.889 / 0.908 | |
| t040_r0_march | March volume + March mesh | +42.0, +54.8 | n/a | 0.162 / 0.024 | 0.122 / 0.812 | +40.4 to +42.4, +54.8 to +55.2 |

Peak-to-second ratio is 14 or more on every row.

- c013: r1 to r2 are about 25,600 px apart and agree within 0.1 px. t035: r0 to far are about 20,400 px apart and agree within 0.4 px. t040: r0 to r1 are about 24,000 px apart and agree to 0.0 px.
- Paris4 w07: the audit's model d = a + (full/canvas - 1) * pos, with a = (-42.2, +23.2), predicts (-27.1, +3.5) at r0 and (-35.5, -20.1) at r1. Mine agree within 0.4 px. The two regions differ by 8.4 and 23.4 px, so the offset does depend on position. "Up to 42 px" is the extrapolated intercept, not a measured value; the largest I measured was 35.2.
- CT pyramid convention: with the strided convention the controls move to (-1.2, +1.9) and (-0.4, +2.0), and every score drops. The mean convention fits better, and with it the controls come out at +0.4.
- The audit's "lower NCC on the 0139 second-shift group" does not reproduce as a property of the data. My phase-corr peaks are 0.39 to 0.47 for 0139, compared with 0.25 to 0.38 for c013 and 0.55 to 0.69 for p4w07 and man5. Raw NCC at d is 0.952 to 0.964 for 0139, compared with 0.959 to 0.975 elsewhere.

## 0139 frame check

- The bucket `ink/0139/w040_2025083102/meta.json` bbox equals the S3 `mesh/...-on-20260102150214-2.399um.tifxyz` bbox to within 0.001. Its z reaches 29740. Both 2.403um 0139 volumes (20260319133554: z 19368; 20260319133050: z 15137) are shorter than that, and the S3 on-20260319133554 mesh has a different bbox (z -2171 to 20308). Bucket x.tif dims are 1244x1552, the same as the S3 2.399 mesh. Row 620 matches with a max abs diff of 0.0039. No meta.json carries a volume field.
- The canvas against the March volume sampled at the March mesh cells gives d = (+42.0, +54.8), peak 0.162, raw NCC 0.812. This is worse on every score than against 20260102150214 (peak 0.447, NCC 0.952). The March frame does not explain the shift.
- The S3 official render `surface-volumes/2.399um-0.22m-78keV-volume-20260102150214.zarr`, slice 54, has the same size as the bucket canvas. After the lattice-centre correction it lines up with my CT-at-tifxyz to within about (-0.5, -0.6) px, with peak 0.450. The bucket canvas sits about (+29.9, -49.6) from that render, with peak 0.769. So the S3 render agrees with the tifxyz and the bucket canvas is the odd one out.

## PNGs

`control_p500_r0.png`, `control_man5_r0.png`, `onecell_c013_r1.png`, `second_t040_r0.png`, `paris4_p4w07_r1.png`. Panels: canvas; CT sampled at tifxyz cells; red/cyan overlay unshifted; red/cyan overlay shifted by d. The yellow line is the inklabels zarr contour on the canvas grid. For man5, the label zarr is the 24061x29841 one (flag a), so its contour is not reliable there.

## Bytes

My cache misses: 22,092,543 bytes in 4,127 responses, on top of the audit's cache. The shared ledger total went from 987,287,351 bytes (67,002 lines) at my start to 1,152,044,667 (76,671 lines). Most of that increase is the concurrent batch_all.sh job, so the ledger delta is not mine.

## Not checked

- Segments other than c013, t035, t040, p4w07, p500 and man5.
- The villa full_3d loader's actual cell convention (assumed stored = cell * scale).
- The CT pyramid convention, checked only indirectly through fit quality.
- The S3 w013 render, which is 41600x79600 and does not match the bucket canvas size, so it was not compared.
- The cause of the (+30.7, -49.3) shift.

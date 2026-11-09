# Before/after figure for result A

`beforeafter_s29.png` shows held-out boxes 5, 17 and 19 of result A on Paris4 w01, numbered as in `../regions.json`, one row per box. From left to right, each row has the CT at the surface as the models read it, the corrected labels (C0), the labels as shipped (AS), the ink probability from the AS model and the ink probability from the C0 model. Both models are seed 29 at step 12000 with EMA weights, the same checkpoints that `../RESULTS.md` scores. `box5_s29.png`, `box17_s29.png` and `box19_s29.png` are the same rows at full resolution.

In all three boxes the C0 model draws strokes that follow the letters. The Π in box 17 is the clearest, then the ω in box 19 and the crossbar of the θ in box 5. In the same places the AS model's map is faint and blurred, and the letter shapes are hard to make out.

Cyan outlines the held-out box that A scored, placed with the corrected labels and padded by the scoring margin. Orange outlines the same letters where the labels as shipped put them. In the label panels white is ink, black is supervised background and grey is unsupervised. Dark blue is outside the canvas or has no prediction.

## The AUC in the titles

Each row title gives a 2D AUC against the C0 labels inside the scored area. This number only checks the figure's readout. A's metric in `../RESULTS.md` scores 3D voxels, while the figure reads each 3D prediction at the voxels that villa's label projection uses. C0 is ahead in every box: 0.855 against 0.765 in box 5, 0.873 against 0.812 in box 17 and 0.830 against 0.788 in box 19.

## How it was made

`render_beforeafter.py` calls villa for the config, dataset, model and forward pass the way A's `scripts/eval_heldout.py` does, and it does not reimplement the model, the normalization or the geometry. It runs every patch of villa's patch lattice over a window that holds both placements of a box. Overlapping patches are blended with villa's Gaussian weights, and each flat pixel takes the prediction at the voxel that villa's label projection marks for it. The docstring at the top of the script has the details.

`provenance_s29.json` records these checks:

- The labels in the figure match the label zarrs that the trainer and both evaluations read (agreement 1.0 in every box).
- Villa's 3D labels, read back through the same projection, land on the flat C0 labels (hit fraction 1.0 in every box), so the readout is aligned.
- Every scored pixel has a prediction from both models.

The run covered 685 patches and none failed. Villa's sampler rejected 232 of them as inadmissible, and the accepted patches still cover every scored pixel.

## Rerun

```
python -I render_beforeafter.py VILLA_SRC OUT_DIR --seed 29 --work A_WORK_TREE
```

VILLA_SRC is villa's `vesuvius/src` at 37d1897fb. The script stops on another commit unless it gets `--allow-other-villa`. A_WORK_TREE holds `runs/`, `labels/`, `data/`, `scripts/` and `regions.json`, and the checkpoints in `runs/` are not in this repository. `--plan-only` runs on the CPU without checkpoints and draws the figure without model predictions, which makes a cheap first check. `run_s29.sh` is the launcher that made these files: box 5 alone first, then all three boxes. On one RTX 3070 Ti the full run took 152 s and peaked at 2.93 GiB of GPU memory.

`figlib.py` holds the numpy parts of the script and `test_figlib.py` tests them on synthetic arrays. `check_villa_api.py` reads villa's source at the commit and checks that every name the script uses exists and accepts the arguments it passes. `eval_config_C0_s29.json` is A's C0 seed 29 evaluation config with only `out_dir` changed. Home directories in paths are written as `~`.

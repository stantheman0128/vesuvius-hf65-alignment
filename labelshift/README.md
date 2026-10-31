# Label offset and held-out ink detection in villa's full_3d trainer (2026-10-31)

The pre-registered gate G0 (C0 seed 27 pooled held-out AUC of at least 0.70) failed in all three attempts: attempt 0, L1 and L2. Amendments F and G were written after those failures, so everything below that compares arms is exploratory. Segment w018 did not learn in any run (AUC near 0.5), which leaves w00 as the only segment where a label effect could show.

## Question

The HF65 labels in `scrollprize/datasets` sit in the canvas frame, offset from the tifxyz cells that `full_3d` training reads (this repository's top-level README and `correct/`). Does training on offset labels cost held-out ink detection, and how do the labels as shipped compare with corrected labels?

## Setup

- Trainer: villa at 37d1897fb, `full_3d` mode, config derived from `dino_guided_teacher.json` by `scripts/mkconfig.py`. Changes from the teacher are listed in `PREREG.md` section 4. The main ones: two segments, CT read at level 2, 128^3 patches.
- Segments, both in training: w00 (`ink/phercparis4/w00_20231016151002`) and w018 (`ink/1667/w018_20240304144031_2um`).
- Arms:
  - C0: corrected labels, resampled into the tifxyz frame with `correct/apply_fit.py` (model "readme").
  - AS: labels as shipped, cell (y, x) carries canvas pixel (y, x). On w00 this is -1.6 to +17.4 px from C0 in y and -10.0 to +14.5 px in x, depending on position.
  - C8y: C0 moved 32 level-0 px in y (8 voxels at the training level).
- Seeds 27 and 28 for each arm, six runs.
- 12,000 steps with the anneal schedule (`MAXSTEPS_MODE=anneal`: the cosine learning rate ends at the last step).
- Held-out boxes: three labelled components per segment (`regions.json`), padded by 64 px. Supervision within 512 px of a box is removed from training in every arm, and inside the boxes every arm carries C0 labels.
- Scoring: `scripts/eval_heldout.py` runs the EMA weights of the final checkpoint on every held-out patch and compares them with the C0 labels where the validation mask projects.
- Primary readout: held-out voxel ROC AUC on w00. Pooled and w018 AUCs are reported as secondary.
- Hardware: one RTX 5090, 71 to 76 minutes per run including evaluation.

## Results

Held-out AUC per run (`results/F/results_F.md`, `results/F/results_G.md`):

| run | w00 | w018 | pooled |
|---|---|---|---|
| C0 s27 | 0.7425 | 0.5307 | 0.6550 |
| C0 s28 | 0.7384 | 0.5134 | 0.6443 |
| AS s27 | 0.7516 | 0.5079 | 0.6530 |
| AS s28 | 0.7510 | 0.5129 | 0.6521 |
| C8y s27 | 0.7510 | 0.5172 | 0.6551 |
| C8y s28 | 0.7361 | 0.5357 | 0.6541 |

Mean w00 AUC over the two seeds: C0 0.7405, AS 0.7513, C8y 0.7436.

### Amendment F

Amendment F waives G0 and applies the PREREG noise rule to the w00 AUC. sigma = sqrt(mean over arms of (s1 - s2)^2 / 2) = 0.0063, and delta = max(2 sigma, 0.01) = 0.0126. C8y differs from C0 by 0.0031 and AS by 0.0108, both within delta, so the verdict is NO COST for C8y and NO COST for AS. Under Amendment F, AS is reported and does not decide COST. Read with the original PREREG section 7 rules, the same runs give "G0 failed, inconclusive" (`results/F/results_F_prereg_rules.md`).

### Amendment G

Amendment G reads each segment on its own with 95% Student t intervals from the seed noise (df 3, t = 3.182). d is m(C0) minus m(arm), so a negative d means the arm scored above C0.

On w00, C0's lower bound is 0.7263, above chance, so the segment can measure a difference.

| arm | d | 95% interval | upper end as share of headroom |
|---|---|---|---|
| AS | -0.0108 | -0.0308 to 0.0093 | 4% |
| C8y | -0.0031 | -0.0231 to 0.0170 | 7% |

Both intervals include zero, so no cost was detected. The upper ends are the largest costs the data allow: 0.0093 AUC for AS and 0.0170 for C8y. Headroom is m(C0) - 0.5 = 0.2405. For scale, doubling the training of C0 seed 27 from 6,000 to 12,000 steps gained 0.0483 on w00 (one run each).

On w018, C0's lower bound is 0.4983, so C0 is not distinguishable from chance there and the w018 differences get no verdict.

With two seeds per arm, the smallest w00 cost this design detects at 80% power is 0.0262 AUC (`results_G.md` lists 3 to 6 seeds).

### Correction to results_F.md

The NO COST line in `results_F.md` says "with a base model near 0.69 AUC". That text is fixed in `scripts/decide_f.py`, which was frozen when the L1 setting (w00 0.6943) was expected to be used. The runs used the L2 setting, and C0's w00 AUC is 0.7405 (mean of both seeds). decide_f.py is left unchanged because its hash is part of the record.

### The three G0 attempts

All three trained and scored C0 seed 27 only.

| attempt | steps | schedule | pooled | w00 | w018 | files |
|---|---|---|---|---|---|---|
| attempt 0 | 6,000 | teacher (lr still at 88% of peak at the end) | 0.6182 | 0.6901 | 0.5119 | `results/attempt0/` |
| L1 | 6,000 | anneal | 0.6231 | 0.6943 | 0.5156 | `results/L1/` |
| L2 | 12,000 | anneal | 0.6550 | 0.7425 | 0.5307 | `results/L2/` |

L2's C0 seed 27 run is the C0 seed 27 run of the comparison above (same files, copied into `results/F/runs/C0_s27`).

## Limits

- Exploratory. Amendment F was written after two G0 failures and Amendment G after three, with C0 seed 27 already known.
- One segment that learns (w00), two seeds per arm, and fixed held-out boxes. The intervals cover run to run variation in training, not the choice of boxes.
- A weak model, about 0.74 AUC on w00. A stronger model could be more sensitive to label position, so "no cost detected" holds for this model at this level.
- C8y shifts y only. AS's offset on w00 is two-axis and varies with position.
- C0 seed 27 was trained before a pod restart and the other five runs after it (`F_RESTART_NOTE.md`). Checked: the configs differ only in seed and output folder, both C0 seeds drew the same 2,478 patches (`sampling_audit` in the trainer log), and every install used the same pinned `scripts/pod_requirements.txt`. The package freeze before the restart was not recorded, so matching versions are inferred. The freeze after the restart is `results/F/pod_venv_freeze_after_restart.txt`.

## Files

- `PREREG.md`, `AMENDMENT-A-2026-10-31.md` to `AMENDMENT-G-2026-10-31.md`, `F_RESTART_NOTE.md`: the pre-registration and every change to it, as written at the time.
- `logs/*timestamp*.txt`, `logs/amendment_choice.txt`: save times and sha256 of the frozen files. Where a file was corrected after its first save, the timestamp file lists both hashes and the later one is the file here. `logs/amendment_timestamp.txt` names the files by their absolute paths on the machine where they were written.
- `logs/check_tif_vs_zarr*.log`, `logs/components_*.log`: the checks and component boxes `PREREG.md` section 2 cites.
- `pod_run.sh`, `go5.sh`, `go6.sh`, `go7.sh`, `regions.json`, `labels/manifest_full.json`, `scripts/`: everything the pod ran. `scripts/hf65/` is a copy of `correct/apply_fit.py` and three files of `measure/` from this repository at a7b7468.
- `results/<attempt>/`: each run's `config.json`, `eval.json` and the trainer's `validation_metrics.jsonl`, plus the readouts and run logs.

Not included: model checkpoints, trainer and evaluation stdout and stderr logs (progress bars, up to 1.5 MB each), the label arms (1.1 GB, rebuilt by `prepare`), downloaded data and the volume cache, configs written for arms that never trained in attempt 0, the 5 minutes of AS seed 27 that ran before the pod stopped, and the local CPU smoke test and WSL prep scripts.

Check the frozen files against the timestamp files:

```
sha256sum PREREG.md AMENDMENT-*.md logs/amendment_choice.txt scripts/decide_f.py scripts/decide_g.py
```

`scripts/decide_f.py` has no timestamp file of its own. Its hash is recorded only as the prefix `82ca0c24` in `AMENDMENT-G-2026-10-31.md`, which was frozen at 14:49 +0800 on 2026-10-31, when C0 seed 27 was the only finished run (`logs/amendment_g_timestamp.txt`). The repository's `.gitattributes` turns off line ending conversion for this folder, so the hashes also match after a checkout on Windows.

## Rerun

Re-score the published runs without a GPU (Python 3, standard library only):

```
mkdir -p out
python -I scripts/decide_f.py results/F/runs out/results_F.md
python -I scripts/decide_g.py results/F/runs out/results_G.md results/L1/runs/C0_s27/eval.json results/L2/runs/C0_s27/eval.json
python -I scripts/decide.py results/F/runs out/results_prereg_rules.md
```

Each writes a `.md` and a `.json` that match the files in `results/F/`.

Train again: a Linux pod with one NVIDIA GPU with bf16 and driver 580 or newer, at least 8 vCPUs, 32 GB RAM and 60 GB disk. `prepare` installs uv and Python 3.14.6, checks out villa 37d1897fb, downloads the public tifxyz and label TIFFs, rebuilds the label arms and checks them against `labels/manifest_full.json`. The scripts expect to sit in `~/labelshift`.

```
cp -r labelshift ~/labelshift && cd ~/labelshift
export STEPS=12000 ARMS=C0,AS,C8y SEEDS="27 28" MAXSTEPS_MODE=anneal WORKERS=16 G0_WAIVED=1
bash pod_run.sh prepare
FORCE=1 bash pod_run.sh run
venv/bin/python -I scripts/decide_f.py runs results_F.md
```

These are the settings `go7.sh` used. `pod_run.sh run` trains and scores each run in turn, skipping runs that already have `train.done` and `eval.json`. Without `G0_WAIVED=1` it stops after C0 seed 27 when that run's pooled AUC is below 0.70, as PREREG section 7 says.

The gate attempts used `pod_run.sh` with these settings (from Amendment C and `go5.sh`). For attempt 0, `run` passed its budget check because `scripts/reproject.py` had rewritten `runs/gpusmoke/projection.txt` from the measured training rate (Amendment C). On a fresh pod, run `smoke` first or set `FORCE=1`.

```
# attempt 0 (default teacher schedule)
export STEPS=6000 ARMS=C0,AS,C8y,C4y,C2y SEEDS="27 28" WORKERS=16 CAP_HOURS=13.03
bash pod_run.sh prepare && bash pod_run.sh run

# L1 and L2 (STEPS=12000 for L2)
export STEPS=6000 ARMS=C0 SEEDS=27 MAXSTEPS_MODE=anneal WORKERS=16
bash pod_run.sh prepare && FORCE=1 bash pod_run.sh run
```

`go5.sh`, `go6.sh` and `go7.sh` are the drivers as they ran: L1 then L2, the Amendment F choice, and the restart. `go6.sh` installs an Amendment F copy of `pod_run.sh` (`pod_run.sh.F`). The `pod_run.sh` here has the `G0_WAIVED` switch, so it is that version. Its defaults (teacher schedule, gate on) are the attempt 0 behaviour. Rerunning `scripts/mkconfig.py` with the attempt 0 settings and with the Amendment F settings gave `results/attempt0/runs/C0_s27/config.json` and `results/F/runs/AS_s28/config.json` again; the only differences were paths that Git Bash on Windows rewrote.

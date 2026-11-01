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

## Amendment H

Amendment H adds two arms, C16y and C32y, which move C0 by 64 and 128 level-0 px in y, with seeds 27 and 28 on the settings above. It reuses the F runs of C0, AS and C8y and judges w00 only, with the seed noise pooled over all five arms. Its rules and `scripts/decide_h.py` were frozen at 22:08 +0800 on 2026-10-31 (`logs/amendment_h_timestamp.txt`), before any of its runs started.

Held-out AUC of the four new runs (`results/H/results_H.md`):

| run | w00 | w018 | pooled |
|---|---|---|---|
| C16y s27 | 0.7374 | 0.5098 | 0.6416 |
| C16y s28 | 0.7447 | 0.5269 | 0.6557 |
| C32y s27 | 0.7020 | 0.5183 | 0.6266 |
| C32y s28 | 0.7072 | 0.5283 | 0.6324 |

Mean w00 AUC over the two seeds: C16y 0.7410, C32y 0.7046. The seed noise pooled over the five arms is s = 0.0056 (df 5, t = 2.571). C0's w00 lower bound is 0.7302, above chance. As in Amendment G, d is m(C0) minus m(arm):

| arm | d | 95% interval | verdict | upper end as share of headroom |
|---|---|---|---|---|
| C16y | -0.0006 | -0.0151 to 0.0140 | no cost detected | 6% |
| C32y | 0.0359 | 0.0214 to 0.0504 | cost detected | 21% |

For C16y the data rule out a cost above 0.0140 AUC, 6% of C0's headroom of 0.2405. C32y's whole interval lies above zero. The dose response, reported only, is -0.0095 AUC per 32 px, a least squares fit over the arm means at 0, 32, 64 and 128 px.

Session check: C0 seed 28 was retrained in H's session (`results/H/runs/C0_s28`). Its `eval.json` equals Amendment F's in every value except the run time, and its `validation_metrics.jsonl` is byte for byte the same, so the retrain reproduced the earlier run exactly and the new session did not move its score. `scripts/h_crosscheck.py`, written apart from `decide_h.py`, recomputes the reading from the `eval.json` files and agrees with `results_H.json`. The largest difference is 4e-4, in t, from rounding in the t table.

Amendment I adds a third seed (29) for C0, C16y and C32y, trained after H's runs on the same pod, and reads w00 again with `scripts/decide_i.py`. It was frozen at 23:46 +0800 on 2026-10-31 and `decide_i.py` at 23:53, both while H's first run was still training (`logs/amendment_i_timestamp.txt`), so the extra runs do not depend on what H shows.

Amendment J adds a fourth seed (30) for C0, C16y and C32y. It runs on a second pod with Amendment I's settings and reads w00 over seeds 27 to 30 with `scripts/decide_j.py`. It was frozen at 04:50 +0800 on 2026-11-01, before any Amendment I result existed and before the second pod was created (`logs/amendment_j_timestamp.txt`). If all three seed 30 runs have an `eval.json` by 11:30 +0800 that day, the four-seed reading is the primary one; otherwise Amendment I's three-seed reading is.

The same pod also runs an eval offset scan (`SCAN-2026-11-01.md`). Every training arm carries C0's labels in the held-out boxes, so every run above is scored against C0. The scan scores finished checkpoints against held-out labels moved down by 16, 32, 64 and 128 px, and against the held-out labels as shipped (`labels/full_eval`, built by `scripts/build_eval_labels.py`). It asks how fast a C0 model's held-out AUC falls when the held-out labels are off, and whether a model trained on moved labels scores higher on labels moved the same way. It is descriptive and outside the rules of `PREREG.md` and the amendments. It was frozen at 01:36 +0800 on 2026-11-01 and revised at 01:37 to add C0 on the as-shipped labels for both seeds, then at 02:40 to start while H trains instead of after it, each time before any scan output existed (`logs/scan_timestamp.txt`).

Pair 1 gave C0 seed 27's `eval.json` back exactly (CHECK_OK, largest AUC difference 0), so the scan's numbers are reported. `results/scan/scan_readout.md` has the tables below with pooled and w018 next to them.

Question 1, w00 AUC of the C0 checkpoints on the held-out labels as placed (E0) and moved down, with the change against E0 in brackets:

| checkpoint | as placed | 16 px | 32 px | 64 px | 128 px |
|---|---|---|---|---|---|
| C0 s27 | 0.7425 | 0.7416 (-0.0009) | 0.7376 (-0.0050) | 0.7171 (-0.0255) | 0.6603 (-0.0822) |
| C0 s28 | 0.7384 | not run | 0.7312 (-0.0072) | 0.7098 (-0.0286) | 0.6550 (-0.0834) |

The 16 px move was scored for seed 27 only. For scale, H's seed noise is s = 0.0056.

Question 2, w00 AUC of each checkpoint trained on moved labels, scored on the held-out labels moved the same way (its own move) and on E0:

| checkpoint | own move | w00 on E0 | w00 on own move | difference | C0 same seed: difference there |
|---|---|---|---|---|---|
| C8y s27 | 32 px | 0.7510 | 0.7476 | -0.0034 | -0.0050 |
| C8y s28 | 32 px | 0.7361 | 0.7356 | -0.0005 | -0.0072 |
| C16y s27 | 64 px | 0.7374 | 0.7216 | -0.0158 | -0.0255 |
| C16y s28 | 64 px | 0.7447 | 0.7308 | -0.0139 | -0.0286 |
| C32y s27 | 128 px | 0.7020 | 0.6475 | -0.0545 | -0.0822 |
| C32y s28 | 128 px | 0.7072 | 0.6482 | -0.0590 | -0.0834 |
| AS s27 | as shipped | 0.7516 | 0.7498 | -0.0018 | -0.0012 |
| AS s28 | as shipped | 0.7510 | 0.7492 | -0.0017 | -0.0022 |

None of the 8 checkpoints scored higher on its own move than on the labels as placed. 7 of 8 lost less there than C0 of the same seed; AS seed 27 lost 0.0018 against C0's 0.0012. Each pair is one evaluation, and no significance test is applied. For scale, moving the held-out labels 64 px costs C0 0.0255 and 0.0286 w00 AUC, about five times the seed noise, while training on labels moved 64 px showed no detectable cost in Amendment H.

## Amendment I

All three seed 29 runs finished by 09:12 +0800 on 2026-11-01, before Amendment I's deadline of 11:00. Held-out AUC of the new runs (`results/I/results_I.md`):

| run | w00 | w018 | pooled |
|---|---|---|---|
| C0 s29 | 0.7258 | 0.5156 | 0.6390 |
| C16y s29 | 0.6975 | 0.5011 | 0.6204 |
| C32y s29 | 0.6878 | 0.5025 | 0.6105 |

Seed 29 scored below seeds 27 and 28 on w00 in all three arms, C16y most of all (0.6975 against 0.7374 and 0.7447). The pooled seed noise rose from H's s = 0.0056 (df 5) to s = 0.0148 (df 8, t = 2.306), so the intervals came out wider than H's instead of the 27% narrower that Amendment I expected. Mean w00 AUC over the three seeds: C0 0.7356, C16y 0.7265, C32y 0.6990. C0's w00 lower bound is 0.7159, above chance.

| arm | d | 95% interval | verdict | upper end as share of headroom |
|---|---|---|---|---|
| C16y | 0.0091 | -0.0188 to 0.0370 | no cost detected | 16% |
| C32y | 0.0366 | 0.0087 to 0.0645 | cost detected | 27% |

Both verdicts are the same as in H. With the larger seed noise the data rule out a C16y cost above 0.0370 AUC, 16% of C0's headroom of 0.2356, where H's two seeds ruled out one above 0.0140. C32y's interval still lies above zero. The dose response, reported only, is -0.0103 AUC per 32 px.

Checks: `decide_i.py` run again off the pod gives the same `results_I.md` and `results_I.json`, and so does `decide_j.py` with no seed 30 run. `scripts/ij_crosscheck.py`, written apart from both, agrees with `results_I.json`; the largest difference is about 1e-16.

Amendment J's seed 30 runs are still training on the second pod. If all three finish by 11:30 +0800, their four-seed reading becomes the primary one and is added below.

## Limits

- Exploratory. Amendment F was written after two G0 failures and Amendment G after three, with C0 seed 27 already known.
- One segment that learns (w00), two seeds per arm (three for C0, C16y and C32y from Amendment I on), and fixed held-out boxes. The intervals cover run to run variation in training, not the choice of boxes.
- A weak model, about 0.74 AUC on w00. A stronger model could be more sensitive to label position, so "no cost detected" holds for this model at this level.
- C8y shifts y only. AS's offset on w00 is two-axis and varies with position.
- C0 seed 27 was trained before a pod restart and the other five runs after it (`F_RESTART_NOTE.md`). Checked: the configs differ only in seed and output folder, both C0 seeds drew the same 2,478 patches (`sampling_audit` in the trainer log), and every install used the same pinned `scripts/pod_requirements.txt`. The package freeze before the restart was not recorded, so matching versions are inferred. The freeze after the restart is `results/F/pod_venv_freeze_after_restart.txt`. Amendment H's session check later retrained C0 seed 28 in a new session and got F's `eval.json` back exactly, apart from the run time.

## Files

- `PREREG.md`, `AMENDMENT-A-2026-10-31.md` to `AMENDMENT-J-2026-11-01.md`, `F_RESTART_NOTE.md`: the pre-registration and every change to it, as written at the time.
- `SCAN-2026-11-01.md`, `scan/pairs.tsv`: the eval offset scan and its pairs, as frozen. `scan/eval_labels_manifest.json` lists its label folders with their sha256, `logs/build_eval_labels.log` is the build and `logs/scan_precheck_patches.log` the patch counts on the pod.
- `logs/*timestamp*.txt`, `logs/amendment_choice.txt`: save times and sha256 of the frozen files. Where a file was corrected after its first save, the timestamp file lists both hashes and the later one is the file here. `logs/amendment_timestamp.txt` names the files by their absolute paths on the machine where they were written.
- `logs/check_tif_vs_zarr*.log`, `logs/components_*.log`: the checks and component boxes `PREREG.md` section 2 cites.
- `pod_run.sh`, `go5.sh`, `go6.sh`, `go7.sh`, `go8.sh`, `go9.sh`, `go9b.sh`, `go9c.sh`, `scan.sh`, `scan2.sh`, `scan_wait_h.sh`, `go_j.sh`, `pod_setup_j.sh`, `vault_sync_j.sh`, `final_sync_j.sh`, `regions.json`, `labels/manifest_full.json`, `scripts/`: everything the pods ran. `scripts/hf65/` is a copy of `correct/apply_fit.py` and three files of `measure/` from this repository at a7b7468.
- `results/<attempt>/`: each run's `config.json`, `eval.json` and the trainer's `validation_metrics.jsonl`, plus the readouts and run logs.
- `results/H/`: Amendment H's four new runs and the C0 seed 28 session check, laid out as in `results/F/`, with `results_H.md`, `results_H.json` and the logs of `go8.sh`.
- `results/I/`: Amendment I's three seed 29 runs, laid out as in `results/H/`, with `results_I.md`, `results_I.json` and the logs of `go9c.sh` (`go9.log`, `prepare_I.log`, `run_I.log`, `decide_i.log`).
- `results/scan/`: the scan's readout, pair 1's check (`check.txt`), `scan.log`, `go9.log` and the `eval_heldout.py` output of each of the 18 pairs (`out/<checkpoint>__<folder>.json`).
- `scripts/h_crosscheck.py`: an independent recompute of the Amendment H reading, written apart from `decide_h.py` and run off the pod (`python -I scripts/h_crosscheck.py results/F/runs results/H/runs results/H/results_H.json`).
- `scripts/ij_crosscheck.py`: an independent recompute of the Amendment I and J readings, written apart from `decide_i.py` and `decide_j.py` (`python -I scripts/ij_crosscheck.py results/F/runs results/H/runs results/I/runs - results/I/results_I.json`).

Not included: model checkpoints, trainer and evaluation stdout and stderr logs (progress bars, up to 1.5 MB each), the scan's stdout and stderr logs for each pair, the label arms (1.1 GB, rebuilt by `prepare`), the scan's label folders (rebuilt by `scripts/build_eval_labels.py`), downloaded data and the volume cache, configs written for arms that never trained in attempt 0, the 5 minutes of AS seed 27 that ran before the pod stopped, and the local CPU smoke test and WSL prep scripts.

Check the frozen files against the timestamp files:

```
sha256sum PREREG.md AMENDMENT-*.md logs/amendment_choice.txt scripts/decide_f.py scripts/decide_g.py scripts/decide_h.py scripts/decide_i.py scripts/decide_j.py SCAN-2026-11-01.md scan/pairs.tsv scan.sh scan2.sh scripts/decide_scan.py
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

Amendment I's reading needs H's and I's runs in one folder, as they were on the pod:

```
mkdir -p hi && cp -r results/H/runs/. results/I/runs/. hi/
python -I scripts/decide_i.py results/F/runs hi out/results_I.md
```

It writes `out/results_I.md` and `out/results_I.json`, which match the files in `results/I/`.

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

`go5.sh`, `go6.sh` and `go7.sh` are the drivers as they ran: L1 then L2, the Amendment F choice, and the restart. `go8.sh` runs Amendment H and `go9.sh` Amendment I. `pod_setup_j.sh` starts `go_j.sh`, which runs Amendment J on the second pod. `go9.sh` was stopped at 01:37 +0800 on 2026-11-01, while it was still waiting for H, and `go9b.sh` took its place: it is `go9.sh` with `scan.sh` run after H and before Amendment I's prepare. `go9b.sh` was stopped the same way at 02:39 and `go9c.sh` took its place: the scan's first pass (`scan2.sh`) runs while H trains, `scan_wait_h.sh` runs it again once C32y seed 28 has trained, and `go9c.sh` runs it once more after H for any pair still missing. `go6.sh` installs an Amendment F copy of `pod_run.sh` (`pod_run.sh.F`). The `pod_run.sh` here has the `G0_WAIVED` switch, so it is that version. It also has Amendment H's `BUILD_ARMS` switch, whose default is the old list of eight arms. Its defaults (teacher schedule, gate on) are the attempt 0 behaviour. Rerunning `scripts/mkconfig.py` with the attempt 0 settings and with the Amendment F settings gave `results/attempt0/runs/C0_s27/config.json` and `results/F/runs/AS_s28/config.json` again; the only differences were paths that Git Bash on Windows rewrote.

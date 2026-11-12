# Addendum to A: seeds 31 and 32

Written on 2026-11-12. After A's result was published in `RESULTS.md` on 2026-11-07, the GPU machine was free, so both arms were trained once more with seeds 31 and 32. These four runs are not part of `PREREG.md`. The section 7 outcome stays as it was, decided on seeds 27 to 30. The extra runs show how the result looks with six seeds per arm.

## Result

All six C0 runs scored above all six AS runs on the primary reading. The lowest C0 run is 0.8215 and the highest AS run 0.7707. With the same seed, C0 was ahead in all six pairs, by 0.0642 to 0.1249.

| run | primary: AUC on C0 labels | secondary: AUC on labels as shipped |
|---|---|---|
| C0 s31 | 0.8215 | 0.6272 |
| C0 s32 | 0.8374 | 0.6359 |
| AS s31 | 0.7573 | 0.6090 |
| AS s32 | 0.7679 | 0.6138 |

Over six seeds per arm, with the formula of `scripts/decide_w01.py` (10 degrees of freedom, t = 2.228), d = 0.0847 with a 95% interval of 0.0644 to 0.1051. The four pre-registered seeds gave d = 0.0937 and 0.0641 to 0.1233. The gap in each new pair is smaller than in any of the first four, and the interval is still above zero. The two seeds were added after the outcome was known, so the six-seed interval is descriptive only.

On the secondary reading, against the labels as shipped inside the held-out boxes, C0 was ahead in all six pairs as well: d = 0.0286, interval 0.0119 to 0.0453.

`scripts/addendum_w01.py` wrote `results/addendum.md` from `results/runs/` and `results/extra/`. Its four-seed lines match `results/results.md`.

## How the runs were made

The runs used villa at 37d1897fb, A's environment and A's `desk_launch.sh`. `extra_seeds.sh` started them in a separate work folder so that A's `runs/` stayed as published. Each config differs from the same arm's seed 27 config only in the seed, the description and three folder paths. The patch lists the trainer wrote, which include each label's fingerprint, match A's byte for byte once the folder name is replaced. The sampling audit counts 1350 patches for C0 and 1374 for AS, as in A.

The runs went in the order C0 s31, AS s31, C0 s32, AS s32, between 2026-11-11 01:08 and 2026-11-12 13:13 +0800. None failed or restarted. Each took 12,916 to 21,260 s including both evaluations. `results/extra/extra.log` is the launcher's log.

## Correction: patch 63 in the primary reading

Checking these runs showed a detail of the evaluation that applies to A's eight runs as well. In every primary evaluation, held-out patch 63 cannot produce an admissible crop, so villa's InkDataset hands back a different held-out patch in its place, and that patch is scored. Which patch it draws depends on the run's seed: 124 for seed 27, 88 for 28, 206 for 29, 115 for 30, 38 for 31 and 295 for 32. Both arms of a seed get the same one. Their primary n_pos and n_neg are identical, as the table in `results/addendum.md` shows, so each same-seed comparison is on the same patches. From one seed to the next, the primary set differs by that one patch.

The primary reading therefore covers 333 distinct held-out patches plus one repeat, while `PREREG.md` section 6 meant every held-out patch once and `RESULTS.md` said all 334. The secondary reading has no such substitution: its n_pos and n_neg are the same in all twelve runs. The decision rule counts seeds as the unit of noise, so the one-patch change between seeds is part of s. Why patch 63 has no admissible crop was not looked into.

## Files

- `results/extra/<arm>_s<seed>/`: `config.json`, `eval.json`, `eval_AS.json` and `out/validation_metrics.jsonl` for seeds 31 and 32.
- `results/extra/extra.log`: the launcher's log.
- `results/addendum.md` and `results/addendum.json`: output of `scripts/addendum_w01.py`.
- `extra_seeds.sh`: the launcher.

Home directories in paths are written as `~`. Checkpoints are not in the repository.

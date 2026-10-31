# Amendment A to PREREG.md: 7-hour budget

Written 2026-10-31 05:24 +0800, before any GPU run. PREREG.md is unchanged (sha256
`5adbff4741cad2a6d1fd632e0fe9d80f19088c98b6a32163254c16d4c3e792f7`, checked at 05:23). Where this file and
PREREG.md disagree, this file wins; everything it does not mention stays as in PREREG.md.

## Why

Stan's GPU balance is 213 points at 30.28 points per pod hour, so 7.0 pod hours in total, `prepare`
included. The PREREG design (C0, AS, C8y, C4y, C2y, two seeds each) is 15 to 25 GPU hours on a 5090 by the
REPORT.md estimate and does not fit.

## Budget

- Hard stop 6.5 h after the pod starts (about 197 points, leaving about 16). Whatever is running then is
  stopped and counts as missing under G1.
- `prepare` runs with `ARMS=C0,AS,C8y` (the only arms any plan below can use).
- After `smoke`, T = 6.5 minus the pod hours used so far. per_run(S) is the smoke projection's GPU hours per
  run at STEPS=S (it already adds 10% for validation and the held-out eval).

## Plan, chosen once from the smoke projection

The first plan that fits is used. The choice and the numbers behind it go to `logs/amendment_choice.txt`
before `run` starts, and are not changed afterwards.

- P1: `ARMS=C0,AS,C8y SEEDS="27 28"`, STEPS the largest of 6000, 4000, 3000 with 6 x per_run(S) <= T.
- P2: `ARMS=C0,AS SEEDS="27 28"`, STEPS the largest of 6000, 4000, 3000 with 4 x per_run(S) <= T.
- P3: `ARMS=C0,AS SEEDS=27`, STEPS=3000, if 2 x per_run(3000) <= T. Pilot only: no rule below is
  evaluated and nothing from it goes into a villa comment.
- None fits: no `run`; the pod is stopped.

`run` gets `CAP_HOURS` = T, so `pod_run.sh` refuses a plan that its own projection puts over the budget.
Runs go in the script's order: every arm of seed 27, then every arm of seed 28. No two runs share the GPU.

## Rules as amended

- G0 unchanged: C0 seed 27 below 0.70 pooled AUC stops everything.
- G1: the complete set is the chosen plan's arms with both seeds.
- Noise: the PREREG formula over the complete arms.
- R1: evaluated under P1 only (it needs C8y).
- R2: dropped (it needs C2y and C4y).
- R3: d_AS = m(C0) - m(AS) against delta, as in PREREG. The equivalent-shift step uses the two-point line
  through (0, m(C0)) and (32, m(C8y)) under P1, and the report says it is a two-point yardstick. Under P2
  only the comparison of |d_AS| with delta is reported.
- R4 unchanged.
- A smaller STEPS than 6000 makes G0 more likely to stop the run. That outcome is reported as it is.

# Amendment C to PREREG.md: plan re-chosen from the measured training rate

Written 2026-10-31 09:47 +0800. PREREG.md, Amendment A and Amendment B are unchanged. This file replaces the
choice recorded in `logs/amendment_choice.txt` (P1, STEPS=4000). No held-out evaluation has run. The only
outputs that exist are the throughput numbers below and the first run's progress bar and in-training
validation lines (C0 seed 27, steps 250 to 750 of 4000), which are not the PREREG outcome metric.

## Why

The smoke projection measured 0.693 steps/s (data wait 0.96 s of 1.44 s per step). The first real run,
C0 seed 27 at STEPS=4000 with the same WORKERS=16, reached step 900 at 6 min 41 s of wall clock, 2.24 steps/s
including start-up and three validation passes (2.41 steps/s between steps 150 and 900). The smoke rate was
about 3.2 times too slow, most likely because its 150 measured steps ran on a cold volume cache. The plan was
chosen from a number that turned out to be wrong, so it is chosen again with the same rule and the measured
rate. That run was stopped at step 923 and its files moved to `aborted_p1_4000/` on the pod. It is not used.

## Rule

Amendment B unchanged, with two substitutions:
- per_run(S) = S / 2.24 / 3600 * 1.10, using the measured wall-clock rate above in place of the smoke rate.
- T = hours from 10:00 +0800 to the 23:20 hard stop, minus 0.3 h: 13.03 h. The platform auto-stop is now
  23:30 +0800 (Stan, 09:4x).

per_run(6000) = 0.818 h. P0 needs 10 x 0.818 + 0.25 = 8.43 h <= 13.03 h, so P0 is chosen: the PREREG design,
`ARMS=C0,AS,C8y,C4y,C2y SEEDS="27 28" STEPS=6000`, WORKERS=16, CAP_HOURS=13.03. PREREG section 7 applies
unchanged (Amendment A's "Rules as amended" do not apply under P0).

P0 still fits if the rate drops to 1.43 steps/s. If the 23:20 hard stop arrives first, whatever is unfinished
counts as missing under G1, as in Amendments A and B. Run order is the script's: every arm of seed 27, then
every arm of seed 28.

## Mechanics

`prepare` re-run with the P0 arms and STEPS=6000 rewrites the ten run configs and two eval configs; the label
arms are already built and are not rebuilt. `scripts/reproject.py` writes `runs/gpusmoke/projection.txt` from
`runs/gpusmoke/rate_measured.json` (steps_per_second 2.24, source as above) so `run` passes its gate without
FORCE. The smoke's own projection stays in `projection_smoke.txt`.

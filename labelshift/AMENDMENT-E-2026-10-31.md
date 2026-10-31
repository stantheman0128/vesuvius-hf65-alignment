# Amendment E to PREREG.md: a schedule fix after G0 failed

Written 2026-10-31 10:53 +0800, after G0 failed under Amendment C and before any run of this amendment. PREREG.md
and Amendments A to D are unchanged as records. Stan asked at 10:4x to keep going on the GPU.

## What happened

Under Amendment C (P0, STEPS=6000, teacher schedule) the pilot C0 seed 27 scored pooled held-out AUC 0.6182
(w00 0.6901, w018 0.5119), below the 0.70 gate, so P0 stopped as PREREG section 7 says. That result stands and is
reported as attempt 0. No other arm was trained, so no arm comparison exists.

## Why a second try is allowed

G0 is a competence gate on C0 alone. It says the setting did not learn ink; it says nothing about the label
shift. The configs scaled the teacher's scheduler horizon (`max_steps` 250000 for 60001 iterations) to 25000 for
6000 steps, so the cosine learning rate was still at 88% of its peak when training stopped (lr 8.81e-3 of 1e-2
at step 6000 in the progress bar), and the in-training validation was flat from step 1500 on. villa's own default, when a config
does not set `max_steps`, is the iteration count (`config.py:1467-1469` at main), which ends the schedule near
zero. Changing the schedule horizon changes only how C0 is trained, before any arm is compared, and every arm
of the comparison uses the same setting. All attempts are reported.

## The ladder

Each step trains and scores C0 seed 27 only, with `MAXSTEPS_MODE=anneal` (max_steps = STEPS; warmup and EMA
start scaled as before). Steps are tried in order and the first one that passes G0 (pooled AUC >= 0.70) is used:

- L1: STEPS=6000.
- L2: STEPS=12000, only if L1 fails.
- If L2 fails: stop. Outcome: inconclusive after three attempts, all reported.

## Comparison after a pass

per_run = the measured wall-clock time of the passing pilot (training plus held-out evaluation). The passing
pilot is the C0 seed 27 run of the comparison. T = hours from the moment of the pass to the 23:20 +0800 hard stop,
minus 0.3 h. The first plan that fits is used and written to `logs/amendment_e_choice.txt` before it starts:

- P0: the PREREG design (C0, AS, C8y, C4y, C2y, seeds 27 and 28): 9 more runs, if 9 x per_run <= T. PREREG
  section 7 applies. Amendment D (x arms) applies after P0 with its own N rule.
- P1: C0, AS, C8y with both seeds: 5 more runs, if 5 x per_run <= T. Amendment A's "Rules as amended" apply.
- P2: C0, AS with both seeds: 3 more runs, if 3 x per_run <= T. Amendment A's "Rules as amended" apply.
- None fits: stop after the pilot.

Run order is the script's (seed 27 arms, then seed 28). The 23:20 hard stop and the 23:30 platform auto-stop
still apply; a run cut off by either counts as missing.

## Mechanics

The pod lost /workspace when it stopped, so `prepare` runs again from the bundle (labels are rebuilt and checked
against the manifest). `scripts/mkconfig.py --max-steps-mode anneal` and `MAXSTEPS_MODE` in `pod_run.sh` are the
only code changes; the default stays `teacher`. The budget logic above lives in `go5.sh`, which calls
`pod_run.sh run` with FORCE=1 because no smoke projection is made under this amendment. A failed ladder step's
`runs/` folder is moved to `runs_<step>/` before the next step.

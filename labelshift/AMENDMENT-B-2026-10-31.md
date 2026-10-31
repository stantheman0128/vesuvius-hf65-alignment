# Amendment B to PREREG.md: budget raised

Written 2026-10-31 06:13 +0800, before `smoke` has run and before any plan is chosen. PREREG.md and
AMENDMENT-A-2026-10-31.md are unchanged. This file replaces Amendment A's "Budget" section and adds one plan above
P1. Everything else in Amendment A stays.

## Why

Stan added 350 points at about 06:10. The first pod (booted 05:18:57, paused about 05:29) used about 6 points of
the 213, so about 557 points are left for the current pod (e105f54d, booted 05:50:52 +0800) at 30.29 points per
hour: 18.4 pod hours.

## Budget

- Hard stop 17.5 h after this pod booted: 2026-10-31 23:20 +0800 (about 530 points, leaving about 27).
- The platform auto-stop is set to 12:25 +0800. Stan is asked to extend it after 09:30. If the pod stops before a
  run finishes, that run counts as missing under G1, as in Amendment A.
- T = hours from the end of `smoke` to the hard stop, minus 0.3 h. per_run(S) as in Amendment A.

## Plan, chosen once from the smoke projection

The first plan that fits is used, recorded in `logs/amendment_choice.txt` before `run` starts.

- P0: the PREREG design, `ARMS=C0,AS,C8y,C4y,C2y SEEDS="27 28" STEPS=6000`, if 10 x per_run(6000) + 0.25 <= T.
  The 0.25 h is for re-running `prepare` to build the C4y and C2y label arms. Under P0, PREREG section 7 applies
  unchanged.
- P1, P2, P3: as in Amendment A, with this T.
- None fits: no `run`.

Run order is the script's: every arm of seed 27, then every arm of seed 28.

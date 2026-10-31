# Amendment I (2026-10-31): a third seed for C0, C16y and C32y

Written on 2026-10-31 between 23:45 and 00:00 +0800, while Amendment H's first run (C16y seed 27) was still training and
no Amendment H result existed; the save time, the sha256 and the pod state at that moment are in
`logs/amendment_i_timestamp.txt`. At about 23:40 +0800 Stan asked to run more whenever that helps or the answer is
uncertain, and said the budget allows it. Exploratory like F, G and H, since G0 failed in all three attempts.

## Why

H gives every arm two seeds, so its w00 intervals for C16y and C32y are d +/- 2.571 s (df 5, se = s). A third seed for
C0, C16y and C32y makes them d +/- 2.306 s sqrt(2/3) = d +/- 1.883 s (df 8), about 27% narrower. These runs are decided
now, before any H result, so whether they run does not depend on what H shows.

## Runs

- After `go8.sh` ends (ALL_DONE_H), the same pod trains seed 29 of C0, C32y and C16y, in that order, with H's settings:
  STEPS=12000, MAXSTEPS_MODE=anneal, WORKERS=16, G0_WAIVED=1, villa 37d1897fb, the labels H built, and the same
  regions, held-out boxes and scoring. `pod_run.sh prepare` with SEEDS=29 writes the configs; `pod_run.sh` is unchanged.
- The local stop deadline moves to 2026-11-01 11:00 +0800. A run that has not finished by then is missing, and the
  reading uses the seeds that exist.

## Readout (`scripts/decide_i.py`)

Amendment H's readout items 1 to 5, with these changes:

1. Seeds 27, 28 and 29. C0 seeds 27 and 28 and every AS and C8y run come from Amendment F; C0 seed 29 and every C16y
   and C32y run come from this pod session. C0 seed 28 retrained in this session (H's session check) still enters no
   interval.
2. Seed noise: s^2 is the sum, over the arms among C0, AS, C8y, C16y and C32y that have at least two seeds, of
   sum over seeds of (x - m(arm))^2, divided by df, where df is the sum over those arms of (n(arm) - 1). With two seeds
   per arm this is H's formula. t is the 97.5% quantile of Student t with that df (2.306 for df 8).
3. The C0 lower bound m(C0) - t s / sqrt(n(C0)), d, se = s sqrt(1/n(C0) + 1/n(arm)), the interval, the three verdicts
   and the dose response slope are computed as in H with these n, s and t.
4. H's two-seed reading is reported first and is not replaced. I's reading is reported next to it as the three-seed
   update. With only seeds 27 and 28 present, `decide_i.py` gives H's numbers.

## Limits

- The limits of H hold.
- AS and C8y keep two seeds.
- C0 seed 29 is trained in this pod session and C0 seeds 27 and 28 in F's sessions. H's session check (C0 seed 28
  retrained here) shows how far a session moves C0.

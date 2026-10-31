# Amendment H (2026-10-31): shifts of 64 and 128 px

Written on 2026-10-31 after 22:00 +0800; the save time and sha256 are in `logs/amendment_h_timestamp.txt`.
Amendments F and G had been read out (`results_F.md`, `results_G.md`), and none of the runs below existed. Stan
approved the rules in chat at about 21:59 +0800. Exploratory like F and G, since G0 failed in all three attempts.

## Why

The 26 offset segments of the shift file are off by 6.6 to 147 px. Amendment F tested a 32 px shift (C8y) and the
labels as shipped (on w00 -1.6 to +17.4 px in y and -10.0 to +14.5 px in x) and detected no cost. H adds 64 and
128 px to see whether a cost appears inside the range that the real offsets cover.

## Runs

- Arms C16y and C32y: C0 moved 16 and 32 cells in y, which is 64 and 128 level-0 px (`shift_cell_level0_px` = 4).
  `scripts/build_labels.py` builds them like every other arm. Its tile range is padded by guard_px + 8 cells +
  64 = 608 px, which already holds a 128 px shift, so the builder is unchanged.
- Seeds 27 and 28, in the order C16y s27, C32y s27, C16y s28, C32y s28.
- The settings of Amendment F's runs: STEPS=12000, MAXSTEPS_MODE=anneal, WORKERS=16, villa 37d1897fb,
  `scripts/pod_requirements.txt`, and the same teacher config, regions, held-out boxes and scoring.
- C0, AS and C8y are not rerun. Their Amendment F `eval.json` files (`results/F/runs`) are reused.
- `pod_run.sh` gets one change: `BUILD_ARMS` sets the arms that `build_labels.py` builds, and its default is the
  old list. `go8.sh` sets it to the eight earlier arms plus C16y and C32y, so the pod's label check against
  `labels/manifest_full.json` still covers the eight earlier arms. If that check fails, nothing trains.
- Optional session check: if Stan approves before the fourth run ends, a fifth run retrains C0 seed 28 on this
  pod session. It is listed next to Amendment F's C0 seed 28 and enters no interval or verdict.

## Readout (`scripts/decide_h.py`)

Only w00 is judged, because w018 did not learn in any run (its C0 lower bound under G was 0.4983). The w018 and
pooled means are listed.

1. Seed noise: s^2 is the mean of (x_s27 - x_s28)^2 / 2 over the arms among C0, AS, C8y, C16y and C32y that have
   both seeds. df is the number of those arms (5 when all ran), and t is the 97.5% quantile of Student t with
   that df (2.571 for df 5).
2. w00 must measure something: m(C0) - t s / sqrt(n(C0)) > 0.5, as in G. Otherwise no arm gets a verdict.
3. For C16y and C32y: d = m(C0) - m(arm), se = s sqrt(1/n(C0) + 1/n(arm)), and the 95% interval is d +/- t se.
   - Lower end above 0: cost detected.
   - Upper end below 0: the arm scored above C0. This is reported and not counted as a finding.
   - Otherwise: no cost detected. The upper end is the largest cost the data allow, also given as a share of the
     headroom m(C0) - 0.5.
   An arm with one finished seed is read the same way with n(arm) = 1.
4. Dose response, reported only: the least squares slope of the w00 arm means against the shift for C0 (0 px),
   C8y (32), C16y (64) and C32y (128), in AUC per 32 px. AS is left out because its shift varies with position
   and runs in two axes.
5. Amendment G's verdicts for AS and C8y stand and are not recomputed.

## Stop rules

- The local autostop stops the pod when `go8.sh` writes ALL_DONE_H, after 60 minutes without a prepare, training
  or evaluation process, or 7 hours after the restart (8.5 hours if the session check is approved). The restart
  time and the deadline go into `logs/h_launch.txt` at launch.
- Runs that did not finish by then are reported as missing.
- Expected cost at NT$30.29/h: four runs of 71 to 76 minutes plus about 40 minutes of setup, about 5.5 to 6
  hours or NT$170 to 180, and about NT$38 more for the session check.

## Limits known now

- C0 comes from earlier pod sessions (seed 27 before the F restart, seed 28 after it), while C16y and C32y come
  from a new one. The session check measures this for one run.
- One segment that learns, two seeds per arm, and shifts in y only.

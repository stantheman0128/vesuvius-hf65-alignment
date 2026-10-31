# Amendment F to PREREG.md: an exploratory w00-only comparison if L2 fails G0

Written 2026-10-31 at about 12:40 +0800, while L2 (Amendment E, STEPS=12000) is still training and before its
result exists. The exact save time and sha256 are in `logs/amendment_f_timestamp.txt`. PREREG.md and Amendments
A to E are unchanged as records.

This amendment applies only if L2 fails G0 and Stan approves the GPU time. If L2 passes G0, Amendment E's plan
runs and this amendment is void.

## Why, and what kind of result this is

G0 failed twice: attempt 0 scored pooled 0.6182 (w00 0.6901, w018 0.5119) and L1 scored 0.6231 (w00 0.6943,
w018 0.5156). On w018 the model is at chance, so no label effect can show there. On w00 it learned something.
The 0.70 figure in G0 was chosen by the prep agent when it wrote PREREG.md, and no reason for that number was
recorded.

This amendment is post hoc. It was written after two G0 failures and after seeing the per-segment AUCs. Its
results are exploratory. Any write-up must say in its first paragraph that G0 failed in all attempts, that this
analysis was added afterwards, and that w018 did not learn.

## What changes

- G0 is waived for this amendment. No new gate replaces it.
- Primary readout: the w00 held-out voxel ROC AUC, which `eval_heldout.py` already writes per segment. Pooled
  and w018 AUCs are reported as secondary.
- Training setting: L1's (STEPS=6000, MAXSTEPS_MODE=anneal, WORKERS=16, same label arms, both segments in
  training), and L1's C0 seed 27 run is reused as C0 seed 27. Exception, fixed now: if L2's C0 seed 27 has a w00
  AUC at least 0.01 above L1's (0.6943, so 0.7043 or more), use L2's setting and L2's C0 seed 27 instead, but
  only if five runs at L2's measured wall clock end before the 23:20 stop. 0.01 is PREREG's floor for delta.
- Arms and seeds: C0, AS and C8y, seeds 27 and 28 (PREREG's P1 set). New runs, in this order: AS s27, C8y s27,
  C0 s28, AS s28, C8y s28.
- Code: `pod_run.sh run` skips its G0 check when `G0_WAIVED=1` is set. `prepare` is rerun with the new ARMS and
  SEEDS to write the missing configs. Nothing else in the run path changes.

## Rules (PREREG section 7 notation, applied to the w00 AUC)

- G1: an arm missing a seed is reported as missing and left out.
- Noise: sigma = sqrt(mean over complete arms of (s1 - s2)^2 / 2); delta = max(2 * sigma, 0.01).
- COST if m(C0) - m(C8y) > delta. NO COST if |m(C0) - m(C8y)| <= delta, stated as "no measurable cost of a
  32 px shift on w00 in this setting, with a base model near 0.69 AUC". If m(C8y) - m(C0) > delta, it is
  reported as unexplained.
- AS is compared with C0 the same way and reported. It does not decide COST.

## Budget and stop

- L1's measured wall clock is 2645 s per run (training plus held-out evaluation, `pilots.txt`), so five runs
  take about 3.7 h. Under the L2 exception, about twice that. A pod restart adds the rebuild.
- The 23:20 +0800 hard stop and the 23:30 platform auto-stop stay. Runs not finished by then are reported as
  missing.

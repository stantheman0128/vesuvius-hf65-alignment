# Amendment D to PREREG.md: the optional x arms, added after P0

Written 2026-10-31 10:07 +0800, while C0 seed 27 (the first P0 run) is at step 2947 of 6000 and before any
held-out evaluation has run. PREREG.md and Amendments A, B and C are unchanged. Stan asked for this follow-up
at 10:05.

## What

PREREG section 3 lists C2x, C4x and C8x (C0 moved right by 2, 4 and 8 cells) as optional arms that run "only if
`ARMS` includes them at launch". P0 was launched without them. They are added here as a follow-up that starts
only after P0 has finished. As PREREG R4 says, they are reported as a separate curve with the same rules. Because
they were not in the launch, that curve is also labelled as a follow-up added after launch, it is never mixed
into the P0 numbers, and nothing in the P0 results for R1 to R3 changes because of it.

## When and how much

- Start: when P0 has finished (`GO_DONE` on the pod). If G0 stopped P0, the x arms do not run.
- Per-run time is measured from P0 itself: (time P0 ended - time P0 started) / number of P0 runs that
  trained. N = the number of whole runs that fit between the x start and the 23:20 +0800 hard stop minus
  0.3 h.
- N >= 6: `ARMS=C0,C8x,C4x,C2x SEEDS="27 28"` (C0 is already trained and evaluated and is skipped).
  3 <= N < 6: `SEEDS="27"` only; the x arms are then single-seed and reported as such, with no noise estimate
  of their own. N < 3: no x arms.
- STEPS=6000 and WORKERS=16, as in P0. Order is the script's: C8x, C4x, C2x for seed 27, then seed 28.
- The 23:20 hard stop and the 23:30 platform auto-stop still apply. A run cut off by either counts as
  missing.

## Reporting

For each x arm: both seed values and the mean of the primary metric, and the difference from m(C0), next to
the y arm with the same shift. R1 and R2 are applied to the x curve (C0, C2x, C4x, C8x) with the P0 delta,
because the x arms alone give too few seed pairs for a noise estimate of their own. If the x arms are
single-seed, that is stated next to every x number.

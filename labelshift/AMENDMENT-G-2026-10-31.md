# Amendment G: a standard with stated reasons in place of the 0.70 bar

Written 2026-10-31 (time and sha256 in `logs/amendment_g_timestamp.txt`), after G0 failed three times and
before any new Amendment F run finished. The only Amendment F result known when this was written is C0 seed
27, reused from L2: w00 0.7425, w018 0.5307, pooled 0.6550. AS seed 27 is the first new run.

This is exploratory. It is a second reading of Amendment F's runs and changes nothing that runs: same arms,
seeds, steps and evaluation. Amendment F's own rules are applied as written and reported first.

## Why

Stan asked for a standard whose numbers have reasons. Three numbers in the current rules have none on record:

- G0's 0.70 pooled AUC. The prep agent chose it; the dispatch brief contains no threshold.
- The factor 2 and the floor 0.01 in Amendment F's delta = max(2 sigma, 0.01).
- G0 judged the pooled AUC, so w018, where C0 stays near chance, pulled down a number meant to say whether w00
  works.

Two questions need answers, and neither needs an invented number:

1. Does a segment measure anything? If C0 is at chance on a segment, every arm is at chance there and "no
   difference" is guaranteed whatever the labels do. The test is whether C0 is above 0.5 by more than its noise.
2. Is the difference between C0 and a shifted arm larger than run to run noise? The test is whether the
   difference is away from zero by more than its noise.

The noise is measured from the two seeds per arm. The one convention left is the 95% level, the usual default,
stated here instead of hidden in a constant.

## Rules

Each segment is read on its own: w00 and w018. Pooled is listed for continuity and not judged. x(a, s) is the
held-out AUC of arm a and seed s, n_a the number of seeds of arm a (2 when complete), m(a) the mean over seeds.

- G1, noise: s^2 is the mean over arms with both seeds of (x1 - x2)^2 / 2, with df equal to the number of those
  arms (3 when all are complete). This is Amendment F's sigma, computed for each segment. It assumes the three
  arms of a segment vary by seed the same amount.
- G2, critical value: t is the 0.975 quantile of Student's t with df degrees of freedom (3.182 at df 3). With
  df 3 this is wider than Amendment F's 2 sigma; that is the price of two seeds per arm.
- G3, does the segment measure anything (takes G0's role for that segment): if m(C0) - t s / sqrt(n_C0) is at
  or below 0.5, the segment is reported as "C0 not distinguishable from chance" and its arm differences are
  listed without a verdict.
- G4, cost of arm a (AS and C8y): d = m(C0) - m(a), SE = s sqrt(1/n_C0 + 1/n_a), interval d +/- t SE.
  - Lower end above 0: "cost detected", with d and the interval.
  - Upper end below 0: "a scored above C0", with d and the interval; the design does not explain this.
  - Otherwise: "no cost detected; the data rule out a cost above U", where U is the upper end.
- G5, context for U (reported, decides nothing): the headroom H = m(C0) - 0.5 and U as a share of H, which is
  the part of what C0 learned that the data cannot rule out losing; and what doubling the training bought on
  the same segment (C0 seed 27, L2 minus L1, one run each, so its own noise is about s sqrt(2)).
- G6, planning (decides nothing): the smallest cost detectable at 80% power with n seeds per arm,
  (t_0.975 + t_0.80 at df 3(n - 1)) s sqrt(2/n), for n = 2 to 6, and the GPU time of the extra runs.

For a future pilot the same reasoning gives a gate without an invented number: run C0 with two seeds first, and
go on only if G3 holds on the segment of interest and the planned design's G6 value is below that segment's
headroom. If the smallest detectable cost is larger than the headroom, no cost can be detected at all.

## Reading order in the report

1. Amendment F under its own rules (`results_F.md`).
2. Amendment G (`results_G.md`, from `scripts/decide_g.py`).
3. Where they disagree, both are given with the reason. The usual case would be Amendment F's 2 sigma calling a
   difference a cost while the t interval at df 3 still includes zero.

## Limits that go next to any result

- Exploratory: written after G0 failed and after C0 seed 27 was known.
- One segment where C0 learns (w00), two seeds per arm, fixed held-out boxes. The intervals cover training
  variation only, not the choice of boxes.
- C0's w00 AUC is about 0.74, so "no cost detected" is a statement about this model at this level.

## Known issue found while writing this

`scripts/decide_f.py` prints "with a base model near 0.69 AUC" in its NO COST sentence. That was written when
the L1 setting was expected; under the L2 setting C0's w00 AUC is 0.7425. decide_f.py is frozen (82ca0c24...)
and is not edited. The report gives the right number.

# Amendment G reading (exploratory; written after G0 failed, before Amendment F's new runs finished)

Each segment is read on its own. Intervals are 95% Student t intervals from the seed to seed noise, so the
only convention is the 95% level. Amendment F's own readout (results_F.md) comes first and is unchanged.

| run | w00 AUC | w018 AUC | pooled AUC |
|---|---|---|---|
| C0_s27 | 0.7425 | 0.5307 | 0.6550 |
| C0_s28 | 0.7384 | 0.5134 | 0.6443 |
| AS_s27 | 0.7516 | 0.5079 | 0.6530 |
| AS_s28 | 0.7510 | 0.5129 | 0.6521 |
| C8y_s27 | 0.7510 | 0.5172 | 0.6551 |
| C8y_s28 | 0.7361 | 0.5357 | 0.6541 |

## w00

Means: C0 0.7405 (n=2), AS 0.7513 (n=2), C8y 0.7436 (n=2)
Seed noise s = 0.0063 (df 3), t = 3.182. C0 lower bound 0.7263, headroom above chance 0.2405. Doubling training (C0 s27, L2 minus L1) gained 0.0483.
C0 is above chance on this segment.
- AS vs C0: d = C0 minus AS = -0.0108, 95% interval -0.0308 to 0.0093; no cost detected; the data rule out a cost above 0.0093 AUC (upper end is 4% of the headroom)
- C8y vs C0: d = C0 minus C8y = -0.0031, 95% interval -0.0231 to 0.0170; no cost detected; the data rule out a cost above 0.0170 AUC (upper end is 7% of the headroom)
Smallest cost detectable at 80% power with n seeds per arm: n=2 0.0262, n=3 0.0172, n=4 0.0140, n=5 0.0122, n=6 0.0109

## w018

Means: C0 0.5221 (n=2), AS 0.5104 (n=2), C8y 0.5265 (n=2)
Seed noise s = 0.0106 (df 3), t = 3.182. C0 lower bound 0.4983, headroom above chance 0.0221. Doubling training (C0 s27, L2 minus L1) gained 0.0151.
C0 is not distinguishable from chance on this segment: arm differences get no verdict.
- AS vs C0: d = C0 minus AS = 0.0117, 95% interval -0.0220 to 0.0453; no verdict (C0 not above chance)
- C8y vs C0: d = C0 minus C8y = -0.0044, 95% interval -0.0380 to 0.0292; no verdict (C0 not above chance)
Smallest cost detectable at 80% power with n seeds per arm: n=2 0.0440, n=3 0.0289, n=4 0.0235, n=5 0.0204, n=6 0.0183

## pooled (listed for continuity, not judged)

Means: C0 0.6496 (n=2), AS 0.6526 (n=2), C8y 0.6546 (n=2)
Seed noise s = 0.0044 (df 3), t = 3.182. C0 lower bound 0.6397, headroom above chance 0.1496. Doubling training (C0 s27, L2 minus L1) gained 0.0319.
C0 is above chance on this segment.

Extra GPU time for n seeds per arm (three arms, 71 min per run, NT$30.29/h): n=3 3 runs 3.5 h NT$108, n=4 6 runs 7.1 h NT$215, n=5 9 runs 10.7 h NT$323, n=6 12 runs 14.2 h NT$430

# Amendment I reading (exploratory; a third seed for C0, C16y and C32y, decided before any H result)

Only w00 is judged. C0 seeds 27 and 28, AS and C8y are Amendment F's runs; C0 seed 29, C16y and C32y are
this session's. Intervals are 95% Student t intervals from the pooled seed to seed variance of every arm with
two or more seeds. Amendment H's two-seed reading is in results_H.md and is not replaced.

| run | w00 AUC | w018 AUC | pooled AUC |
|---|---|---|---|
| C0_s27 | 0.7425 | 0.5307 | 0.6550 |
| C0_s28 | 0.7384 | 0.5134 | 0.6443 |
| C0_s29 | 0.7258 | 0.5156 | 0.6390 |
| AS_s27 | 0.7516 | 0.5079 | 0.6530 |
| AS_s28 | 0.7510 | 0.5129 | 0.6521 |
| C8y_s27 | 0.7510 | 0.5172 | 0.6551 |
| C8y_s28 | 0.7361 | 0.5357 | 0.6541 |
| C16y_s27 | 0.7374 | 0.5098 | 0.6416 |
| C16y_s28 | 0.7447 | 0.5269 | 0.6557 |
| C16y_s29 | 0.6975 | 0.5011 | 0.6204 |
| C32y_s27 | 0.7020 | 0.5183 | 0.6266 |
| C32y_s28 | 0.7072 | 0.5283 | 0.6324 |
| C32y_s29 | 0.6878 | 0.5025 | 0.6105 |

## w00

Means: C0 0.7356 (n=3), AS 0.7513 (n=2), C8y 0.7436 (n=2), C16y 0.7265 (n=3), C32y 0.6990 (n=3)
Seed noise s = 0.0148 (df 8), t = 2.306. C0 lower bound 0.7159, headroom above chance 0.2356.
C0 is above chance on this segment.
- C16y vs C0: d = C0 minus C16y = 0.0091, 95% interval -0.0188 to 0.0370; no cost detected; the data rule out a cost above 0.0370 AUC (n = 3 and 3) (upper end is 16% of the headroom)
- C32y vs C0: d = C0 minus C32y = 0.0366, 95% interval 0.0087 to 0.0645; cost detected (n = 3 and 3) (upper end is 27% of the headroom)
Dose response (reported only): AUC change per 32 px of shift -0.0103, least squares over the arm means at 0 px, 32 px, 64 px, 128 px.

## w018 and pooled (listed, not judged)

w018 means: C0 0.5199, AS 0.5104, C8y 0.5265, C16y 0.5126, C32y 0.5164
pooled means: C0 0.6461, AS 0.6526, C8y 0.6546, C16y 0.6392, C32y 0.6231

Session check (reported only): C0 seed 28 rerun in this session scored w00 0.7384 against 0.7384 in Amendment F (difference +0.0000).

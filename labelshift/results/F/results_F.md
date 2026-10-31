# Amendment F readout (exploratory, post hoc)

G0 failed in every attempt (attempt 0, L1, L2). Amendment F was added afterwards and waives it. w018 did not
learn (AUC near 0.5), so the primary metric is the w00 held-out AUC. Pooled and w018 are secondary.

| run | w00 AUC | pooled AUC | w018 AUC |
|---|---|---|---|
| C0_s27 | 0.7425 | 0.6550 | 0.5307 |
| C0_s28 | 0.7384 | 0.6443 | 0.5134 |
| AS_s27 | 0.7516 | 0.6530 | 0.5079 |
| AS_s28 | 0.7510 | 0.6521 | 0.5129 |
| C8y_s27 | 0.7510 | 0.6551 | 0.5172 |
| C8y_s28 | 0.7361 | 0.6541 | 0.5357 |

Means of the w00 AUC over seeds 27 and 28: C0 0.7405, AS 0.7513, C8y 0.7436
Missing arms (left out): none
sigma = 0.0063, delta = 0.0126

- C8y vs C0: NO COST (no measurable cost of a 32 px shift on w00 in this setting, with a base model near 0.69 AUC)
- AS vs C0 (reported, does not decide COST): NO COST

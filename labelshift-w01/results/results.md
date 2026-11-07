# A: w01 reading (PREREG.md sections 5 and 7)

| run | w01 AUC on C0 labels (primary) | w01 AUC on labels as shipped (secondary) |
|---|---|---|
| C0 s27 | 0.8380 | 0.6341 |
| C0 s28 | 0.8393 | 0.6321 |
| C0 s29 | 0.8414 | 0.6445 |
| C0 s30 | 0.8294 | 0.6172 |
| AS s27 | 0.7130 | 0.5848 |
| AS s28 | 0.7456 | 0.6001 |
| AS s29 | 0.7707 | 0.6253 |
| AS s30 | 0.7440 | 0.5864 |

Means on C0 labels: C0 0.8370 (n=4), AS 0.7434 (n=4)
C0 lower bound 0.8161, headroom above chance 0.3370 (reported only).
Secondary, labels as shipped (reported only, no verdict): C0 0.6320 (n=4), AS 0.5991 (n=4); d = 0.0329, interval 0.0061 to 0.0596

Outcome: cost measured: d = mean(C0) - mean(AS) = 0.0937, 95% interval 0.0641 to 0.1233 (s = 0.0171, df 6, t = 2.447, n = 4 and 4).

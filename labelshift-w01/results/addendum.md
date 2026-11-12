# A, exploratory addendum: seeds 27 to 32

| run | primary: AUC on C0 labels | secondary: AUC on labels as shipped | primary n_pos / n_neg |
|---|---|---|---|
| C0 s27 | 0.8380 | 0.6341 | 2033645 / 4788661 |
| C0 s28 | 0.8393 | 0.6321 | 2052405 / 4785229 |
| C0 s29 | 0.8414 | 0.6445 | 2033584 / 4790080 |
| C0 s30 | 0.8294 | 0.6172 | 2043200 / 4794756 |
| C0 s31 (added) | 0.8215 | 0.6272 | 2047681 / 4791071 |
| C0 s32 (added) | 0.8374 | 0.6359 | 2044480 / 4793358 |
| AS s27 | 0.7130 | 0.5848 | 2033645 / 4788661 |
| AS s28 | 0.7456 | 0.6001 | 2052405 / 4785229 |
| AS s29 | 0.7707 | 0.6253 | 2033584 / 4790080 |
| AS s30 | 0.7440 | 0.5864 | 2043200 / 4794756 |
| AS s31 (added) | 0.7573 | 0.6090 | 2047681 / 4791071 |
| AS s32 (added) | 0.7679 | 0.6138 | 2044480 / 4793358 |

Primary, seeds 27 to 30, pre-registered: C0 0.8370, AS 0.7434, d = 0.0937, 95% interval 0.0641 to 0.1233 (s = 0.0171, df 6, t = 2.447)
Primary, seeds 27 to 32, descriptive: C0 0.8345, AS 0.7498, d = 0.0847, 95% interval 0.0644 to 0.1051 (s = 0.0158, df 10, t = 2.228)
Secondary, seeds 27 to 30, pre-registered: C0 0.6320, AS 0.5991, d = 0.0329, 95% interval 0.0061 to 0.0596 (s = 0.0155, df 6, t = 2.447)
Secondary, seeds 27 to 32, descriptive: C0 0.6319, AS 0.6032, d = 0.0286, 95% interval 0.0119 to 0.0453 (s = 0.0130, df 10, t = 2.228)

Same-seed difference C0 - AS on the primary AUC: s27 0.1249, s28 0.0937, s29 0.0707, s30 0.0854, s31 0.0642, s32 0.0695.
Lowest C0 run 0.8215, highest AS run 0.7707.

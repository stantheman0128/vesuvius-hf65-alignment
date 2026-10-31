# Label-shift results (rules from PREREG.md section 7)

| arm | seed | pooled AUC | best F0.5 | per-segment AUC | trainer val balanced acc |
|---|---|---|---|---|---|
| AS | 27 | 0.6530 | 0.4156 | w00 0.7516, w018 0.5079 | 0.6372 |
| AS | 28 | 0.6521 | 0.4209 | w00 0.7510, w018 0.5129 | 0.5540 |
| C0 | 27 | 0.6550 | 0.4299 | w00 0.7425, w018 0.5307 | 0.6183 |
| C0 | 28 | 0.6443 | 0.4247 | w00 0.7384, w018 0.5134 | 0.5495 |
| C8y | 27 | 0.6551 | 0.4302 | w00 0.7510, w018 0.5172 | 0.6047 |
| C8y | 28 | 0.6541 | 0.4316 | w00 0.7361, w018 0.5357 | 0.5334 |

```
{
 "gate_G0": {
  "c0_seed27_auc": 0.6549824153032139,
  "pass": false
 },
 "outcome": "G0 failed: C0 seed 27 AUC 0.6550 < 0.7; inconclusive"
}
```

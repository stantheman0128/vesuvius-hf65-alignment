# Eval offset scan readout

Pair 1 check: CHECK_OK max |auc diff| per segment w00_20231016151002 0.00e+00, pooled 0.00e+00, w018_20240304144031_2um 0.00e+00

## Question 1: C0 checkpoints, held-out labels moved down

| checkpoint | segment | E0 (0 px) | E4y (16) | E8y (32) | E16y (64) | E32y (128) |
|---|---|---|---|---|---|---|
| C0_s27 | w00 | 0.7425 | 0.7416 (-0.0009) | 0.7376 (-0.0050) | 0.7171 (-0.0255) | 0.6603 (-0.0822) |
| C0_s27 | pooled | 0.6550 | 0.6551 (+0.0001) | 0.6529 (-0.0021) | 0.6399 (-0.0151) | 0.6083 (-0.0467) |
| C0_s27 | w018 | 0.5307 | 0.5310 (+0.0003) | 0.5300 (-0.0008) | 0.5277 (-0.0031) | 0.5245 (-0.0063) |
| C0_s28 | w00 | 0.7384 | not run | 0.7312 (-0.0072) | 0.7098 (-0.0286) | 0.6550 (-0.0834) |
| C0_s28 | pooled | 0.6443 | not run | 0.6415 (-0.0028) | 0.6296 (-0.0147) | 0.6015 (-0.0428) |
| C0_s28 | w018 | 0.5134 | not run | 0.5135 (+0.0001) | 0.5140 (+0.0006) | 0.5155 (+0.0021) |

Differences in brackets are against E0.

## Question 2: trained on moved labels, scored on the same move

| checkpoint | own move | w00 on E0 | w00 on own move | difference | C0 same seed: difference there | pooled difference | w018 difference |
|---|---|---|---|---|---|---|---|
| C8y_s27 | E8y | 0.7510 | 0.7476 | -0.0034 | -0.0050 | -0.0011 | -0.0011 |
| C8y_s28 | E8y | 0.7361 | 0.7356 | -0.0005 | -0.0072 | -0.0005 | -0.0027 |
| C16y_s27 | E16y | 0.7374 | 0.7216 | -0.0158 | -0.0255 | -0.0079 | +0.0012 |
| C16y_s28 | E16y | 0.7447 | 0.7308 | -0.0139 | -0.0286 | -0.0091 | -0.0040 |
| C32y_s27 | E32y | 0.7020 | 0.6475 | -0.0545 | -0.0822 | -0.0299 | -0.0026 |
| C32y_s28 | E32y | 0.7072 | 0.6482 | -0.0590 | -0.0834 | -0.0348 | -0.0080 |
| AS_s27 | EAS | 0.7516 | 0.7498 | -0.0018 | -0.0012 | -0.0003 | -0.0005 |
| AS_s28 | EAS | 0.7510 | 0.7492 | -0.0017 | -0.0022 | -0.0000 | +0.0003 |

Wording check (SCAN-2026-11-01.md): "models trained on moved labels predict moved ink" is not allowed. Trained checkpoints that ran: 8, gaining on their own move: 0; C0 comparisons available: 8, losing there: 8.

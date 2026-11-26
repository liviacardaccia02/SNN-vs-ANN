## Verification

- [x] 18 result files present (3 seeds x 6 T)
- [x] same protocol version in every run
- [x] threshold_s identical across T within each seed (one ANN per seed)
- [x] threshold_s = ANN test acc - 0.02
- [x] hybrid reused the seed's ANN with conv1 scale T/24
- [x] no conversion-gate failure (all hybrid runs status ok)
- [x] every SNN evaluated on the full test set (10 000) at its training T
- [x] sample-time-step budgets = epochs x 48 000 x T (SG, FT) and ANN + FT (hybrid)
- [x] logged best_val_acc equals max of per-epoch validation curve
- [x] ANN trained once per seed on 24-bin mean (1 440 000 sample-time-steps = 30 epochs)

Runs reaching the 30-epoch cap without early stopping: SG 15/18, FT 5/18

| seed | T | SG epochs / best | SG early-stopped | FT epochs / best | FT early-stopped |
|---|---|---|---|---|---|
| 0 | 1 | 30 / 27 | False | 17 / 10 | True |
| 0 | 2 | 28 / 21 | True | 30 / 23 | True |
| 0 | 3 | 30 / 27 | False | 30 / 23 | True |
| 0 | 6 | 30 / 26 | False | 29 / 22 | True |
| 0 | 12 | 30 / 25 | False | 28 / 21 | True |
| 0 | 24 | 30 / 25 | False | 28 / 21 | True |
| 1 | 1 | 30 / 27 | False | 30 / 24 | False |
| 1 | 2 | 30 / 28 | False | 26 / 19 | True |
| 1 | 3 | 30 / 26 | False | 28 / 21 | True |
| 1 | 6 | 30 / 27 | False | 29 / 22 | True |
| 1 | 12 | 30 / 27 | False | 18 / 11 | True |
| 1 | 24 | 30 / 28 | False | 30 / 30 | False |
| 2 | 1 | 30 / 25 | False | 30 / 26 | False |
| 2 | 2 | 28 / 21 | True | 30 / 28 | False |
| 2 | 3 | 30 / 25 | False | 30 / 30 | False |
| 2 | 6 | 30 / 27 | False | 28 / 21 | True |
| 2 | 12 | 30 / 30 | False | 30 / 23 | True |
| 2 | 24 | 19 / 12 | True | 21 / 14 | True |

## ANN baseline

| seed | val (best) | test | threshold_s | best epoch |
|---|---|---|---|---|
| 0 | 97.45 | 97.25 | 95.25 | 28/30 |
| 1 | 97.58 | 97.42 | 95.42 | 25/30 |
| 2 | 97.40 | 97.51 | 95.51 | 30/30 |

Binomial SE of one test accuracy near 98 % (n = 10 000): 0.14 pp

## Test accuracy (%)

| T | SG s0 / s1 / s2 | SG mean ± sd | hybrid s0 / s1 / s2 | hybrid mean ± sd | hybrid − SG (pp), 95 % CI | paired t, p |
|---|---|---|---|---|---|---|
| 1 | 97.85 / 97.57 / 97.76 | 97.73 ± 0.14 | 95.44 / 96.07 / 96.29 | 95.93 ± 0.44 | -1.79 ± 1.33 | t = -5.8, p = 0.0283 |
| 2 | 98.47 / 98.65 / 98.61 | 98.58 ± 0.09 | 97.24 / 97.22 / 97.50 | 97.32 ± 0.16 | -1.26 ± 0.40 | t = -13.5, p = 0.0055 |
| 3 | 98.83 / 98.91 / 98.81 | 98.85 ± 0.05 | 97.31 / 97.67 / 97.69 | 97.56 ± 0.21 | -1.29 ± 0.51 | t = -10.9, p = 0.0083 |
| 6 | 98.89 / 99.11 / 99.07 | 99.02 ± 0.12 | 97.93 / 98.11 / 97.79 | 97.94 ± 0.16 | -1.08 ± 0.43 | t = -10.7, p = 0.0086 |
| 12 | 99.13 / 99.08 / 99.23 | 99.15 ± 0.08 | 98.01 / 98.03 / 98.18 | 98.07 ± 0.09 | -1.07 ± 0.10 | t = -46.0, p = 0.0005 |
| 24 | 99.12 / 99.17 / 98.96 | 99.08 ± 0.11 | 98.16 / 98.28 / 97.90 | 98.11 ± 0.19 | -0.97 ± 0.21 | t = -19.7, p = 0.0026 |

## Primary outcome: smallest qualifying T (threshold = ANN test − 2 pp)

| seed | SG | hybrid |
|---|---|---|
| 0 | 1 | 1 |
| 1 | 1 | 1 |
| 2 | 1 | 1 |

## POST-HOC: smallest qualifying T, threshold = ANN test acc − margin (seeds 0/1/2)

| required vs ANN | SG | hybrid |
|---|---|---|
| -2.0 pp | 1 / 1 / 1 | 1 / 1 / 1 |
| -1.0 pp | 1 / 1 / 1 | 2 / 2 / 2 |
| -0.5 pp | 1 / 1 / 1 | 2 / 2 / 2 |
| +0.0 pp | 1 / 1 / 1 | 3 / 3 / 3 |
| +0.5 pp | 1 / 2 / 2 | 6 / 6 / 12 |
| +1.0 pp | 2 / 2 / 2 | never / never / never |
| +1.5 pp | 3 / 6 / 6 | never / never / never |

## POST-HOC: common absolute targets (seeds 0/1/2)

| target | SG | hybrid |
|---|---|---|
| 95.0 % | 1 / 1 / 1 | 1 / 1 / 1 |
| 96.0 % | 1 / 1 / 1 | 2 / 1 / 1 |
| 97.0 % | 1 / 1 / 1 | 2 / 2 / 2 |
| 97.5 % | 1 / 1 / 1 | 6 / 3 / 2 |
| 98.0 % | 2 / 2 / 2 | 12 / 6 / 12 |
| 98.5 % | 3 / 2 / 2 | never / never / never |
| 99.0 % | 12 / 6 / 6 | never / never / never |

## POST-HOC: within-arm iso-accuracy (own T = 24 accuracy − margin; seeds 0/1/2)

| margin | SG | hybrid |
|---|---|---|
| 2.0 pp | 1 / 1 / 1 | 2 / 2 / 1 |
| 1.0 pp | 2 / 2 / 2 | 2 / 3 / 2 |
| 0.5 pp | 3 / 3 / 2 | 6 / 6 / 2 |

## Conversion (validation, before fine-tuning; mean of 3 seeds)

| T | gate outcome | ANN val | converted val | agreement | rate corr. lif1/2/3 | after 1 FT epoch | after FT (best) |
|---|---|---|---|---|---|---|---|
| 1 | sanity_pass (strict pass 0/3) | 97.48 | 70.32 | 70.73 | 0.87 / 0.76 / 0.71 | 93.98 | 96.38 |
| 2 | sanity_pass (strict pass 0/3) | 97.48 | 87.63 | 88.08 | 0.93 / 0.84 / 0.82 | 95.66 | 97.50 |
| 3 | sanity_pass (strict pass 0/3) | 97.48 | 87.36 | 87.89 | 0.94 / 0.87 / 0.85 | 95.81 | 97.81 |
| 6 | sanity_pass (strict pass 0/3) | 97.48 | 90.39 | 90.80 | 0.96 / 0.89 / 0.87 | 96.63 | 98.17 |
| 12 | sanity_pass (strict pass 0/3) | 97.48 | 87.84 | 88.14 | 0.94 / 0.88 / 0.84 | 96.81 | 98.29 |
| 24 | sanity_pass (strict pass 0/3) | 97.48 | 82.29 | 82.52 | 0.92 / 0.85 / 0.80 | 96.51 | 98.38 |

## Training cost (as used; M sample-time-steps, mean of 3 seeds) and wall-clock (T4)

| T | SG budget | hybrid budget (ANN + FT) | SG wall (h) | hybrid FT wall (h) |
|---|---|---|---|---|
| 1 | 1.44 | 2.67 (1.44 + 1.23) | 0.52 | 0.44 |
| 2 | 2.75 | 4.19 (1.44 + 2.75) | 0.51 | 0.51 |
| 3 | 4.32 | 5.66 (1.44 + 4.22) | 0.54 | 0.54 |
| 6 | 8.64 | 9.70 (1.44 + 8.26) | 0.59 | 0.57 |
| 12 | 17.28 | 16.03 (1.44 + 14.59) | 0.75 | 0.63 |
| 24 | 30.34 | 31.78 (1.44 + 30.34) | 1.44 | 1.44 |

ANN training wall-clock: 0.56 h per seed. Total GPU time per seed: seed 0 9.4 h, seed 1 9.5 h, seed 2 8.6 h

## Inference operations per test sample (mean of 3 seeds)

| T | SG SynOps (M) | hybrid SynOps (M) | SG E/E_ANN | hybrid E/E_ANN | SG rate lif1/2/3 | hybrid rate lif1/2/3 |
|---|---|---|---|---|---|---|
| 1 | 2.36 | 2.76 | 0.10 | 0.12 | 0.099 / 0.052 / 0.407 | 0.117 / 0.056 / 0.132 |
| 2 | 4.49 | 5.24 | 0.18 | 0.20 | 0.093 / 0.055 / 0.479 | 0.112 / 0.049 / 0.114 |
| 3 | 7.17 | 7.95 | 0.27 | 0.29 | 0.100 / 0.057 / 0.458 | 0.113 / 0.052 / 0.120 |
| 6 | 12.38 | 18.52 | 0.44 | 0.62 | 0.084 / 0.057 / 0.442 | 0.139 / 0.051 / 0.103 |
| 12 | 21.16 | 34.80 | 0.71 | 1.12 | 0.071 / 0.055 / 0.408 | 0.131 / 0.048 / 0.099 |
| 24 | 35.91 | 66.17 | 1.17 | 2.07 | 0.060 / 0.046 / 0.362 | 0.125 / 0.045 / 0.102 |

ANN reference: 6,608,384 MACs per sample.

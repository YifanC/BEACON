# 6D optimization diagnosis and BO_TR2 outcome

## BO_TR2 outcome (final)

- The diagnosis below was performed on the frozen clean 272-row pre-BO_TR2 snapshot. It motivated selecting Model C (Matérn-3/2) and launching BO_TR2; it is retained as scientific provenance rather than rewritten as a post-hoc diagnosis.
- BO_TR2 Slurm job `34681675` completed successfully (`COMPLETED`, `ExitCode 0:0`) with 60 new simulator evaluations.
- Final clean lineage: 72 INITIAL + 100 BO_1 + 100 BO_TR + 60 BO_TR2 = 332 rows. Snapshot SHA-256: `3599ee904c69654e8ec9da5404669db32ba13345b4205b1be1d16d9236a1fafd`.
- BO_TR2 best LLHD: **21026.755859375**. Held-out nominal reference: **21028.986328125**.
- Signed gap (BO_TR2 best − nominal): **−2.23046875 LLHD**; absolute gap: **2.23046875 LLHD**; closure category: **STRONG**.
- Fresh validation job `34682537` completed successfully (`COMPLETED`, `ExitCode 0:0`): 16 global + 16 local Sobol points (seed 20260827), plus six direct profiles. None entered training.
- Final Matérn-3/2 model metrics: held-out score RMSE 0.1470, MAE 0.1079, Spearman ρ 0.9872, 68%/95% coverage 0.8438/0.9688; five-fold CV score RMSE 0.1454, MAE 0.0464, Spearman ρ 0.9947, 68%/95% coverage 0.8373/0.9488.
- All canonical final health checks pass. Nominal is a held-out reference, not a mathematically proven global minimum.

## Historical pre-BO_TR2 diagnosis

## Executive summary

- Clean 272-row snapshot SHA-256: `71de4ad327cea079edda1fdb5177522cbf229a6452b9d83873fc47fbf7baeee7` (INITIAL 72, BO_1 100, BO_TR 100).
- Actual best BO_TR LLHD = 21099.2031; all-nominal LLHD = 21028.9863; absolute gap = 70.2168 LLHD (BO_TR above nominal).
- Nominal-closure category: **INCOMPLETE** (gap > 20).

## 1. Nominal standardized-residual (corrected)

| quantity | value |
|---|---|
| actual score at nominal | -9.953657 |
| Model A predicted score at nominal | -9.975864 |
| Model A score SD at nominal | 0.006423 |
| score residual (pred − actual) | -0.022207 |
| standardized residual (z) | -3.4574 |
| nominal inside ±1σ ? | **False** |
| nominal inside ±1.96σ ? | **False** |

The earlier text claiming '~1.7σ' was wrong: the correct z = -3.4574. Model A is dramatically overconfident at nominal — it sits far outside even the 95% interval.

## 2. Ranking

- Actual score at BO_TR best = -9.956991; at nominal = -9.953657. Nominal score is HIGHER (nominal LLHD is LOWER).
- Model A predicted score at BO_TR best = -9.957141; at nominal = -9.975864. Model A predicts BO_TR best has the higher score (nominal has the lower score in the model).
- Actual ranking: **nominal better** (lower LLHD).
- GP ranking: **BO_TR best better** according to Model A.
- **LOCAL RANK REVERSAL** — the previous 'correctly ranks BO_TR best above nominal' sentence was factually wrong; the GP orders these two points in the opposite direction from the real simulator.

## 3. Training support along the bridge (normalized Euclidean)

| t | d_nn | d_2nd | d_5th | d_10th | n<0.01 | n<0.025 | n<0.05 | n<0.10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.000 | 0.0000 | 0.0014 | 0.0030 | 0.0041 | 18 | 25 | 78 | 88 |
| 0.125 | 0.0065 | 0.0068 | 0.0069 | 0.0073 | 17 | 22 | 71 | 88 |
| 0.250 | 0.0118 | 0.0118 | 0.0135 | 0.0139 | 0 | 19 | 58 | 88 |
| 0.375 | 0.0182 | 0.0183 | 0.0203 | 0.0210 | 0 | 17 | 45 | 87 |
| 0.500 | 0.0250 | 0.0252 | 0.0274 | 0.0282 | 0 | 0 | 36 | 87 |
| 0.625 | 0.0321 | 0.0323 | 0.0346 | 0.0354 | 0 | 0 | 26 | 87 |
| 0.750 | 0.0392 | 0.0394 | 0.0418 | 0.0426 | 0 | 0 | 21 | 86 |
| 0.875 | 0.0464 | 0.0466 | 0.0491 | 0.0498 | 0 | 0 | 13 | 83 |
| 1.000 | 0.0535 | 0.0538 | 0.0562 | 0.0567 | 0 | 0 | 0 | 81 |

Nearest training-observation distance to nominal (t=1) = **0.0535** in unit space; 5th-nearest = 0.0562; 10th-nearest = 0.0567; n≤0.10 = 81.

The first bridge t with zero training rows inside r=0.05 is t = 1.0; from that point onward the GP is unsupported at short-range.

## 4-6. Fair A-vs-C shootout

| metric | Model A (M-5/2) | Model C (M-3/2) |
|---|---:|---:|
| n_train | 272 | 272 |
| bridge score RMSE | 0.0099 | 0.0121 |
| bridge score MAE | 0.0082 | 0.0099 |
| bridge median |resid| | 0.0072 | 0.0087 |
| bridge worst |resid| | 0.0177 | 0.0216 |
| bridge Spearman | -1.000 | -1.000 |
| bridge slope agreements (8 intervals) | 1 | 1 |
| shell score RMSE | 0.1146 | 0.0807 |
| shell Spearman | 0.965 | 0.958 |
| combined-local RMSE | 0.0913 | 0.0646 |
| held-out global RMSE (n=16) | 0.2224 | 0.1499 |
| held-out global Spearman | 0.944 | 0.947 |
| held-out local RMSE (n=16) | 0.1836 | 0.1750 |
| held-out local Spearman | 0.897 | 0.929 |
| held-out combined (n=32) RMSE | 0.2040 | 0.1629 |
| held-out combined Spearman | 0.975 | 0.979 |
| held-out combined 68/95 coverage | 0.88/0.94 | 0.81/0.91 |
| 5-fold CV score RMSE | 0.1472 | 0.1592 |
| 5-fold CV Spearman | 0.996 | 0.995 |
| 5-fold CV 68/95 coverage | 0.78/0.89 | 0.78/0.92 |
| bridge slope predicted (score/Δt) | -0.0187 | -0.0237 |
| bridge slope actual (score/Δt) | +0.0034 | +0.0034 |
| GP thinks nominal better than BO_TR best? | False | False |

### Comparative selection rule (each check)

- combined_local_improv_>=0.20: **PASS** (measured value +0.2923)
- ho_all_degrade_<=0.10: **PASS** (measured value -0.2011)
- ho_all_rho_drop_<=0.02: **PASS** (measured value -0.0044)
- cv_score_degrade_<=0.10: **PASS** (measured value +0.0815)

**Selected surrogate: MODEL C.** Every comparative rule was satisfied; Matérn-3/2 improves both local RMSE and does not degrade global HO metrics or CV.

## 7. Score-scale context

| subset | n | score min | score max | score SD | score range |
|---|---:|---:|---:|---:|---:|
| all 272 | 272 | -13.7674 | -9.9570 | 1.1072 | 3.8105 |
| best 100 | 100 | -9.9866 | -9.9570 | 0.0078 | 0.0296 |
| best 50 | 50 | -9.9598 | -9.9570 | 0.0009 | 0.0028 |
| best 25 | 25 | -9.9580 | -9.9570 | 0.0004 | 0.0010 |

Actual score gain BO_TR-best → nominal = +0.003333. As a fraction of the score SD:
- all-272 SD (1.1072): 0.0030
- best-100 SD (0.0078): 0.4279
- best-50 SD (0.0009): 3.7163
- best-25 SD (0.0004): 9.4316

Model A predicted score difference (best − nominal) = +0.018722; Model C = +0.023515.

## 8. Acquisition discovery test (200k Sobol candidates in a length-0.20 trust region around BO_TR best; no simulator)

| quantity | Model A | Model C |
|---|---:|---:|
| nominal inside trust region? | True | True |
| top qLogEI value | -10.3807 | -7.2600 |
| qLogEI at nominal | -13.0209 | -9.9582 |
| rank of nominal (out of 200 000) | 86 | 231 |
| any top-100 candidate closer to nominal than BO_TR best is? | True | True |
| median top-10 distance from BO_TR best (unit) | 0.1125 | 0.1011 |
| median top-10 distance from nominal (unit) | 0.0936 | 0.1125 |

### qLogEI rank along the bridge (out of 200 000)

| t | Model A qLogEI | Model A rank | Model C qLogEI | Model C rank |
|---:|---:|---:|---:|---:|
| 0.000 | -9.6760 | 0 | -8.9150 | 54 |
| 0.125 | -9.7513 | 0 | -8.4718 | 23 |
| 0.250 | -10.1119 | 0 | -8.3826 | 20 |
| 0.375 | -10.5942 | 1 | -8.5103 | 24 |
| 0.500 | -11.1047 | 10 | -8.7286 | 35 |
| 0.625 | -11.6156 | 29 | -8.9897 | 63 |
| 0.750 | -12.0600 | 57 | -9.2817 | 109 |
| 0.875 | -12.5771 | 77 | -9.5992 | 159 |
| 1.000 | -13.0209 | 86 | -9.9582 | 231 |

## Primary diagnosed failure mode

**SURROGATE_LOCAL_FIT (kernel choice — Matérn-5/2 too smooth for this basin)**

## Recommendation

**OPTION A — SURROGATE FIX.**

Model C passes all four comparative selection rules: combined-local RMSE improved by -29.2% ≥ 20%; held-out combined RMSE degraded by -20.1% ≤ 10%; Spearman dropped by -0.0044 ≤ 0.02; CV degraded by +8.2% ≤ 10%.

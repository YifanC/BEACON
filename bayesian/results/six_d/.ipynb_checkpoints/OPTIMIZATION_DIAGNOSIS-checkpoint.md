# 6D BO_1 optimization diagnosis

This diagnosis uses NO simulator calls. It reconstructs the clean 72+100=172-row baseline, replays the BO_1 GP at six iteration checkpoints, audits acquisition-optimizer stability, measures GP accuracy on the retained old diagnostic evaluations (DIRECT/INTERACTION/BO_2/heldout/reference/profile), measures training geometry, characterises boundary behaviour, and compares three GP model variants offline.

## Clean baseline

- SHA-256: `d47f108da8b9a169bc07670a1fec3300bfd899ca7a75e431bc3ba9a5465ead3c`
- Rows: INITIAL 72 + BO_1 100 = 172. Best INITIAL LLHD = 44702.832, best BO_1 LLHD = 21484.777.

## A. Replay of BO_1 model states

| state | n_train | best LLHD | rank of nominal by posterior mean | rank of nominal by qLogEI | top-100 qLogEI boundary frac | top-100 median dist from best (unit) |
|---|---:|---:|---:|---:|---:|---:|
| INITIAL_only | 72 | 44702.832 | 1014 / 100000 | 1014 / 100000 | 0.01 | 0.402 |
| BO_1_iter_10 | 82 | 36026.977 | 652 / 100000 | 652 / 100000 | 0.00 | 0.598 |
| BO_1_iter_25 | 97 | 30066.584 | 567 / 100000 | 567 / 100000 | 0.00 | 0.538 |
| BO_1_iter_50 | 122 | 28141.230 | 57 / 100000 | 57 / 100000 | 0.00 | 0.597 |
| BO_1_iter_75 | 147 | 27643.529 | 1 / 100000 | 1 / 100000 | 0.00 | 0.659 |
| BO_1_iter_100 | 172 | 21484.777 | 0 / 100000 | 0 / 100000 | 0.02 | 0.425 |

At every state the pool contains 100 000 scrambled Sobol candidates in unit space. Rank counts how many pool points have a higher score (posterior mean) or higher qLogEI than the nominal coordinate.

## B. Acquisition optimizer audit at BO_1 iter 100

- Historic settings (`num_restarts=24, raw_samples=2048`): qLogEI = 1.333001, boundary = False.
- Strong settings (`num_restarts=64, raw_samples=8192`): qLogEI = 1.333001, boundary = False.
- Optimizer instability: no (Δ = +0.0000).

## C. GP accuracy on retained old diagnostic data

(Model = clean BO_1 GP trained on 172 rows. Diagnostic data are OLD DIRECT+INTERACTION+BO_2+heldout+reference+profile evaluations, held OUT of GP training here.)

| region | n | score RMSE | score MAE | Spearman ρ | 68% cov | 95% cov |
|---|---:|---:|---:|---:|---:|---:|
| global (unit dist > 0.20) | 146 | 0.3523 | 0.2525 | 0.872 | 0.35 | 0.50 |
| local (unit dist ≤ 0.20) | 160 | 0.1598 | 0.1066 | 0.560 | 0.26 | 0.43 |

## D. Training geometry (normalized Euclidean)

- Nearest-neighbour distance across 172 rows: median 0.185, p10 0.015, p90 0.492.
- Observations within normalized radius r of BO_1 best: r=0.05 → 1; r=0.10 → 11; r=0.20 → 21; r=0.30 → 23.
- Overall boundary fraction (any axis within 1e-3 of 0 or 1 in unit space): 0.43.

## E. Boundary behaviour of BO_1 proposals

- BO_1 boundary fraction (any axis at 1e-3 of 0 or 1): **0.74**.
- Per-axis boundary hits (BO_1 only):
  - Ab: 3
  - kb: 0
  - eField: 0
  - lifetime: 2
  - tran_diff: 5
  - long_diff: 72
- Boundary vs interior mean LLHD: 34006.7  vs  22953.0.
- Improvements of running best: boundary 28 times, interior 12 times.

## Model audit (A vs B vs C)

Candidate A: production Matérn-5/2 ARD(6) + Standardize(m=1) + `train_Yvar = 1/LLHD²` (fixed).  
Candidate B: same but no fixed `train_Yvar`; one homoskedastic likelihood noise inferred.  
Candidate C: same as A but Matérn-3/2.  

| variant | all RMSE(score) | all ρ | local RMSE | local ρ | local 68% cov | global RMSE | global ρ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.2694 | 0.880 | 0.1598 | 0.560 | 0.26 | 0.3523 | 0.872 |
| B | 0.2660 | 0.881 | 0.1493 | 0.557 | 0.33 | 0.3519 | 0.863 |
| C | 0.2595 | 0.883 | 0.1450 | 0.561 | 0.33 | 0.3437 | 0.861 |

## Primary diagnosis

Primary classification: ACQUISITION_POLICY, GLOBAL_BOUNDARY_EXPLORATION, INSUFFICIENT_LOCAL_DATA.

**Numerical evidence:**

- Final-state qLogEI top-100 candidates: 2% touch a boundary; median unit-space distance from best = 0.425.
- Only 11 of 172 rows lie within normalized radius 0.10 of the final BO_1 best (a small local sample).
- BO_1 boundary fraction = 0.74; boundary vs interior mean LLHD = 34007 vs 22953 — boundary points are on average worse but were repeatedly proposed because of high posterior uncertainty there.
- Acquisition-optimizer stability: strong-vs-historic Δ qLogEI = +0.0000.
- Local vs global GP accuracy (RMSE(score)): 0.160 local vs 0.352 global.

## GP model selection

Selected variant: **A** — no other candidate showed a clear, repeatable improvement in LOCAL predictive accuracy without materially degrading global accuracy. Keep the production surrogate.

## Improved strategy

Given the evidence — global qLogEI proposes boundary/exploratory points because the BO_1 GP has high posterior uncertainty far from the observed cluster, and the local neighbourhood of the running best is sparsely sampled — the improved optimizer is:

  **adaptive trust-region qLogExpectedImprovement (BO_TR)**

Trust-region bounds are constructed from the BO_TR-current best (never nominal), sized by ARD lengthscales, and expand/shrink on running-best improvement / stagnation. All acquisition remains genuine qLogEI over the trust-region rectangle — no grid, no coordinate descent, no gradient on the simulator.

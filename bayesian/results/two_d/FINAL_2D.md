# 2D BO — final results

## Physical dataset

- 999.93-cm safe dataset: 176 events, 100 010 HDF5 rows, 999.926642 cm physical track, one simulator batch, 9 368 target hits, target seed 0.
- Target NPZ: `.local/two_d/target.npz` (SHA-256 recorded at generation).

## Method (same for all three pairs)

- Normalized inputs in [0, 1]².
- SingleTaskGP with Matérn-5/2 ARD(2), Standardize(m=1), `train_Yvar = 1 / native_LLHD²`.
- score = -ln(native_LLHD); qLogExpectedImprovement (q=1) with `optimize_acqf` (24 restarts, 2048 raw samples).
- Post-BO direct 1D simulator scans with the other parameter held at the BO best.

## Final evaluated points (predeclared consistency rule: BO-B for every pair)

| Pair | eField/Ab/tran_diff | lifetime/kb/long_diff | Best native LLHD |
| --- | --- | --- | --- |
| Ab + kb | Ab = 0.80 | kb = 0.0486 | 21 029.006 |
| eField + lifetime (BO-B reproduction) | eField = 0.4999921690 | lifetime = 2 209.32 µs | 21 024.980 |
| Diffusion (tran_diff + long_diff) | tran_diff ≈ 8.85 × 10⁻⁶ cm²/µs | long_diff ≈ 3.91 × 10⁻⁶ cm²/µs | 21 025.762 |

## Files

- `Ab_kb/{01…05}_*.png`, `eField_lifetime/{01…05}_*.png`, `diffusion/{01…05}_*.png` — five presentation plots per pair.
- `PLOT_GUIDE.md` — global plot-status + provenance table.
- Per-pair `PLOT_GUIDE.md` — beginner-friendly walkthrough of each plot.

## Reproduction

Everything can be regenerated with `workflows/two_d/finalize.py --config <pair>.yaml` from the histories and direct scans stored under `.local/two_d/`.

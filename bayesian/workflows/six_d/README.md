# Reusable 6D Bayesian-optimization workflow

## Scientific purpose

Joint optimization of six detector parameters — Ab, kb, eField, lifetime, tran_diff, long_diff — on the 999.93-cm safe dataset target (target seed 0, 176 events, 100 010 HDF5 rows, 9 368 hits).

Method (identical to the validated 2D repair, extended to 6D):

- Normalized inputs in [0, 1]⁶.
- SingleTaskGP, Matérn-5/2 ARD(6), Standardize(m=1).
- score = -ln(native_LLHD); `train_Yvar = 1 / native_LLHD²`.
- qLogExpectedImprovement (q=1) with `optimize_acqf`.
- 72 Sobol init + 100 BO iterations, resumable via `checkpoint_latest.pt`.

## Files

- `build_6d.py` — shared constants, GP fit, objective factory.
- `build_initial_design.py` — 72 Sobol initial evaluations.
- `run_bo.py` + `run_bo.sbatch` — 100-iteration BO.
- `continue_bo.py` + `continue_bo.sbatch` — resumable continuation. Writes into the same `.local/six_d/current/raw/continuation/` tree; does **not** create a new top-level directory per continuation.
- `validate.py` + `validate.sbatch` — one 1D direct simulator scan per axis at the BO best.
- `finalize.py` — final five plots + PLOT_GUIDE.md + FINAL_6D.md + health scorecard.
- `config.yaml` — records bounds, seeds, `.local` state root.

## Input

- Shared target NPZ at `../../.local/two_d/target.npz`.
- 2D direct-scan CSVs at `../../.local/two_d/<pair>/direct_profile_parameter{1,2}.csv` provide the grid values reused for 6D validation.

## Output

- Resumable state under `../../.local/six_d/current/raw/`:
  - `initial_design/`, `traces/`, `checkpoints/`, `simulator/`, `logs/`, `config/`, `continuation/`, `direct_validation/`.
- Final GitHub-safe results in `../../results/six_d/`:
  - `plots/{01_convergence_and_efficiency, 02_parallel_coordinates, 03_pairwise_observation_matrix, 04_gp_cross_validation, 05_direct_validation_scorecard}.png`
  - `FINAL_6D.md`, `PLOT_GUIDE.md`, `health_metrics.json`, `README.md`.

## Run

Fresh run from scratch (only after archiving any previous `.local/six_d/current/`):

```
sbatch optimize/bayesian/workflows/six_d/run_bo.sbatch
```

Continue an interrupted run:

```
sbatch optimize/bayesian/workflows/six_d/continue_bo.sbatch
```

Validate at the BO best:

```
sbatch optimize/bayesian/workflows/six_d/validate.sbatch
```

Finalize (deterministic, no simulator, no Slurm needed):

```
python optimize/bayesian/workflows/six_d/finalize.py
```

## What belongs in `.local/six_d/current/`

- The current resumable history (`raw/traces/bo_6d_history.csv`).
- All numbered and latest checkpoints (`raw/checkpoints/*.pt`).
- Simulator per-call JSON (`raw/simulator/*.json`).
- Direct-validation raw CSV rows (`raw/direct_validation/*.csv`).
- Current continuation subtree (`raw/continuation/`).
- Slurm logs (`raw/logs/*.out`, `raw/continuation/logs/*.out`, `raw/direct_validation/logs/*.out`).

## What belongs in `results/six_d/`

- The five final PNGs above, `PLOT_GUIDE.md`, `FINAL_6D.md`, `health_metrics.json`, `README.md`.
- Nothing larger than a few hundred KB.

## Regenerating the final plots

`finalize.py` is deterministic. Delete `results/six_d/plots/*.png` and re-run.

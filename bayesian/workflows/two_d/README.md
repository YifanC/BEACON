# Reusable 2D BO workflow

## Scientific purpose

One reusable Bayesian-optimization pipeline for the three validated 2D parameter pairs:

- Ab + kb
- eField + lifetime
- diffusion (tran_diff + long_diff)

Method (all pairs):

- Normalized inputs.
- SingleTaskGP, Matérn-5/2 ARD(2), Standardize(m=1).
- score = -ln(native_LLHD); `train_Yvar = 1 / native_LLHD²`.
- qLogExpectedImprovement (q=1) with `optimize_acqf` (24 restarts, 2048 raw samples).
- 16 scrambled Sobol init + 40 BO iterations.
- Post-BO direct 1D simulator scans, other parameter fixed at BO best.

## Files

- `run_bo.py` + `run_bo.sbatch` — BO runner.
- `validate.py` + `validate.sbatch` — post-BO direct simulator scans.
- `finalize.py` — reads the history + scans, fits the final GP, writes the five presentation plots + compact summaries.
- `configs/{Ab_kb,eField_lifetime,diffusion}.yaml` — the only per-pair customization.

## Input

- The shared 2D target NPZ at `../../.local/two_d/target.npz`.
- Config file chosen via `--config workflows/two_d/configs/<pair>.yaml`.

## Output

- `../../.local/two_d/<pair>/history.csv, checkpoints/*, simulator/*, trace_latest.npz, summary.json, direct_profile_parameter{1,2}.csv, direct_profile_summary.json`.
- `../../results/two_d/<pair>/{01..05}_*.png, summary.csv, summary.json`.

## Run

```
CONFIG=optimize/bayesian/workflows/two_d/configs/eField_lifetime.yaml \
  sbatch optimize/bayesian/workflows/two_d/run_bo.sbatch
```

Then, after the BO job completes:

```
CONFIG=optimize/bayesian/workflows/two_d/configs/eField_lifetime.yaml \
  sbatch optimize/bayesian/workflows/two_d/validate.sbatch
```

Then, on any node with matplotlib/torch/botorch available:

```
python optimize/bayesian/workflows/two_d/finalize.py \
    --config optimize/bayesian/workflows/two_d/configs/eField_lifetime.yaml
```

## What belongs in `.local/two_d/`

- The shared target NPZ (`target.npz`) — used by all pairs.
- Per-pair histories, checkpoints, simulator JSON, direct-scan CSVs.

## What belongs in `results/two_d/`

- Per-pair five final PNGs, `summary.csv`, `summary.json`.
- One pair-agnostic `README.md`, `PLOT_GUIDE.md`, `FINAL_2D.md`.

## Regenerating the final plots

After the BO and validation runs have finished, `finalize.py` is deterministic:
delete the plots and re-run.

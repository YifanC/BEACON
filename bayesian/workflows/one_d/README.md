# One-dimensional eField scan analysis

## Scientific purpose

Analyze the completed 1D eField scan (native LLHD across scanned values, with all other detector parameters held at nominal). Produce a single presentation-ready plot plus compact CSV/JSON summaries.

## Optimized parameter

- `eField` (kV/cm), nominal 0.5, search range [0.49, 0.51].

## Authoritative script

- `run_scan_analysis.py` — analysis-only, reads one PKL and writes three files.

## Input

- `../../.local/one_d/eField_scan_history.pkl` — the completed scan history.
- `config.yaml` — points to the PKL and records parameter bounds.

## Output

- `../../results/one_d/scan.png`
- `../../results/one_d/scan_summary.csv`
- `../../results/one_d/scan_summary.json`

## Run

```
python optimize/bayesian/workflows/one_d/run_scan_analysis.py \
    --config optimize/bayesian/workflows/one_d/config.yaml
```

or inside a Slurm job:

```
sbatch optimize/bayesian/workflows/one_d/run_scan_analysis.sbatch
```

## What belongs in `.local/one_d/`

- The single authoritative scan history PKL. This is the raw data; it is not committed.

## What belongs in `results/one_d/`

- The three files above only. No PKL, no NPZ, no checkpoints.

## Regenerating the final plot

Delete `results/one_d/scan.png` (if any) and re-run `run_scan_analysis.py`.

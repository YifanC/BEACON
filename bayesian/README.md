# Bayesian calibration workflows for larnd-sim-jax

This directory holds all custom research code for detector-parameter calibration on `larnd-sim-jax`. It is intentionally small and reusable: three workflows (`one_d`, `two_d`, `six_d`), one presentation tree (`results/`), and one machine-local artifact tree (`.local/`).

## Purpose of each workflow

- `workflows/one_d/` — analyze the completed 1D eField scan and produce one plot plus one compact CSV/JSON summary.
- `workflows/two_d/` — reusable 2D Bayesian optimization for the three validated pairs (Ab + kb, eField + lifetime, diffusion). One BO runner + one validator + one finalizer + three tiny pair configs.
- `workflows/six_d/` — reusable joint 6D BO with resumable continuation. One initial-design builder + one BO runner + one continuation runner + one validator + one finalizer.

## Final directory structure

```
optimize/bayesian/
├── README.md
├── .gitignore
├── CLEANUP_REPORT.md
├── workflows/
│   ├── common.py
│   ├── one_d/
│   ├── two_d/
│   └── six_d/
├── results/                       (final, GitHub-safe artifacts)
│   ├── one_d/
│   ├── two_d/{Ab_kb,eField_lifetime,diffusion}/
│   └── six_d/{plots/, ...}
└── .local/                        (HPC-only, intentionally not committed)
    ├── shared/
    ├── one_d/
    ├── two_d/{target.npz, <pair>/}
    └── six_d/current/raw/{initial_design,traces,checkpoints,simulator,logs,config,direct_validation,continuation}
```

## Running each workflow

- 1D scan analysis:
  ```
  python optimize/bayesian/workflows/one_d/run_scan_analysis.py \
      --config optimize/bayesian/workflows/one_d/config.yaml
  ```
- 2D BO for a pair (Slurm):
  ```
  CONFIG=optimize/bayesian/workflows/two_d/configs/eField_lifetime.yaml \
    sbatch optimize/bayesian/workflows/two_d/run_bo.sbatch
  ```
  then `validate.sbatch`, then locally:
  ```
  python optimize/bayesian/workflows/two_d/finalize.py \
      --config optimize/bayesian/workflows/two_d/configs/eField_lifetime.yaml
  ```
- 6D BO (Slurm):
  ```
  sbatch optimize/bayesian/workflows/six_d/run_bo.sbatch
  sbatch optimize/bayesian/workflows/six_d/continue_bo.sbatch     # resumable
  sbatch optimize/bayesian/workflows/six_d/validate.sbatch
  python optimize/bayesian/workflows/six_d/finalize.py
  ```

## Where local HPC artifacts go

Everything simulator-touching (targets, per-run traces, checkpoints, simulator JSON, direct-scan CSVs, Slurm logs) lives under `.local/<workflow>/`. `.local/` is machine-local — see `.gitignore` — and is used to resume work in-place instead of creating a new experiment directory per invocation.

## Where final GitHub-safe results go

Everything under `results/` is small and static: presentation-quality PNGs, per-workflow READMEs, plot guides, final Markdown reports, and compact CSV/JSON summaries. No PKL, no NPZ, no PT, no HDF5, no full Slurm logs.

## Required container

All workflows run inside the pinned project container:

```
/sdf/group/neutrino/pgranger/larnd-sim-jax.sif
```

with a `PYTHONPATH` that includes the machine-local site-packages tree at `optimize/bayesian/.local/shared/site-packages` (staged separately; it is not committed).

## Required external data

The simulator input HDF5 is external:

```
/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/production_884072/job_23771825_0000/output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_costheta0.966_range_0.05cm.h5
```

The 2D target NPZ used by every workflow lives at `.local/two_d/target.npz`.

## `.local/` is not committed

The `.local/` tree is HPC-only. Nothing there is included in a normal `git add`. See `.gitignore`.

## 4D work was intentionally removed

Every 4D directory, checkpoint, history, plot, and script has been deleted permanently as part of the reorganization documented in `CLEANUP_REPORT.md`. New experiments must reuse `workflows/one_d`, `workflows/two_d`, or `workflows/six_d`.

## Adding new experiments

Do not create a new top-level directory per invocation. Add a new pair config under `workflows/two_d/configs/` for a new 2D pair, or edit `workflows/six_d/config.yaml`. All state lives in `.local/<workflow>/`; all final outputs land in `results/<workflow>/`.

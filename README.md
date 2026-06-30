# BEACON
BEACON: Bayesian Error And Calibration Optimization for Neutrinos.

## Layout

```
BEACON/
├── examples/                 # didactic BO walk-throughs
│   ├── bo_kinematics.ipynb   # toy kinematic-fit BO (BoTorch)
│   └── bo_kinematics.py      # standalone-script form of the notebook
├── detector_calibration/     # realistic BO over larnd-sim-jax detector params
│   ├── forward.py            # programmatic forward sim + MMD loss wrapper
│   ├── generate_target.py    # one-shot: nominal-params target snapshot
│   ├── bo_detector.py        # BoTorch SingleTaskGP + qLogEI driver
│   └── slurm/
│       ├── run_target.sh
│       └── run_bo.sh
└── configs/
    └── bo_detector.yaml      # input file, sim opts, param bounds, BO knobs
```

## Detector-calibration BO

Tunes 6 parameters (`Ab`, `kb`, `eField`, `lifetime`, `long_diff`, `tran_diff`)
against a synthetic target produced by larnd-sim-jax at nominal values. Both
target generation and BO iterations import larnd-sim-jax as a Python module,
so they must run inside the apptainer image. The `--noise` and
`--diffusion_in_current_sim` flags are baked into the YAML config.

```bash
# Ground-truth target (one-shot, ~minutes on an A100):
sbatch detector_calibration/slurm/run_target.sh

# BO loop (default: 16 Sobol seeds + 60 acquisitions + L-BFGS-B polish):
sbatch detector_calibration/slurm/run_bo.sh
```

Override defaults at submit time via env vars:
```bash
sbatch --export=ALL,CONFIG=/path/to/my_config.yaml detector_calibration/slurm/run_bo.sh
```

Forward simulation, target, and BO settings all live in
`configs/bo_detector.yaml`. Param bounds follow the `min`/`max` columns of
[`optimize/ranges.py`](https://github.com/NuLads/larnd-sim-jax/blob/main/optimize/ranges.py).
Loss is `GenericLossStrategy` wrapping
[`mse_adc`](https://github.com/NuLads/larnd-sim-jax/blob/main/src/larndsim/losses_jax.py)
(misnamed — it is an MMD + relative-charge loss).

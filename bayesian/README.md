# Bayesian calibration workflows

This directory calibrates six `larnd-sim-jax` detector-response parameters against a fixed simulated target: Birks amplitude `Ab`, Birks constant `kb`, electric field, electron lifetime, transverse diffusion, and longitudinal diffusion. The native LLHD is minimized; the GP models `-ln(LLHD)`, so acquisition maximizes its score.

## Layout

- `workflows/` contains reusable objectives, GP infrastructure, and the 1D, 2D, and 6D workflows.
- `batch_study/` contains the repeat studies, the large one-batch study, launchers, and analysis.
- `results/` contains accepted summaries and figures.
- `.local/` contains retained production histories, checkpoints, targets, and validation provenance for the original studies.

The simulator implementation remains in `/sdf/home/i/iatif/larnd-sim-jax`; it is imported rather than copied here.

## Canonical adaptive 6D optimization

The production sequence is 72 scrambled Sobol observations, 100 global qLogEI observations, 100 six-dimensional trust-region observations, and a 60-observation Matérn-3/2 trust-region refinement (332 total). Nominal is a held-out closure reference and is never supplied to optimizer training. BO uses objective values, not simulator gradients.

Submit a configured optimizer repeat from a login node; the batch-study README
lists the required variables:

```bash
sbatch batch_study/run_optimizer_repeat.sbatch
```

`BAYESIAN_OPTIMIZER_SEED` controls the scrambled Sobol design and all optimizer-side
Torch, NumPy, BoTorch acquisition, and trust-region randomness. Target, data, and
candidate-simulator seeds remain fixed across optimizer repeats.

Targets are frozen NPZ files. Candidate and target must use identical tracks, detector configuration, LUT, and fixed-seed convention. Keep a long workload as one logical batch: ordinary dataset multi-batching changes the nonlinear LLHD rather than merely reducing memory.

Large one-batch jobs require the upstream signal and probabilistic pixel blocking paths. Launchers set `LARNDSIM_MEMORY_EFFICIENT_SIGNALS`, `LARNDSIM_SIGNAL_CONTRIB_BLOCK_SIZE`, `LARNDSIM_PROBABILISTIC_BLOCK_SIZE`, and disable JAX preallocation. Blocking changes floating-point reduction order by a few ulps; validated LLHD differences are negligible.

## Analysis and interpretation

Regenerate supported batch-study figures without running the simulator:

```bash
singularity exec --nv --bind /sdf:/sdf /sdf/group/neutrino/pgranger/larnd-sim-jax.sif \
  python3 batch_study/regenerate_plots.py
```

Plots and summaries are written inside each experiment. Five-fold GP cross-validation tests surrogate reconstruction of saved observations; it is not independent detector validation, and GP uncertainty is model uncertainty. Adaptive BO does not establish a global optimum or a physically superior calibration to nominal.

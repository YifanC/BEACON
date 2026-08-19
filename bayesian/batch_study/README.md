# Physical-length study

This directory preserves repeated optimizer executions and the memory-qualified 2000 cm workflow.

- `200cm_seed0/` and `200cm_seed1/` are completed 332-observation optimizer repeats. Their historical names are retained for provenance; the probabilistic candidate simulator does not sample from that seed argument, so they are not independent simulator-seed realizations.
- `2000cm/` contains the frozen target, successful nominal evaluation, completed canonical optimizer run, logs, summaries, and plots.
- `1000cm/repeat_1/` and `repeat_2/` hold optimizer-seed repeats of the canonical 999.926642 cm objective in `.local/six_d/current/`.
- `2000cm/repeat_1/` and `repeat_2/` use the single frozen target stored at the 2000 cm level.
- `2000cm_seed0/` and `2000cm_seed1/` retain compact status provenance from blocked attempts.
- `shared/` holds the frozen 200 cm target and representative validation evidence.
- `comparison/` contains cross-run summaries.

`run_optimizer_repeat.sbatch` is the parameterized production launcher. Set
`BAYESIAN_DATA_SIZE` (`1000cm` or `2000cm`), `BAYESIAN_REPEAT`, and
`BAYESIAN_OPTIMIZER_SEED`; the launcher validates the corresponding target hash
before creating history. The fixed schedule is 72 + 100 + 100 + 60 observations.
The 2000 cm branch activates the validated signal and probabilistic blocking
controls. Histories and targets must never be combined across experiment directories.

`regenerate_plots.py` reads frozen histories only and regenerates convergence,
parallel-coordinate, pairwise conditional-GP, cross-validation, and poster figures.

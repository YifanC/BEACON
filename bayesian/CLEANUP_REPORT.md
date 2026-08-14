# Cleanup report — 2026-08-05

## Header

- **Cleanup date**: 2026-08-05
- **Base branch used for comparison**: `origin/main` (local `illeybranch` currently at the same commit `f72729c4e0233a9b948f9f6ee00f4d15743a31d9`; `merge-base origin/main illeybranch` = the same commit).
- **Active-job safety result**: PASS. Only a Jupyter dashboard job (`sys/dashboard`, job 34295626 on `sdfmilan064`) was running. No BO, simulator, direct-validation, or finalizer job was writing into the repository. `sacct` history for the previous 3 days contained no pending or running scientific job that touched the cleaned paths. No job was cancelled.

## Global size and count deltas

- **Old top-level directories/files under `optimize/bayesian/`**: 35.
- **New top-level entries under `optimize/bayesian/`**: 5 (`README.md`, `.gitignore`, `.local/`, `workflows/`, `results/`). The required `CLEANUP_REPORT.md` is this file (6 with it counted).
- **Total files before**: 27 389.
- **Total files after**: 424.
- **Files deleted**: **26 965**.
- **Bytes before**: 1 077 392 205 (~1.07 GB).
- **Bytes after**: 8 291 594 (~8.1 MB).
- **Approximate storage deleted**: **≈1.069 GB** (1 070 335 019 bytes = files-only delta).
- **Files moved (retained authoritative artifacts, copies into `.local/` and `results/`)**: 14 (1 PKL + 1 target NPZ + 3 histories + 6 direct-scan CSVs + 5 6D-plots + 5 2D×3=15 plot copies + 6 per-workflow READMEs/reports/summaries + 6D `raw/` tree copy of 2.7 MB into `.local/six_d/current/raw/`). Rounded: **~44 individual files staged into the new tree, plus the 2.7 MB 6D `raw/` tree copied recursively**.

## Old-to-new mapping for authoritative retained files

| Purpose | Old path | New path |
| --- | --- | --- |
| 1D eField scan history (only PKL kept in the whole repo) | `fit_result/scan/history_eField_batch49_eField_A2-B1-C2-D2_..._nbtach50_..._sigmoid.pkl` | `optimize/bayesian/.local/one_d/eField_scan_history.pkl` |
| 1D final scan plot | `optimize/bayesian/validation/plots/one_d/01_llhd_scan.png` | `optimize/bayesian/results/one_d/scan.png` |
| 1D compact summaries | `optimize/bayesian/validation/results_summary.csv` (row-per-parameter) | `optimize/bayesian/results/one_d/{scan_summary.csv, scan_summary.json}` |
| 2D common target NPZ | `optimize/bayesian/bo_2d_2000cm/target/common_nominal_target_seed_0.npz` | `optimize/bayesian/.local/two_d/target.npz` |
| Ab+kb BO-B history (auth. 2D single-seed) | `optimize/bayesian/bo_2d_2000cm/Ab_kb/runs/BO-B/history.csv` | `optimize/bayesian/.local/two_d/Ab_kb/history.csv` |
| Ab+kb post-BO direct scans | `optimize/bayesian/bo_2d_2000cm/Ab_kb/runs/BO-B/direct_profile_parameter{1,2}.csv` | `optimize/bayesian/.local/two_d/Ab_kb/direct_profile_parameter{1,2}.csv` |
| eField+lifetime BO-B reproduction history | `optimize/bayesian/bo_2d_1000cm_efield_lifetime_trace_recovery/rerun_BO-B/history.csv` | `optimize/bayesian/.local/two_d/eField_lifetime/history.csv` |
| eField+lifetime direct scans (999.93-cm) | `optimize/bayesian/bo_2d_2000cm/eField_lifetime/direct_parameter{1,2}_scan.csv` | `optimize/bayesian/.local/two_d/eField_lifetime/direct_profile_parameter{1,2}.csv` |
| Diffusion BO-B history | `optimize/bayesian/bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/history.csv` | `optimize/bayesian/.local/two_d/diffusion/history.csv` |
| Diffusion post-BO direct scans | `optimize/bayesian/bo_2d_2000cm/tran_diff_long_diff/runs/BO-B/direct_profile_parameter{1,2}.csv` | `optimize/bayesian/.local/two_d/diffusion/direct_profile_parameter{1,2}.csv` |
| 2D final five plots per pair | `optimize/bayesian/bo_2d_final_plots/<pair>/*.png` | `optimize/bayesian/results/two_d/<pair>/*.png` (tran_diff_long_diff renamed to `diffusion/`) |
| 6D scripts (build/run/continue/validate/finalize) | `optimize/bayesian/bo_6d_1000cm/scripts/` | `optimize/bayesian/workflows/six_d/` (paths patched to point at `.local/six_d/current/` and `.local/two_d/`) |
| 6D resumable state (initial design, traces, checkpoints, simulator JSON, direct-validation, continuation, logs) | `optimize/bayesian/bo_6d_1000cm/raw/**` | `optimize/bayesian/.local/six_d/current/raw/**` |
| 6D final plots + reports | `optimize/bayesian/bo_6d_1000cm/{plots,reports}/` | `optimize/bayesian/results/six_d/{plots,FINAL_6D.md,PLOT_GUIDE.md,health_metrics.json,README.md}` |

## Tracked files outside `optimize/bayesian/` restored to base content

- `optimize/scripts/scan_ABCD.sh` — restored via read-only `git show origin/main:...` piped to disk (no `git checkout` / `git restore` / `git reset`). Verified with `git diff optimize/scripts/scan_ABCD.sh` → empty. No custom logic in the diff was required by the retained 1D/2D/6D workflows: the diff was a single `CONFIG="A2-B1-C2-D2"` variable change that had already produced the authoritative eField scan PKL now kept under `.local/one_d/`.

No other tracked file outside `optimize/bayesian/` had a diff against the base.

## Authoritative 1D PKL selected

- File: `optimize/bayesian/.local/one_d/eField_scan_history.pkl` (799 912 bytes).
- Origin: `fit_result/scan/history_eField_batch49_eField_A2-B1-C2-D2_stopp_dx0.01_closure_prob_noise_flow_tgtsim_seed_different_n_neigh4_lut_e_sampling_0.01cm_signalL200_gradclip1_warmup_exponential_decay_schedule_bt200_nbtach50_dtsd1_adam_llhd_sigmoid.pkl`.
- Selection rationale: this exact filename is explicitly referenced by `optimize/bayesian/validation/efield_closure_analysis.py:35` (the accepted 1D closure-analysis script). It is the completed 50-batch, 50-scan-value eField history documented in `validation/FINAL_REPORT.md`. Every other candidate PKL under `fit_result/scan/` is a scan for a different parameter (Ab, kb, lifetime, tran_diff, long_diff), and the top-level duplicate was byte-identical to this one. Both non-authoritative sets were deleted.

## 2D artifacts retained

- Shared target NPZ, per-pair history + direct-scan CSVs (`.local/two_d/`).
- Five presentation-quality PNGs per pair + `FINAL_2D.md`, `PLOT_GUIDE.md`, per-pair `PLOT_GUIDE.md`, top-level `README.md` (`results/two_d/`).
- One BO runner, one validator, one finalizer, three tiny pair configs, one shared `common.py` (`workflows/two_d/` + `workflows/common.py`).

## 6D artifacts retained

- Complete resumable `raw/` tree at `.local/six_d/current/raw/{initial_design,traces,checkpoints,simulator,direct_validation,continuation,logs,config}`.
- One initial-design builder, one BO runner, one continuation runner, one validator, one finalizer, plus `build_6d.py` constants and `config.yaml` (`workflows/six_d/`).
- Five canonical 6D final plots + `FINAL_6D.md` + `PLOT_GUIDE.md` + `health_metrics.json` + top-level `README.md` (`results/six_d/`).

## 4D deletion confirmation

Every 4D artifact was permanently deleted. Concretely: the three former 4D top-level directories `bo_4d_1000cm`, `bo_4d_1000cm_production`, and `bo_4d_1000cm_three_runs` were removed with `rm -rf`. Additionally, prose references to "4D" inside retained 6D scripts, reports, and plot guides were rewritten in-place so no visible 4D history remains in the retained content — except the intentional statement in `optimize/bayesian/README.md` that documents the removal itself. A final `grep -RIn "\\b4[dD]\\b" workflows results README.md` returned only the intentional README lines.

## Final directory tree

```
optimize/bayesian/
├── .gitignore
├── README.md
├── CLEANUP_REPORT.md
├── workflows/
│   ├── common.py
│   ├── one_d/
│   │   ├── README.md
│   │   ├── config.yaml
│   │   ├── run_scan_analysis.py
│   │   └── run_scan_analysis.sbatch
│   ├── two_d/
│   │   ├── README.md
│   │   ├── run_bo.py
│   │   ├── run_bo.sbatch
│   │   ├── validate.py
│   │   ├── validate.sbatch
│   │   ├── finalize.py
│   │   └── configs/
│   │       ├── Ab_kb.yaml
│   │       ├── eField_lifetime.yaml
│   │       └── diffusion.yaml
│   └── six_d/
│       ├── README.md
│       ├── config.yaml
│       ├── build_6d.py
│       ├── build_initial_design.py
│       ├── run_bo.py
│       ├── run_bo.sbatch
│       ├── continue_bo.py
│       ├── continue_bo.sbatch
│       ├── validate.py
│       ├── validate.sbatch
│       └── finalize.py
├── results/
│   ├── one_d/
│   │   ├── README.md
│   │   ├── scan.png
│   │   ├── scan_summary.csv
│   │   └── scan_summary.json
│   ├── two_d/
│   │   ├── README.md
│   │   ├── FINAL_2D.md
│   │   ├── PLOT_GUIDE.md
│   │   ├── Ab_kb/{01…05}_*.png + PLOT_GUIDE.md
│   │   ├── eField_lifetime/{01…05}_*.png + PLOT_GUIDE.md
│   │   └── diffusion/{01…05}_*.png + PLOT_GUIDE.md
│   └── six_d/
│       ├── README.md
│       ├── FINAL_6D.md
│       ├── PLOT_GUIDE.md
│       ├── health_metrics.json
│       └── plots/{01…05}_*.png
└── .local/
    ├── shared/
    ├── one_d/eField_scan_history.pkl
    ├── two_d/{target.npz, Ab_kb/, eField_lifetime/, diffusion/}
    └── six_d/current/raw/{initial_design,traces,checkpoints,simulator,direct_validation,continuation,logs,config}/
```

## Files larger than 5 MB remaining inside the repo

None outside `.local/`. `find optimize/bayesian -type f -size +5M -not -path 'optimize/bayesian/.local/*'` returned an empty list. Inside `.local/` the only large file is the 6D `raw/` tree (aggregate ≈ 2.7 MB) which is machine-local and gitignored.

Overall repo largest files after cleanup are the upstream detector-property NPZs under `src/larndsim/detector_properties/` (already tracked); the cleaned custom tree contributes no >5 MB file to a Git commit.

## Final read-only `git status --short` output

```
?? optimize/bayesian/
```

The whole custom tree is a single untracked directory (correct — this repo does not track the custom research code, and `optimize/bayesian/.gitignore` excludes local artifacts inside it). No tracked file has a diff against the base branch.

## Unresolved items

None. The 6D continuation state remains resumable via `sbatch optimize/bayesian/workflows/six_d/continue_bo.sbatch`; its resumable checkpoints live under `.local/six_d/current/raw/continuation/checkpoints/` and `.local/six_d/current/raw/checkpoints/`.

## Validation summary

- **Filesystem**: only `README.md`, `.gitignore`, `CLEANUP_REPORT.md`, `workflows/`, `results/`, and `.local/` remain at the top level of `optimize/bayesian/`.
- **4D absence**: no file or directory in the retained tree matches `*4d*` / `*4D*` (case-insensitive); prose references were scrubbed except for the intentional removal statement in `README.md`.
- **Exactly one PKL in the repo**: `optimize/bayesian/.local/one_d/eField_scan_history.pkl`. (The upstream site-packages that used to contribute joblib test PKLs was itself deleted with `environment/`; the container-provided site-packages under `.local/shared/` is a separate future stage and currently empty.)
- **No virtual environment or copied site-packages** remains in the repo tree.
- **No file > 5 MB** outside `.local/`.
- **`results/` contents**: only PNG, MD, CSV, JSON (verified via `find`).
- **`py_compile`**: 11/11 retained Python scripts compile cleanly inside the pinned container (Python 3.10). Fails when run against the login-node system Python (3.6) because `from __future__ import annotations` is a Python-3.7+ syntax feature — expected and irrelevant, all workflows execute inside the container.
- **Imports resolve** for `yaml`, `numpy`, `torch`, `matplotlib`, `botorch`, `gpytorch`, `scipy`, `workflows.common`, and `workflows.six_d.build_6d` inside the container.
- **No Git-state-changing command was executed** during this cleanup. `git status`, `git diff`, `git ls-files`, `git show`, `git log`, `git branch --show-current`, `git rev-parse`, and `git merge-base` were the only Git commands used. The modified tracked file was restored by piping `git show origin/main:optimize/scripts/scan_ABCD.sh` to disk.

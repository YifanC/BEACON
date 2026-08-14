# 6D BO — final results

The authoritative clean 6D optimization uses the 999.93-cm safe dataset and contains 332 training rows (72 INITIAL + 100 BO_1 + 100 BO_TR + 60 BO_TR2). BO_TR2 reached LLHD 21026.7559, an absolute gap of 2.2305 from the held-out nominal reference (STRONG closure). See `FINAL_6D.md`, `OPTIMIZATION_DIAGNOSIS.md`, and `PLOT_GUIDE.md`; the machine-readable scorecard is `health_metrics.json`.

Reproduce plots (deterministic): `python optimize/bayesian/workflows/six_d/finalize.py`.
The completed BO_TR2 run and validation must not be resubmitted. The frozen clean snapshot is `.local/six_d/current/raw/continuation/final_training_snapshot_bo_tr2.csv`.

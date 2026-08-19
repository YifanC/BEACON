# Physical-length optimizer study

The completed six-dimensional studies are organized uniformly by data size and repeat:

```text
batch_study/
├── 200cm/{repeat_0,repeat_1,comparison}
├── 1000cm/{repeat_0,repeat_1,repeat_2,comparison}
├── 2000cm/{repeat_0,repeat_1,repeat_2,comparison,provenance}
├── comparison/
└── shared/
```

Each `repeat_N/run.json` is the analysis-facing record for the run and points to its authoritative frozen history and target. The 1000 cm repeat 0 record intentionally points to `.local/six_d/current/`, which remains authoritative for the primary six-dimensional workflow; its scientific state is not duplicated here.

The 200 cm repeats are repeated executions using the same optimizer seed. The 1000 cm and 2000 cm repeats use distinct optimizer seeds. Every completed run has the same six canonical plots under `plots/`, numbered `01` through `06`.

Within-scale comparison plots live in `<data-size>/comparison/`. Cross-scale plots and the authoritative numerical summary live in `comparison/`; the summary is `comparison/optimizer_repeats.csv`. The immutable 200 cm target is shared in `shared/`, the 1000 cm target remains at `.local/two_d/target.npz`, and the 2000 cm target is `2000cm/target_2000cm.npz`.

`regenerate_plots.py` reconstructs all per-run plots from the frozen histories only. `compare_batch_study.py` regenerates the within-scale and cross-scale comparisons.

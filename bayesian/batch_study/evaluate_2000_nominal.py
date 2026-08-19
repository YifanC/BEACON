"""Run exactly one nominal 2000 cm objective evaluation."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import time

BAYESIAN = Path("/sdf/home/i/iatif/REAL_BEACON/bayesian")
UPSTREAM = Path("/sdf/home/i/iatif/larnd-sim-jax")
sys.path[:0] = [
    str(BAYESIAN / "workflows"),
    str(BAYESIAN / "workflows/six_d"),
    str(UPSTREAM),
    str(UPSTREAM / "src"),
]
os.environ["BAYESIAN_PHYSICAL_LENGTH_CM"] = "2000"
os.environ["BAYESIAN_N_EVENTS"] = "-1"
os.environ["BAYESIAN_MAX_NBATCH"] = "1"
os.environ["BAYESIAN_SIMULATOR_SEED"] = "0"
os.environ["BAYESIAN_DATA_SEED"] = "0"
os.environ["BAYESIAN_TARGET_NPZ"] = str(
    BAYESIAN / "batch_study/2000cm/target_2000cm.npz")

from build_6d import NOM, eval_point, make_objective

started = time.perf_counter()
obj, target = make_objective()
value = eval_point(obj, target, NOM)
elapsed = time.perf_counter() - started
print(json.dumps({
    "coordinate": {name: float(value) for name, value in zip(
        ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff"), NOM)},
    "native_LLHD": float(value),
    "finite": bool(math.isfinite(value)),
    "elapsed_seconds": elapsed,
    "selected_length_cm": float(obj.dataset.tot_data_length),
    "n_events": len(obj.dataset.batch_event_global_ids[0]),
    "n_batches": len(obj.dataset),
}, indent=2))

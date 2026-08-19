"""Validate pixel-blocked probabilistic simulation against the accepted path."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np

BAYESIAN = Path("/sdf/home/i/iatif/REAL_BEACON/bayesian")
UPSTREAM = Path("/sdf/home/i/iatif/larnd-sim-jax")
sys.path[:0] = [
    str(BAYESIAN / "workflows"),
    str(BAYESIAN / "workflows/six_d"),
    str(UPSTREAM),
    str(UPSTREAM / "src"),
]
os.environ.update({
    "BAYESIAN_PHYSICAL_LENGTH_CM": "200",
    "BAYESIAN_N_EVENTS": "-1",
    "BAYESIAN_MAX_NBATCH": "1",
    "BAYESIAN_SIMULATOR_SEED": "0",
    "BAYESIAN_DATA_SEED": "0",
    "BAYESIAN_TARGET_NPZ": str(
        BAYESIAN / "batch_study/shared/target_200cm.npz"),
})

from build_6d import NOM, NAMES, make_objective

obj, target = make_objective()
obj._prepare_batches()
params = obj.base_params.replace(**dict(zip(NAMES, map(float, NOM))))
batch = obj._batches[0]


def run(signal_safe, signal_block, probabilistic_block):
    os.environ["LARNDSIM_MEMORY_EFFICIENT_SIGNALS"] = str(int(signal_safe))
    os.environ["LARNDSIM_SIGNAL_CONTRIB_BLOCK_SIZE"] = str(signal_block)
    os.environ["LARNDSIM_PROBABILISTIC_BLOCK_SIZE"] = str(probabilistic_block)
    prediction = obj._sim_strategy.predict(
        params, batch["tracks"], batch["fields"], rngkey=1)
    loss = obj._loss_strategy.compute(params, prediction, target[0])[0]
    arrays = {
        key: np.asarray(prediction[key])
        for key in (
            "adcs_distrib", "hit_prob", "pixel_x", "pixel_y",
            "pixel_plane", "event", "unique_pixels")
    }
    return float(np.asarray(loss)), arrays


reference_llhd, reference = run(True, 500000, 0)
results = {}
for block_size in (512, 128):
    llhd, candidate = run(True, 500000, block_size)
    comparison = {}
    for key, ref in reference.items():
        other = candidate[key]
        item = {
            "reference_shape": list(ref.shape),
            "candidate_shape": list(other.shape),
            "exact": bool(np.array_equal(ref, other)),
        }
        if np.issubdtype(ref.dtype, np.number):
            delta = np.abs(ref.astype(np.float64) - other.astype(np.float64))
            item["max_abs_difference"] = float(np.nanmax(delta))
            item["n_different"] = int(np.count_nonzero(ref != other))
        comparison[key] = item
    results[str(block_size)] = {
        "native_LLHD": llhd,
        "absolute_difference": abs(llhd - reference_llhd),
        "relative_difference": abs(llhd - reference_llhd) / abs(reference_llhd),
        "comparison": comparison,
    }

print(json.dumps({
    "selected_length_cm": float(obj.dataset.tot_data_length),
    "n_events": len(obj.dataset.batch_event_global_ids[0]),
    "track_shape": list(batch["tracks"].shape),
    "reference_LLHD": reference_llhd,
    "blocked": results,
}, indent=2))

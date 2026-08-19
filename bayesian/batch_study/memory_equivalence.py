"""One-batch equivalence check for the memory-efficient signal reduction."""
from __future__ import annotations

import hashlib
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

os.environ["BAYESIAN_PHYSICAL_LENGTH_CM"] = "200"
os.environ["BAYESIAN_N_EVENTS"] = "-1"
os.environ["BAYESIAN_MAX_NBATCH"] = "1"
os.environ["BAYESIAN_SIMULATOR_SEED"] = "0"
os.environ["BAYESIAN_DATA_SEED"] = "0"
os.environ["BAYESIAN_TARGET_NPZ"] = str(BAYESIAN / "batch_study/shared/target_200cm.npz")

from build_6d import NOM, eval_point, make_objective


def digest(batches):
    h = hashlib.sha256()
    shapes = {}
    for ibatch, batch in enumerate(batches):
        for key in sorted(batch):
            arr = np.asarray(batch[key])
            shapes[f"{ibatch}:{key}"] = list(arr.shape)
            h.update(key.encode())
            h.update(arr.dtype.str.encode())
            h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())
            h.update(arr.tobytes())
    return h.hexdigest(), shapes


obj, retained_target = make_objective()
obj._prepare_batches()
tracks = np.asarray(obj._batches[0]["tracks"])
event_ids = [int(x) for x in obj.dataset.batch_event_global_ids[0]]

def run_mode(memory_efficient, block_size):
    os.environ["LARNDSIM_MEMORY_EFFICIENT_SIGNALS"] = str(int(memory_efficient))
    os.environ["LARNDSIM_SIGNAL_CONTRIB_BLOCK_SIZE"] = str(block_size)
    return obj.generate_nominal_target(), eval_point(obj, retained_target, NOM)


fresh_reference_target, reference_llhd = run_mode(False, 0)
memory_safe_target, memory_safe_llhd = run_mode(True, 0)
blocked_500k_target, blocked_500k_llhd = run_mode(True, 500000)
blocked_100k_target, blocked_100k_llhd = run_mode(True, 100000)

retained_hash, retained_shapes = digest(retained_target)
fresh_reference_hash, fresh_reference_shapes = digest(fresh_reference_target)
safe_hash, safe_shapes = digest(memory_safe_target)
per_key_comparison = {}
for key in sorted(fresh_reference_target[0]):
    ref = np.asarray(fresh_reference_target[0][key])
    safe = np.asarray(memory_safe_target[0][key])
    item = {"exact": bool(np.array_equal(ref, safe))}
    if np.issubdtype(ref.dtype, np.number):
        item["max_abs_difference"] = float(np.max(np.abs(ref.astype(float) - safe.astype(float))))
        item["n_different"] = int(np.count_nonzero(ref != safe))
    per_key_comparison[key] = item
blocked_comparisons = {}
for label, batches in (
        ("500000", blocked_500k_target), ("100000", blocked_100k_target)):
    comparison = {}
    for key in sorted(fresh_reference_target[0]):
        ref = np.asarray(fresh_reference_target[0][key])
        candidate = np.asarray(batches[0][key])
        item = {"exact": bool(np.array_equal(ref, candidate))}
        if np.issubdtype(ref.dtype, np.number):
            item["max_abs_difference"] = float(np.max(
                np.abs(ref.astype(float) - candidate.astype(float))))
            item["n_different"] = int(np.count_nonzero(ref != candidate))
        comparison[key] = item
    blocked_comparisons[label] = comparison
abs_diff = abs(memory_safe_llhd - reference_llhd)
payload = {
    "selected_length_cm": float(obj.dataset.tot_data_length),
    "selected_event_ids": event_ids,
    "n_events": len(event_ids),
    "track_shape": list(tracks.shape),
    "track_dtype": str(tracks.dtype),
    "retained_target_hits": int(sum(len(x["adcs"]) for x in retained_target)),
    "memory_safe_target_hits": int(sum(len(x["adcs"]) for x in memory_safe_target)),
    "retained_target_shapes": retained_shapes,
    "memory_safe_target_shapes": safe_shapes,
    "retained_target_sha256": retained_hash,
    "fresh_reference_target_sha256": fresh_reference_hash,
    "memory_safe_target_sha256": safe_hash,
    "retained_vs_memory_safe_exact": retained_hash == safe_hash,
    "fresh_reference_vs_memory_safe_exact": fresh_reference_hash == safe_hash,
    "fresh_reference_target_shapes": fresh_reference_shapes,
    "per_key_fresh_comparison": per_key_comparison,
    "reference_llhd": reference_llhd,
    "memory_safe_llhd": memory_safe_llhd,
    "blocked_llhd": {
        "500000": blocked_500k_llhd,
        "100000": blocked_100k_llhd,
    },
    "blocked_target_comparisons": blocked_comparisons,
    "absolute_difference": abs_diff,
    "relative_difference": abs_diff / abs(reference_llhd),
}
print(json.dumps(payload, indent=2))

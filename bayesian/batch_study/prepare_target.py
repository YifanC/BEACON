"""Generate a fixed nominal target with the retained simulator semantics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


BAYESIAN = Path(__file__).resolve().parents[1]
UPSTREAM = Path("/sdf/home/i/iatif/larnd-sim-jax")
sys.path.insert(0, str(BAYESIAN / "workflows"))
from objective import LLHDObjective, LossSettings, SimSettings


INPUT = ("/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/"
         "production_884072/job_23771825_0000/"
         "output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_"
         "costheta0.966_range_0.05cm.h5")
NAMES = ("Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff")
KEYS = ("adcs", "pixel_x", "pixel_y", "pixel_z", "ticks",
        "hit_prob", "event", "pixel_id")


def make_objective(length_cm: float) -> LLHDObjective:
    sim = SimSettings(
        input_file=INPUT,
        electron_sampling_resolution=0.1,
        number_pix_neighbors=4,
        signal_length=400,
        noise=True,
        diffusion_in_current_sim=True,
        mc_diff=False,
        use_dedx_density=False,
        dedx_density_mode="histogram",
        max_batch_len=length_cm,
        max_nbatch=1,
        n_events=-1,
        seed=0,
        data_seed=0,
        sim_seed_strategy="same",
    )
    return LLHDObjective(
        sim=sim,
        loss=LossSettings(sigma_charge=500.0, eps=1e-10),
        larndsim_repo=str(UPSTREAM),
        tunable_params=NAMES,
    )


def save_target(path: Path, batches: list[dict]) -> None:
    starts = [0]
    for batch in batches:
        starts.append(starts[-1] + len(batch["adcs"]))
    arrays = {key: np.concatenate([batch[key] for batch in batches]) for key in KEYS}
    arrays["batch_starts"] = np.asarray(starts, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length-cm", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to(BAYESIAN.resolve())
    obj = make_objective(args.length_cm)
    if len(obj.dataset) != 1:
        raise RuntimeError(f"expected one batch, got {len(obj.dataset)}")
    batches = obj.generate_nominal_target()
    save_target(output, batches)
    print(json.dumps({
        "requested_length_cm": args.length_cm,
        "actual_length_cm": float(obj.dataset.tot_data_length),
        "n_batches": len(obj.dataset),
        "n_events": int(len(obj.dataset.batch_event_global_ids[0])),
        "target_hits": int(sum(len(batch["adcs"]) for batch in batches)),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()

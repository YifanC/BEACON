"""Authoritative LLHD candidate objective for the workflows/ tree.

This module wraps the upstream simulator and LLHD implementation in a
deterministic candidate-evaluation object shared by the retained workflows.

Exposes:

    SimSettings           — dataclass describing simulator / dataset config
    LossSettings          — dataclass with sigma_charge + eps
    LLHDObjective         — candidate-evaluation object with .evaluate(dict, target)
    load_target(npz)      — load a saved target NPZ into the list-of-batch-dict
                            shape expected by `ProbabilisticLossStrategy.compute`

Design constraints (enforced here):

    * `sim_seed_strategy='same'` — same as the completed 2D/6D pipelines.
      Under LUT probabilistic simulation the candidate simulator is
      deterministic in `(params, tracks, response)` — two calls at the same
      6D coordinate return byte-identical LLHDs.
    * Parameter order is fixed by `tunable_params`. All numerical arguments to
      `evaluate` are read from a dict keyed by parameter name.
    * The target NPZ is loaded read-only and is never regenerated here.
    * There is no fallback objective. If the upstream simulator import fails,
      construction fails loudly.

The retained upstream modules used here (all in the repository):

    src/larndsim/consts_jax.py:
        build_params_class, load_detector_properties, load_lut
    optimize/strategies.py:
        LUTProbabilisticSimulation, ProbabilisticLossStrategy
    optimize/dataio.py:
        TracksDataset
    src/larndsim/sim_jax.py, src/larndsim/losses_jax.py:
        used transitively via the strategies above
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


REPO = Path("/sdf/home/i/iatif/larnd-sim-jax").resolve()


def _ensure_larndsim_importable(repo_path: str = str(REPO)) -> None:
    """Prepend the retained upstream source paths so `larndsim` and `optimize` import."""
    for p in (repo_path, os.path.join(repo_path, "src")):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# ---------- config dataclasses ----------

@dataclass
class SimSettings:
    """Simulator and dataset configuration for an objective."""
    input_file: str
    detector_props: str = "src/larndsim/detector_properties/module0.yaml"
    pixel_layouts: str = "src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml"
    lut_file: str = "src/larndsim/detector_properties/response_44_v2a_full_tick.npz"
    mode: str = "lut"
    electron_sampling_resolution: float = 0.1
    number_pix_neighbors: int = 4
    signal_length: int = 400
    noise: bool = True
    diffusion_in_current_sim: bool = True
    mc_diff: bool = False
    use_dedx_density: bool = False
    dedx_density_mode: str = "histogram"
    max_batch_len: float = 1000.0
    max_nbatch: Optional[int] = None
    n_events: int = 176
    seed: int = 0
    data_seed: Optional[int] = None
    sim_seed_strategy: str = "same"


@dataclass
class LossSettings:
    """Wrapper around `ProbabilisticLossStrategy` init kwargs."""
    sigma_charge: float = 500.0
    eps: float = 1.0e-10


_TARGET_KEYS = ("adcs", "pixel_x", "pixel_y", "pixel_z", "ticks",
                "hit_prob", "event", "pixel_id")


def _batch_rngkey(strategy: str, ibatch: int, seed: int) -> int:
    """Return the established per-batch key for the selected policy."""
    if strategy == "same":
        return int(seed + ibatch + 1)
    if strategy == "different":
        return int(-ibatch - 1)
    if strategy == "constant":
        return int(seed)
    raise ValueError(f"Unsupported sim_seed_strategy={strategy!r}. "
                      "Use 'same' (default), 'different', or 'constant'.")


def load_target(npz_path: str) -> List[Dict[str, Any]]:
    """Load a saved target NPZ and return it as a list of per-batch dicts.

    The stored NPZ has one flat entry per key across all target hits plus a
    `batch_starts` index array; we split those flat arrays into per-batch
    dicts keyed by the exact names the loss strategy consumes.

    Fails loudly if the file is missing or any required key is absent.
    """
    import numpy as _np
    p = Path(npz_path)
    if not p.exists():
        raise FileNotFoundError(f"target NPZ missing: {p}")
    with p.open("rb") as f:
        z = _np.load(f, allow_pickle=True)
        missing = [k for k in _TARGET_KEYS if k not in z.files]
        if missing:
            raise ValueError(f"target NPZ {p} missing keys: {missing}")
        if "batch_starts" not in z.files:
            raise ValueError(f"target NPZ {p} missing 'batch_starts' partition")
        arrs = {k: _np.asarray(z[k]) for k in _TARGET_KEYS}
        starts = _np.asarray(z["batch_starts"]).astype(int).tolist()
    if starts[0] != 0 or starts[-1] != int(arrs["adcs"].shape[0]):
        raise ValueError(f"target NPZ {p} batch_starts inconsistent with adcs length")
    out: List[Dict[str, Any]] = []
    for a, b in zip(starts[:-1], starts[1:]):
        out.append({k: arrs[k][a:b] for k in _TARGET_KEYS})
    return out


class LLHDObjective:
    """Deterministic candidate-evaluation wrapper.

    Constructed with a `SimSettings`, a `LossSettings`, and the ordered list
    of tunable parameters. `evaluate(overrides, target_batches)` runs the
    fixed-seed LUT probabilistic simulator once and returns the summed native
    LLHD across all batches.

    Target generation and GP fitting are deliberately separate concerns.
    """

    def __init__(self,
                 sim: SimSettings,
                 loss: LossSettings,
                 larndsim_repo: str,
                 tunable_params: Sequence[str]):
        _ensure_larndsim_importable(larndsim_repo)
        # Missing upstream dependencies are fatal: there is no substitute objective.
        from larndsim.consts_jax import (
            build_params_class, load_detector_properties, load_lut,
        )
        from optimize.strategies import (
            LUTProbabilisticSimulation, LUTSimulation, ProbabilisticLossStrategy,
        )
        from optimize.dataio import TracksDataset

        self.sim = sim
        self.loss = loss
        self.larndsim_repo = str(Path(larndsim_repo).resolve())
        self.tunable_params: List[str] = [str(n) for n in tunable_params]
        if len(set(self.tunable_params)) != len(self.tunable_params):
            raise ValueError(f"tunable_params contains duplicates: {self.tunable_params}")

        Params = build_params_class(list(self.tunable_params))
        detprop = os.path.join(self.larndsim_repo, sim.detector_props)
        pixelmap = os.path.join(self.larndsim_repo, sim.pixel_layouts)
        ref = load_detector_properties(Params, detprop, pixelmap)
        # These fixed knobs are part of the scientific objective, not runtime decoration.
        ref = ref.replace(
            electron_sampling_resolution=float(sim.electron_sampling_resolution),
            number_pix_neighbors=int(sim.number_pix_neighbors),
            signal_length=int(sim.signal_length),
            diffusion_in_current_sim=bool(sim.diffusion_in_current_sim),
            mc_diff=bool(sim.mc_diff),
        )
        if not sim.noise:
            # Do not add electronic-noise variance during the candidate simulation
            ref = ref.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)
        self.base_params = ref

        # The updated parameter object carries the LUT-derived drift-length field.
        lut_path = os.path.join(self.larndsim_repo, sim.lut_file)
        response_template, ref = load_lut(lut_path, ref)
        self.base_params = ref
        self._response = response_template
        self._sim_strategy = LUTProbabilisticSimulation(response_template)
        self._target_strategy = LUTSimulation(response_template)
        self._loss_strategy = ProbabilisticLossStrategy(
            sigma_charge=float(loss.sigma_charge), eps=float(loss.eps),
        )

        # Candidate and frozen target use the same track source.
        self.dataset = TracksDataset(
            filename=sim.input_file,
            nevents=int(sim.n_events),
            max_nbatch=sim.max_nbatch,
            max_batch_len=float(sim.max_batch_len),
            data_seed=int(sim.seed if sim.data_seed is None else sim.data_seed),
            electron_sampling_resolution=float(sim.electron_sampling_resolution),
            use_dedx_density=bool(sim.use_dedx_density),
            dedx_density_mode=sim.dedx_density_mode,
        )
        # Populated lazily on first evaluate() so batch fields are known
        self._batches: Optional[List[dict]] = None
        self._fields: Optional[Sequence[str]] = None

    def _prepare_batches(self):
        """Realize the simulator batches once and cache them.

        TracksDataset[idx] returns a 2D float ndarray of shape (nrows, nfields);
        the shared field ordering is retrieved from `dataset.get_track_fields()`.
        """
        if self._batches is not None:
            return
        fields = self.dataset.get_track_fields()
        self._fields = fields
        batches = []
        for i in range(len(self.dataset)):
            tracks = self.dataset[i]
            batches.append({"tracks": tracks, "fields": fields})
        self._batches = batches

    def evaluate(self, overrides: Dict[str, float],
                 target_batches: List[Dict[str, Any]]) -> float:
        """Run the fixed-seed LUT probabilistic simulator and return native LLHD.

        `overrides` maps each tunable parameter name to its physical value.
        `target_batches` is the list-of-batch-dicts returned by `load_target`.
        Returns the summed native LLHD across all batches (float, JAX-safe).
        """
        missing = [n for n in self.tunable_params if n not in overrides]
        if missing:
            raise ValueError(f"missing tunable override(s): {missing}")
        extra = [n for n in overrides if n not in self.tunable_params]
        if extra:
            raise ValueError(f"unknown tunable(s) in overrides: {extra}")

        # Build the per-call params by overriding the retained base
        params = self.base_params.replace(
            **{n: float(overrides[n]) for n in self.tunable_params}
        )
        if not self.sim.noise:
            params = params.replace(RESET_NOISE_CHARGE=0,
                                     UNCORRELATED_NOISE_CHARGE=0)

        self._prepare_batches()
        if len(self._batches) != len(target_batches):
            raise ValueError(
                f"batch count mismatch: candidate has {len(self._batches)} batches, "
                f"target has {len(target_batches)}")

        total = 0.0
        for ibatch, (b, tgt) in enumerate(zip(self._batches, target_batches)):
            # The key is part of the interface; LUTProbabilisticSimulation does not sample it.
            rngseed = _batch_rngkey(self.sim.sim_seed_strategy, ibatch, self.sim.seed)
            prediction = self._sim_strategy.predict(
                params, b["tracks"], b["fields"], rngkey=rngseed,
            )
            out = self._loss_strategy.compute(params, prediction, tgt)
            batch_llhd = out[0] if isinstance(out, tuple) else out
            total += float(np.asarray(batch_llhd))
        return float(total)

    def generate_nominal_target(self) -> List[Dict[str, Any]]:
        """Generate fixed nominal stochastic-hit targets for the configured batches."""
        self._prepare_batches()
        targets: List[Dict[str, Any]] = []
        for ibatch, batch in enumerate(self._batches):
            prediction = self._target_strategy.predict(
                self.base_params,
                batch["tracks"],
                batch["fields"],
                rngkey=ibatch + 1,
            )
            targets.append({
                "adcs": np.asarray(prediction["adcs"]),
                "pixel_x": np.asarray(prediction["pixel_x"]),
                "pixel_y": np.asarray(prediction["pixel_y"]),
                "pixel_z": np.asarray(prediction["pixel_z"]),
                "ticks": np.asarray(prediction["ticks"]),
                "hit_prob": np.asarray(prediction["hit_prob"]),
                "event": np.asarray(prediction["event"]),
                "pixel_id": np.asarray(
                    prediction.get("hit_pixels", prediction.get("unique_pixels"))
                ),
            })
        return targets

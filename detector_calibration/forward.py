"""Programmatic forward simulation + MMD loss for detector calibration BO.

Wraps larnd-sim-jax (https://github.com/NuLads/larnd-sim-jax) to evaluate a
single detector-parameter point: build Params, run simulate_wfs +
simulate_stochastic over all batches with --noise and
--diffusion_in_current_sim active, then compute GenericLossStrategy(mse_adc)
against a cached target.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


def _ensure_larndsim_importable(repo_path: str) -> None:
    if not repo_path:
        return
    src = os.path.join(repo_path, "src")
    for p in (repo_path, src):
        if p and os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


def resolved_target_path(cfg: dict) -> str:
    """Return the target file path, optionally auto-suffixed by sim settings.

    When ``cfg.target_file_auto_suffix`` is true, the user-supplied
    ``cfg.target_file`` (e.g. ``.../target_nominal.npz``) is rewritten to embed
    the settings that change what the simulator actually outputs
    (``max_nbatch``, ``max_batch_len``, ``n_events``, ``sim_seed_strategy``,
    ``seed``). That lets generate_target.py and bo_detector.py keep one cached
    target per sim configuration instead of overwriting the file every time
    ``max_nbatch`` changes.

    Example
    -------
        target_file: /sdf/.../target_nominal.npz
        target_file_auto_suffix: true
        sim:
          max_nbatch: 100
          max_batch_len: 200.0
    →   /sdf/.../target_nominal__nbatch100_maxlen200_nev-1_same-0.npz

    The default (auto_suffix omitted or false) preserves the original behavior
    of a single fixed path.
    """
    base = cfg.get("target_file", "")
    if not cfg.get("target_file_auto_suffix", False) or not base:
        return base
    sim = cfg.get("sim", {})
    nb = sim.get("max_nbatch")
    nb_str = "all" if nb is None else str(int(nb))
    parts = [
        f"nbatch{nb_str}",
        f"maxlen{int(float(sim.get('max_batch_len', 50)))}",
        f"nev{sim.get('n_events', -1)}",
        f"{sim.get('sim_seed_strategy', 'constant')}-{sim.get('seed', 0)}",
    ]
    root, ext = os.path.splitext(base)
    return f"{root}__{'_'.join(parts)}{ext}"


@dataclass
class SimConfig:
    input_file: str
    detector_props: str
    pixel_layouts: str
    lut_file: str
    mode: str = "lut"
    electron_sampling_resolution: float = 0.1
    number_pix_neighbors: int = 4
    signal_length: int = 400
    noise: bool = True
    diffusion_in_current_sim: bool = True
    mc_diff: bool = False
    use_dedx_density: bool = False
    dedx_density_mode: str = "histogram"
    max_batch_len: float = 50.0
    max_nbatch: Optional[int] = None
    n_events: int = -1
    seed: int = 0
    sim_seed_strategy: str = "constant"  # same | different | different_epoch | random | constant


@dataclass
class LossConfig:
    sigma: float = 1.0
    lambda_Q: float = 1.0


@dataclass
class ForwardSession:
    """Holds the dataset, response, base Params and loss strategy.

    A single instance is reused across BO iterations; each call to
    :meth:`evaluate` only varies the per-iteration override dict.
    """

    sim_cfg: SimConfig
    loss_cfg: LossConfig
    larndsim_repo: str = ""
    tunable_params: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        _ensure_larndsim_importable(self.larndsim_repo)

        from larndsim.consts_jax import build_params_class, load_detector_properties, load_lut
        from larndsim.sim_jax import simulate_stochastic, simulate_wfs, pad_size
        from optimize.dataio import TracksDataset
        from optimize.strategies import LUTSimulation, GenericLossStrategy
        from larndsim.losses_jax import mse_adc

        self._build_params_class = build_params_class
        self._load_detector_properties = load_detector_properties
        self._load_lut = load_lut
        self._simulate_wfs = simulate_wfs
        self._simulate_stochastic = simulate_stochastic
        self._pad_size = pad_size
        self._TracksDataset = TracksDataset

        # Base Params (nominal) — overrides applied per iteration via .replace().
        Params = build_params_class(list(self.tunable_params))
        ref_params = load_detector_properties(
            Params, self._resolve(self.sim_cfg.detector_props), self._resolve(self.sim_cfg.pixel_layouts)
        )
        if self.sim_cfg.mode == "lut":
            response, ref_params = load_lut(self._resolve(self.sim_cfg.lut_file), ref_params)
        else:
            response = None

        ref_params = ref_params.replace(
            electron_sampling_resolution=self.sim_cfg.electron_sampling_resolution,
            number_pix_neighbors=self.sim_cfg.number_pix_neighbors,
            signal_length=self.sim_cfg.signal_length,
            time_window=self.sim_cfg.signal_length,
            diffusion_in_current_sim=self.sim_cfg.diffusion_in_current_sim,
            mc_diff=self.sim_cfg.mc_diff,
            use_dedx_density=self.sim_cfg.use_dedx_density,
            dedx_density_mode=self.sim_cfg.dedx_density_mode,
        )
        if not self.sim_cfg.noise:
            ref_params = ref_params.replace(RESET_NOISE_CHARGE=0, UNCORRELATED_NOISE_CHARGE=0)

        self.base_params = ref_params
        self.response = response

        self.dataset = TracksDataset(
            filename=self.sim_cfg.input_file,
            nevents=self.sim_cfg.n_events,
            max_nbatch=self.sim_cfg.max_nbatch,
            swap_xz=True,
            random_nevents=False,
            data_seed=self.sim_cfg.seed,
            max_batch_len=self.sim_cfg.max_batch_len,
            print_input=False,
            chopped=False,
            pad=False,
            electron_sampling_resolution=self.sim_cfg.electron_sampling_resolution,
            live_selection=False,
            use_dedx_density=self.sim_cfg.use_dedx_density,
            dedx_density_mode=self.sim_cfg.dedx_density_mode,
        )
        self.fields = self.dataset.get_track_fields()

        self.sim_strategy = LUTSimulation(self.response)
        self.loss_strategy = GenericLossStrategy(
            loss_fn=mse_adc, sigma=self.loss_cfg.sigma, lambda_Q=self.loss_cfg.lambda_Q
        )

    def _resolve(self, p: str) -> str:
        if os.path.isabs(p) or self.larndsim_repo == "":
            return p
        return os.path.join(self.larndsim_repo, p)

    def _iter_batches(self, batches_to_use: Optional[Sequence[int]] = None):
        """Yield (ibatch, device-resident tracks) for the full dataset or a subset.

        When ``batches_to_use`` is provided, only those batch indices are
        iterated (in the given order). Used by :meth:`predict` / :meth:`evaluate`
        to implement batch rotation: random subsets per BO evaluation.
        """
        import jax
        order = range(len(self.dataset)) if batches_to_use is None else list(batches_to_use)
        for ibatch in order:
            batch = self.dataset[ibatch]
            size = self._pad_size(batch.shape[0], "batch_size", 0.5)
            batch = self.dataset.pad_batch(batch, size, ibatch)
            yield int(ibatch), jax.device_put(batch)

    def _batch_rngkey(self, ibatch: int, epoch: int = 0) -> int:
        """Return the JAX rngkey for batch ``ibatch`` under the configured strategy.

        Mirrors the larnd-sim-jax optimizer convention:
          - "same"            : ibatch + 1   (per-batch noise, fixed across BO θ)
          - "different"       : -ibatch - 1  (per-batch noise, negative offset)
          - "different_epoch" : -ibatch - 1 - epoch * 10000  (per-epoch fresh noise)
          - "random"          : np.random.randint  (different every call — breaks GP determinism)
          - "constant"        : sim_cfg.seed  (every batch identical)
        """
        s = self.sim_cfg.sim_seed_strategy
        if s == "same":
            return int(ibatch + 1)
        if s == "different":
            return int(-ibatch - 1)
        if s == "different_epoch":
            return int(-ibatch - 1 - epoch * 10000)
        if s == "random":
            return int(np.random.randint(0, 1_000_000))
        if s == "constant":
            return int(self.sim_cfg.seed)
        raise ValueError(
            f"Unknown sim_seed_strategy={s!r}. "
            "Must be one of: same, different, different_epoch, random, constant."
        )

    def predict(self, params, epoch: int = 0,
                batches_to_use: Optional[Sequence[int]] = None) -> List[dict]:
        """Run the forward sim and return a list of prediction dicts.

        By default simulates every batch in the dataset. If ``batches_to_use``
        is provided, only those batch indices are simulated (in order). The
        simulator's per-batch rngkey still derives deterministically from the
        absolute batch index via :meth:`_batch_rngkey`, so picking the same
        subset at the same ``params`` produces bit-identical output every call.

        The simulator emits ``event`` as a per-batch local index ``0..k-1``
        (see ``optimize/dataio.remap_event_ids_to_local``). We translate it
        back to the truth-file's global ``event_id`` here so every consumer
        — target generation, BO loss, downstream notebooks — speaks the same
        namespace.
        """
        import jax.numpy as jnp
        preds = []
        for ibatch, tracks in self._iter_batches(batches_to_use):
            rngseed = self._batch_rngkey(ibatch, epoch=epoch)
            pred = self.sim_strategy.predict(params, tracks, self.fields, rngkey=rngseed)

            global_ids = jnp.asarray(self.dataset.batch_event_global_ids[ibatch])
            local = pred["event"]
            # Padded rows can carry event=-1; clamp the gather index then mask back.
            n = max(int(global_ids.size), 1)
            safe_idx = jnp.clip(local.astype(jnp.int32), 0, n - 1)
            mapped = jnp.take(global_ids, safe_idx).astype(local.dtype)
            new_event = jnp.where(local < 0, local, mapped)
            pred = {**pred, "event": new_event}

            preds.append(pred)
        return preds

    def evaluate(self, overrides: Dict[str, float], targets: List[dict],
                 batches_to_use: Optional[Sequence[int]] = None) -> float:
        """Apply param overrides, run the sim, return the scalar MMD+charge loss.

        ``targets`` is the per-batch list produced by :meth:`build_target`. The
        loss is computed batch-by-batch (``pred_b`` vs ``target_b``) and summed.
        When ``batches_to_use`` is provided, both the simulator and the target
        list are restricted to those batch indices — supports batch-rotated BO,
        where each evaluation samples a fresh subset of the available batches.
        """
        params = self.base_params.replace(**overrides)
        preds = self.predict(params, batches_to_use=batches_to_use)
        if batches_to_use is not None:
            targets = [targets[int(b)] for b in batches_to_use]
        if len(preds) != len(targets):
            raise ValueError(
                f"Batch-count mismatch: prediction has {len(preds)} batch(es), "
                f"target has {len(targets)}. Regenerate the target with the same "
                "sim settings (max_batch_len, max_nbatch, n_events, ...)."
            )
        total = 0.0
        for pred, tgt in zip(preds, targets):
            loss_val, _aux = self.loss_strategy.compute(params, pred, tgt)
            total += float(loss_val)
        return total

    def build_target(self) -> List[dict]:
        """Run the forward sim at base (nominal) params and return per-batch dicts.

        Each entry in the returned list corresponds to one simulation batch and
        is the (filtered) prediction dict for that batch. Saves nothing — callers
        decide whether to persist via :func:`save_target`.
        """
        preds = self.predict(self.base_params)
        keys = ["adcs", "pixel_x", "pixel_y", "pixel_z", "ticks", "hit_prob", "event"]
        return [{k: pred[k] for k in keys} for pred in preds]

    def target_meta(self) -> Dict[str, np.ndarray]:
        """Detector constants and batch→global mapping needed by downstream tools.

        Persisted alongside the per-batch hits by :func:`save_target` so the
        visualization notebook can:
          - compute Q via the same ``adc2charge`` formula the loss uses
            (see ``larndsim.losses_jax.adc2charge``);
          - draw TPC boundaries from ``params.tpc_borders``;
          - translate the simulator's per-batch local ``event`` indices
            back to the truth file's global event_ids — the simulator
            renumbers events to ``0..k-1`` per batch
            (``optimize/dataio.remap_event_ids_to_local``), which is
            invisible to the per-batch loss but breaks any cross-batch
            grouping (e.g. event-by-event plotting).
        """
        p = self.base_params
        global_ids = [np.asarray(a, dtype=np.int64) for a in self.dataset.batch_event_global_ids]
        flat = np.concatenate(global_ids) if global_ids else np.empty(0, dtype=np.int64)
        starts = np.cumsum([0] + [len(a) for a in global_ids], dtype=np.int64)
        return {
            "tpc_borders": np.asarray(p.tpc_borders),  # (N_tpc, 3, 2): (x,y,z) × (min,max)
            "adc_v_cm": np.float64(p.V_CM),
            "adc_v_ref": np.float64(p.V_REF),
            "adc_v_pedestal": np.float64(p.V_PEDESTAL),
            "adc_gain": np.float64(p.GAIN),
            "adc_counts": np.int64(p.ADC_COUNTS),
            # Per-batch table of global event_ids; batch b's local event index
            # ``e`` maps to ``flat[starts[b] + e]``.
            "batch_event_global_ids_flat": flat,
            "batch_event_global_ids_starts": starts,
        }


_TARGET_KEYS = ("adcs", "pixel_x", "pixel_y", "pixel_z", "ticks", "hit_prob", "event")
_META_KEYS = (
    "tpc_borders",
    "adc_v_cm", "adc_v_ref", "adc_v_pedestal", "adc_gain", "adc_counts",
    "batch_event_global_ids_flat", "batch_event_global_ids_starts",
)


def save_target(targets: List[dict], path: str, meta: Optional[Dict[str, np.ndarray]] = None) -> None:
    """Persist a per-batch target as a single npz with a ``batch_starts`` index.

    Layout: each key holds the concatenation of that field across batches.
    ``batch_starts`` is an int64 array of length ``n_batches + 1``: batch ``b``
    occupies rows ``[batch_starts[b], batch_starts[b+1])``. If ``meta`` is
    provided, its arrays are stored under their own (non-hit) keys; see
    ``_META_KEYS`` for the recognized names.
    """
    if not targets:
        np.savez(path, batch_starts=np.array([0], dtype=np.int64))
        return
    keys = [k for k in _TARGET_KEYS if k in targets[0]]
    sizes = [int(np.asarray(t[keys[0]]).shape[0]) for t in targets]
    batch_starts = np.cumsum([0] + sizes, dtype=np.int64)
    arrays = {k: np.concatenate([np.asarray(t[k]) for t in targets]) for k in keys}
    arrays["batch_starts"] = batch_starts
    if meta:
        arrays.update({k: np.asarray(v) for k, v in meta.items()})
    np.savez(path, **arrays)


def load_target(path: str) -> List[dict]:
    """Load a per-batch target written by :func:`save_target`.

    For backward compatibility, files without ``batch_starts`` are returned as
    a single-batch list (the old flat-blob format). This will fail loudly in
    :meth:`ForwardSession.evaluate` if the simulator now produces multiple
    batches — regenerate the target. Metadata keys (``_META_KEYS``) are skipped.
    """
    z = np.load(path)
    import jax.numpy as jnp
    files = list(z.files)
    hit_keys = [k for k in _TARGET_KEYS if k in files]
    if "batch_starts" not in files:
        return [{k: jnp.asarray(z[k]) for k in hit_keys}]
    starts = z["batch_starts"]
    out: List[dict] = []
    for i in range(len(starts) - 1):
        s, e = int(starts[i]), int(starts[i + 1])
        out.append({k: jnp.asarray(z[k][s:e]) for k in hit_keys})
    return out


def load_target_meta(path: str) -> Dict[str, np.ndarray]:
    """Load only the detector-metadata side of a target file.

    Returns whatever subset of ``_META_KEYS`` was persisted; older targets
    written before this addition return an empty dict.
    """
    z = np.load(path)
    return {k: np.asarray(z[k]) for k in _META_KEYS if k in z.files}

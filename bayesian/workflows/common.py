"""Shared infrastructure for the 2D and 6D workflows.

This module implements only what every retained workflow needs:
  - The pinned Matern-5/2 ARD SingleTaskGP with Standardize(m=1),
    score = -ln(native_LLHD), and `train_Yvar = 1 / native_LLHD^2`.
  - qLogExpectedImprovement acquisition (q=1) with `optimize_acqf`.
  - LLHDObjective factory that reuses the upstream `optimize.bayesian` code.
"""
from __future__ import annotations
from dataclasses import dataclass
import os
from pathlib import Path
import numpy as np
import torch

from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

DTYPE = torch.double

REPO = Path(os.environ.get("LARNDSIM_REPOSITORY", "/sdf/home/i/iatif/larnd-sim-jax")).resolve()
BAY = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime ownership for one optimizer execution."""

    name: str
    run_root: Path
    target: Path
    physical_length_cm: float
    n_events: int
    max_nbatch: int | None
    simulator_seed: int = 0
    data_seed: int = 0
    optimizer_seed: int = 20260812
    initial_evaluations: int = 72
    global_bo_evaluations: int = 100
    trust_region_evaluations: int = 100
    final_refinement_evaluations: int = 60

    @classmethod
    def from_environment(cls, default_root: Path, default_target: Path) -> "ExperimentConfig":
        max_nbatch = os.environ.get("BAYESIAN_MAX_NBATCH", "")
        return cls(
            name=os.environ.get("BAYESIAN_EXPERIMENT", "6D-1000cm"),
            run_root=Path(os.environ.get("BAYESIAN_RUN_ROOT", default_root)).resolve(),
            target=Path(os.environ.get("BAYESIAN_TARGET_NPZ", default_target)).resolve(),
            physical_length_cm=float(os.environ.get("BAYESIAN_PHYSICAL_LENGTH_CM", "1000")),
            n_events=int(os.environ.get("BAYESIAN_N_EVENTS", "176")),
            max_nbatch=int(max_nbatch) if max_nbatch else None,
            simulator_seed=int(os.environ.get("BAYESIAN_SIMULATOR_SEED", "0")),
            data_seed=int(os.environ.get("BAYESIAN_DATA_SEED", "0")),
            optimizer_seed=int(os.environ.get("BAYESIAN_OPTIMIZER_SEED", "20260812")),
        )

    @property
    def total_evaluations(self) -> int:
        return (self.initial_evaluations + self.global_bo_evaluations
                + self.trust_region_evaluations + self.final_refinement_evaluations)


@dataclass(frozen=True)
class ParameterSpace:
    """Names, physical bounds, and held-out nominal closure coordinate."""

    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    nominal: np.ndarray

    def to_unit(self, physical: np.ndarray) -> np.ndarray:
        return (np.asarray(physical, dtype=float) - self.lower) / (self.upper - self.lower)

    def to_physical(self, unit: np.ndarray) -> np.ndarray:
        return self.lower + np.asarray(unit, dtype=float) * (self.upper - self.lower)


def fit_gp(x_unit: np.ndarray, native_llhd: np.ndarray, *, seed: int,
           ard_num_dims: int) -> SingleTaskGP:
    """Fit the validated transformed GP.

    Args:
        x_unit: (N, d) inputs already normalized to [0, 1]^d.
        native_llhd: (N,) simulator native-LLHD values (positive).
        seed: RNG seed for reproducibility.
        ard_num_dims: input dimensionality.

    The model targets `score = -ln(native_LLHD)` and uses
    `train_Yvar = 1 / native_LLHD^2`. Standardize(m=1) is applied to the
    outcome. Matern-5/2 ARD kernel.
    """
    torch.manual_seed(seed)
    x = torch.tensor(x_unit, dtype=DTYPE)
    l = torch.tensor(native_llhd, dtype=DTYPE).view(-1, 1)
    y = -torch.log(l)
    yv = 1.0 / l**2
    cov = ScaleKernel(
        MaternKernel(nu=2.5, ard_num_dims=ard_num_dims,
                     lengthscale_constraint=Interval(1e-3, 20.0)),
        outputscale_constraint=Interval(1e-4, 100.0),
    )
    cov.base_kernel.lengthscale = torch.full((1, ard_num_dims), 0.3, dtype=DTYPE)
    cov.outputscale = torch.tensor(1.0, dtype=DTYPE)
    m = SingleTaskGP(x, y, train_Yvar=yv, covar_module=cov,
                     outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    m.eval()
    return m


def propose_next(model: SingleTaskGP, native_llhd: np.ndarray, bounds: torch.Tensor,
                 *, num_restarts: int = 24, raw_samples: int = 2048):
    """Return (u_next, acq_value) using qLogExpectedImprovement (q=1)."""
    y = -torch.log(torch.tensor(native_llhd, dtype=DTYPE)).view(-1, 1)
    acq = qLogExpectedImprovement(model, best_f=y.max())
    cand, val = optimize_acqf(acq, bounds, q=1,
                              num_restarts=num_restarts,
                              raw_samples=raw_samples)
    return (cand.detach().cpu().numpy()[0],
            float(val.detach().cpu().reshape(-1)[0]))


def make_llhd_objective(n_events: int, tunable_params, target_seed: int = 0,
                        sim_seed_strategy: str = "same"):
    """Build the six-parameter LLHDObjective the same way the validated 2D and
    6D pipelines do. `tunable_params` selects which subset of parameters the
    caller varies during BO; the rest are held at their base nominal values.

    Returns (obj, load_target) so the caller can call `load_target(str(npz))`.
    """
    import sys
    sys.path[:0] = [str(REPO), str(REPO / "src")]
    # Authoritative in-tree objective (workflows/objective.py). No fallback.
    sys.path.insert(0, str(REPO / "optimize/bayesian/workflows"))
    from objective import LLHDObjective, LossSettings, SimSettings, load_target  # type: ignore
    INPUT = ("/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/"
             "production_884072/job_23771825_0000/"
             "output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_"
             "costheta0.966_range_0.05cm.h5")
    sim = SimSettings(
        input_file=INPUT,
        detector_props="src/larndsim/detector_properties/module0.yaml",
        pixel_layouts="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
        lut_file="src/larndsim/detector_properties/response_44_v2a_full_tick.npz",
        mode="lut", electron_sampling_resolution=0.1, number_pix_neighbors=4,
        signal_length=400, noise=True, diffusion_in_current_sim=True,
        mc_diff=False, use_dedx_density=False, dedx_density_mode="histogram",
        max_batch_len=1000.0, max_nbatch=None,
        n_events=n_events, seed=target_seed, sim_seed_strategy=sim_seed_strategy,
    )
    obj = LLHDObjective(sim=sim, loss=LossSettings(sigma_charge=500.0, eps=1e-10),
                        larndsim_repo=str(REPO),
                        tunable_params=tuple(tunable_params))
    return obj, load_target


def unit_of(x_phys, lo, hi):
    return (np.asarray(x_phys, dtype=float) - np.asarray(lo)) / \
           (np.asarray(hi) - np.asarray(lo))


def physical_of(u, lo, hi):
    return np.asarray(lo) + np.asarray(u, dtype=float) * \
           (np.asarray(hi) - np.asarray(lo))

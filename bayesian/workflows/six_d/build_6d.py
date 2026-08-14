"""Shared 6D constants, GP fit, and simulator factory.

Uses the same validated transformed-GP methodology from the 2D repair:
  - X normalised to [0,1]^6
  - SingleTaskGP with Matern-5/2 ARD(6), Standardize(m=1), train_Yvar = 1/L^2
  - score = -ln(native_LLHD)
  - fit_gpytorch_mll
Constants: parameter names, bounds, nominals, physical dataset paths, target NPZ.
"""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood

REPO = Path("/sdf/home/i/iatif/larnd-sim-jax").resolve()
ALLOWED = REPO / "optimize/bayesian"
# .local/six_d/current/ is the single resumable state tree. Continuation runs
# do not create a new top-level directory: they resume from checkpoints inside
# this tree.
ROOT = (ALLOWED / ".local/six_d/current").resolve()
RAW = ROOT / "raw"

NAMES = ["Ab", "kb", "eField", "lifetime", "tran_diff", "long_diff"]
LO = np.array([0.75, 0.03, 0.49, 400.0, 3.0e-6, 1.0e-6])
HI = np.array([0.90, 0.08, 0.51, 6000.0, 15.0e-6, 10.0e-6])
NOM = np.array([0.80, 0.0486, 0.50, 2200.0, 8.8e-6, 4.0e-6])
UNITS = ["", "kV*g/(MeV*cm^3)", "kV/cm", "us", "cm^2/us", "cm^2/us"]
DTYPE = torch.double

TARGET = (ALLOWED / ".local/two_d/target.npz").resolve()
INPUT_HDF5 = ("/sdf/data/neutrino/cyifan/dunend_train_prod/prod_mod0_mpvmpr/"
              "production_884072/job_23771825_0000/"
              "output_23771825_0000-edepsim_lbl_trklen2cm_containment2cm_"
              "costheta0.966_range_0.05cm.h5")


def guard(p) -> Path:
    p = Path(p).resolve()
    p.relative_to(ROOT)
    return p


def unit(x) -> np.ndarray:
    return (np.asarray(x, dtype=float) - LO) / (HI - LO)


def physical(u) -> np.ndarray:
    return LO + np.asarray(u, dtype=float) * (HI - LO)


def read_csv(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def fit_gp(X, L, seed=20260812, nu=2.5):
    """Fit the 6D surrogate on (X in physical units, L in native LLHD).

    `nu` selects the Matérn smoothness parameter: 2.5 for the historical BO_TR
    surrogate (Model A), 1.5 for the BO_TR2 surrogate (Model C, selected by
    the audit shootout).
    """
    torch.manual_seed(seed)
    x = torch.tensor(unit(X), dtype=DTYPE)
    l = torch.tensor(L, dtype=DTYPE).view(-1, 1)
    y = -torch.log(l)
    yv = 1.0 / l ** 2
    cov = ScaleKernel(
        MaternKernel(nu=nu, ard_num_dims=6,
                     lengthscale_constraint=Interval(1e-3, 20.0)),
        outputscale_constraint=Interval(1e-4, 100.0),
    )
    cov.base_kernel.lengthscale = torch.full((1, 6), 0.3, dtype=DTYPE)
    cov.outputscale = torch.tensor(1.0, dtype=DTYPE)
    m = SingleTaskGP(x, y, train_Yvar=yv, covar_module=cov,
                     outcome_transform=Standardize(m=1))
    fit_gpytorch_mll(ExactMarginalLogLikelihood(m.likelihood, m))
    m.eval()
    return m


def make_objective():
    """Return (objective, target) using the validated common 999.93-cm setup with 6 tunables."""
    import sys
    sys.path[:0] = [str(REPO), str(REPO / "src")]
    # Authoritative in-tree objective (workflows/objective.py). No fallback.
    sys.path.insert(0, str(REPO / "optimize/bayesian/workflows"))
    from objective import LLHDObjective, LossSettings, SimSettings, load_target
    sim = SimSettings(
        input_file=INPUT_HDF5,
        detector_props="src/larndsim/detector_properties/module0.yaml",
        pixel_layouts="src/larndsim/pixel_layouts/multi_tile_layout-2.4.16_v4.yaml",
        lut_file="src/larndsim/detector_properties/response_44_v2a_full_tick.npz",
        mode="lut",
        electron_sampling_resolution=0.1,
        number_pix_neighbors=4,
        signal_length=400,
        noise=True,
        diffusion_in_current_sim=True,
        mc_diff=False,
        use_dedx_density=False,
        dedx_density_mode="histogram",
        max_batch_len=1000.0,
        max_nbatch=None,
        n_events=176,
        seed=0,
        sim_seed_strategy="same",
    )
    obj = LLHDObjective(
        sim=sim,
        loss=LossSettings(sigma_charge=500.0, eps=1e-10),
        larndsim_repo=str(REPO),
        tunable_params=tuple(NAMES),
    )
    assert len(obj.dataset) == 1 and abs(obj.dataset.tot_data_length - 999.926641702652) < 1e-3
    tgt = load_target(str(TARGET))
    assert len(tgt) == 1 and sum(int(t["adcs"].size) for t in tgt) == 9368
    return obj, tgt


def eval_point(obj, tgt, x):
    return float(obj.evaluate(dict(zip(NAMES, map(float, x))), tgt))

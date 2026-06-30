"""Bayesian Optimization for kinematic-fit parameters (x0, v0, a) using BoTorch.

Recovers (x0, v0, a) of x(t) = x0 + v0*t + 0.5*a*t^2 from synthetic noisy
observations by maximizing -MSE with a SingleTaskGP + qLogExpectedImprovement.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP
from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood


DTYPE = torch.double
DEVICE = torch.device("cpu")

TRUE_PARAMS = (1.0, 2.0, -9.81)
NOISE_STD = 0.05
N_INIT = 12
N_ITER = 40
Q = 1

BOUNDS = torch.tensor(
    [[-5.0, -5.0, -20.0], [5.0, 5.0, 20.0]], dtype=DTYPE, device=DEVICE
)


def make_data(true_params, t, noise_std):
    x0, v0, a = true_params
    x_clean = x0 + v0 * t + 0.5 * a * t**2
    x_obs = x_clean + noise_std * torch.randn_like(x_clean)
    return x_obs


def model_x(params: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    # params: (q, 3); t: (T,) -> trajectories: (q, T)
    x0 = params[..., 0:1]
    v0 = params[..., 1:2]
    a = params[..., 2:3]
    t_row = t.unsqueeze(0)
    return x0 + v0 * t_row + 0.5 * a * t_row**2


def objective(params: torch.Tensor, t: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
    # Returns -MSE with shape (q, 1): BoTorch expects a trailing outcome dim.
    x_pred = model_x(params, t)
    mse = ((x_pred - x_obs.unsqueeze(0)) ** 2).mean(dim=-1, keepdim=True)
    return -mse


def run_bo(t: torch.Tensor, x_obs: torch.Tensor):
    train_X = draw_sobol_samples(bounds=BOUNDS, n=N_INIT, q=1, seed=0).squeeze(1).to(
        dtype=DTYPE, device=DEVICE
    )
    train_Y = objective(train_X, t, x_obs)

    history = []  # best -MSE so far per iter (post-eval)

    for it in range(N_ITER):
        # Use BoTorch's input/outcome transforms: Normalize handles the
        # disparate per-dim scales (x0,v0 in [-5,5] vs a in [-20,20]) and
        # Standardize handles outcome scaling, so best_f stays in raw units.
        model = SingleTaskGP(
            train_X,
            train_Y,
            input_transform=Normalize(d=BOUNDS.shape[-1], bounds=BOUNDS),
            outcome_transform=Standardize(m=1),
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_mll(mll)

        acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
        candidate, _ = optimize_acqf(
            acq_function=acq,
            bounds=BOUNDS,
            q=Q,
            num_restarts=10,
            raw_samples=256,
        )

        new_y = objective(candidate, t, x_obs)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)

        best_idx = int(torch.argmax(train_Y).item())
        best_neg_mse = float(train_Y[best_idx].item())
        best_mse = -best_neg_mse
        bx0, bv0, ba = [float(v) for v in train_X[best_idx].tolist()]
        history.append(best_mse)
        print(
            f"iter {it+1:3d}/{N_ITER} | best -MSE = {best_neg_mse: .6e} "
            f"(MSE = {best_mse:.6e}) | x0={bx0: .4f} v0={bv0: .4f} a={ba: .4f}"
        )

    return train_X, train_Y, history


def plot_fit(t, x_obs, best_params, save_path):
    t_dense = torch.linspace(
        float(t.min()), float(t.max()), 200, dtype=DTYPE, device=DEVICE
    )
    x_true = TRUE_PARAMS[0] + TRUE_PARAMS[1] * t_dense + 0.5 * TRUE_PARAMS[2] * t_dense**2
    x_fit = model_x(best_params.unsqueeze(0), t_dense).squeeze(0)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(t.cpu().numpy(), x_obs.cpu().numpy(), color="k", s=20, label="observed")
    ax.plot(t_dense.cpu().numpy(), x_true.cpu().numpy(), "g--", label="true")
    ax.plot(t_dense.cpu().numpy(), x_fit.cpu().numpy(), "r-", label="best fit")
    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.legend()
    ax.set_title("Kinematic fit via BoTorch BO")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_convergence(history, save_path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, "b-o", markersize=3)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("best MSE so far")
    ax.set_title("BO convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    t = torch.linspace(0.0, 2.0, 20, dtype=DTYPE, device=DEVICE)
    x_obs = make_data(TRUE_PARAMS, t, NOISE_STD)

    train_X, train_Y, history = run_bo(t, x_obs)

    best_idx = int(torch.argmax(train_Y).item())
    best_params = train_X[best_idx]
    best_mse = float(-train_Y[best_idx].item())
    bx0, bv0, ba = [float(v) for v in best_params.tolist()]
    tx0, tv0, ta = TRUE_PARAMS

    print()
    print("=" * 60)
    print(f"Best params: x0={bx0:.6f} v0={bv0:.6f} a={ba:.6f}")
    print(f"True params: x0={tx0:.6f} v0={tv0:.6f} a={ta:.6f}")
    print(f"Errors:      dx0={bx0-tx0:+.6f} dv0={bv0-tv0:+.6f} da={ba-ta:+.6f}")
    print(f"Best MSE:    {best_mse:.6e} (noise variance ~ {NOISE_STD**2:.6e})")
    print("=" * 60)

    here = os.path.dirname(os.path.abspath(__file__))
    plot_fit(t, x_obs, best_params, os.path.join(here, "bo_kinematics_fit.png"))
    plot_convergence(history, os.path.join(here, "bo_kinematics_convergence.png"))


if __name__ == "__main__":
    main()

"""Bayesian optimization of larnd-sim-jax detector-calibration parameters.

Mirrors the structure of ``examples/bo_kinematics.py``: SingleTaskGP +
qLogExpectedImprovement on a (D=6) box defined by ``optimize/ranges.py``.
The forward model is one full pass of larnd-sim-jax with --noise and
--diffusion_in_current_sim active; the BO objective is
``-GenericLossStrategy(mse_adc).compute(prediction, target)``, where the
target is a frozen sim at nominal params.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from botorch.acquisition.logei import qLogExpectedImprovement, qLogNoisyExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import ChainedInputTransform, Log10, Normalize
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.utils.sampling import draw_sobol_samples
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import minimize

from . import forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DTYPE = torch.double
DEVICE = torch.device("cpu")  # GP/acquisition stays on CPU; only larnd-sim-jax uses GPU.


def _build_param_space(params: List[dict]):
    """Per-parameter physical bounds + log/linear normalization helpers.

    Each YAML param entry may carry ``log: true`` (default ``false``) to switch
    its normalization to log10. The GP's input transform applies log10 then
    Normalize on those axes; Sobol samples uniformly in ``[0, 1]^D`` and is
    unnormalized via ``from_unit``, which yields log-uniform samples in physical
    space on log axes and linear-uniform samples on the rest. The simulator,
    ``optimize_acqf``, and the L-BFGS-B polish all continue to see physical
    units — only the GP-internal representation changes.

    Returns:
      bounds          : (2, D) physical bounds (min row, max row).
      log_mask        : (D,) bool tensor — true on log-scaled axes.
      from_unit(u)    : (..., D) in [0, 1] → physical units.
      to_unit(x)      : physical → (..., D) in [0, 1] (matches the GP transform).
      input_transform : ready-to-pass module for ``SingleTaskGP``.
    """
    D = len(params)
    lo = torch.tensor([float(p["min"]) for p in params], dtype=DTYPE, device=DEVICE)
    hi = torch.tensor([float(p["max"]) for p in params], dtype=DTYPE, device=DEVICE)
    log_mask = torch.tensor(
        [bool(p.get("log", False)) for p in params],
        dtype=torch.bool, device=DEVICE,
    )
    bounds = torch.stack([lo, hi])
    if log_mask.any():
        bad = log_mask & (lo <= 0)
        if bool(bad.any()):
            bad_names = [params[i]["name"] for i in range(D) if bool(bad[i])]
            raise ValueError(
                "log-scaled params must have min > 0; got non-positive min for: "
                f"{bad_names}"
            )

    # Bounds in the GP's internal normalized space: log10 on log axes, linear elsewhere.
    log_lo = torch.where(log_mask, lo.log10(), lo)
    log_hi = torch.where(log_mask, hi.log10(), hi)
    log_span = log_hi - log_lo

    def from_unit(u: torch.Tensor) -> torch.Tensor:
        x = log_lo + u * log_span
        if log_mask.any():
            x = torch.where(log_mask, torch.pow(torch.tensor(10.0, dtype=x.dtype, device=x.device), x), x)
        return x

    def to_unit(x: torch.Tensor) -> torch.Tensor:
        x_t = torch.where(log_mask, x.log10(), x) if log_mask.any() else x
        return (x_t - log_lo) / log_span

    if log_mask.any():
        log_idx = torch.where(log_mask)[0]
        log_bounds = bounds.clone()
        log_bounds[:, log_idx] = log_bounds[:, log_idx].log10()
        input_transform = ChainedInputTransform(
            log=Log10(indices=log_idx),
            norm=Normalize(d=D, bounds=log_bounds),
        )
    else:
        input_transform = Normalize(d=D, bounds=bounds)

    return bounds, log_mask, from_unit, to_unit, input_transform


def _to_overrides(theta: torch.Tensor, names: List[str]) -> Dict[str, float]:
    return {name: float(theta[i].item()) for i, name in enumerate(names)}


def _evaluate_batch(
    session: forward.ForwardSession,
    targets: List[dict],
    candidates: torch.Tensor,
    names: List[str],
    draw_subset=None,
) -> torch.Tensor:
    """Run the forward sim for each candidate, return -loss as a (q, 1) tensor.

    If ``draw_subset`` is provided, it is called once per candidate to obtain a
    list of batch indices for that single evaluation (batch rotation). When
    ``None``, the simulator uses all batches (current deterministic behavior).
    """
    out = torch.empty(candidates.shape[0], 1, dtype=DTYPE, device=DEVICE)
    for i, theta in enumerate(candidates):
        overrides = _to_overrides(theta, names)
        subset = None if draw_subset is None else draw_subset()
        loss = session.evaluate(overrides, targets, batches_to_use=subset)
        out[i, 0] = -loss
    return out


def run_bo(cfg: dict) -> Tuple[torch.Tensor, torch.Tensor, List[float], dict]:
    bo = cfg["bo"]
    torch.manual_seed(int(bo.get("torch_seed", 0)))
    np.random.seed(int(bo.get("numpy_seed", 0)))

    names = [p["name"] for p in cfg["params"]]
    bounds, log_mask, from_unit, to_unit, input_transform = _build_param_space(cfg["params"])
    D = bounds.shape[-1]
    nominal = torch.tensor(
        [float(p["nominal"]) for p in cfg["params"]], dtype=DTYPE, device=DEVICE
    )
    log_names = [n for n, m in zip(names, log_mask.tolist()) if m]
    if log_names:
        logger.info("Log-normalized params: %s", log_names)

    sim_cfg = forward.SimConfig(input_file=cfg["input_file"], **cfg["sim"])
    loss_cfg = forward.LossConfig(**cfg.get("loss", {}))
    session = forward.ForwardSession(
        sim_cfg=sim_cfg,
        loss_cfg=loss_cfg,
        larndsim_repo=cfg.get("larndsim_repo", ""),
        tunable_params=tuple(names),
    )
    target_path = forward.resolved_target_path(cfg)
    if not os.path.exists(target_path):
        raise FileNotFoundError(
            f"target file not found: {target_path}\n"
            "  - if target_file_auto_suffix is on, regenerate for these sim settings:\n"
            "      sbatch detector_calibration/slurm/run_target.sh\n"
            "  - otherwise check cfg.target_file in the YAML."
        )
    if target_path != cfg.get("target_file"):
        logger.info("target_file_auto_suffix: loading %s", target_path)
    targets = forward.load_target(target_path)
    total_hits = sum(int(t["adcs"].size) for t in targets)
    logger.info(
        "Session ready: %d sim batches, %d target batches, %d total hits, sim_seed_strategy=%s.",
        len(session.dataset), len(targets), total_hits, session.sim_cfg.sim_seed_strategy,
    )

    # Sobol seeding — sample in [0,1]^D and unnormalize via from_unit so the
    # design is uniform in the SAME normalized space the GP sees. On log axes
    # this becomes log-uniform sampling in physical space.
    n_init = int(bo["n_init"])
    n_iter = int(bo["n_iter"])
    q = int(bo.get("q", 1))
    unit_bounds = torch.stack([
        torch.zeros(D, dtype=DTYPE, device=DEVICE),
        torch.ones(D, dtype=DTYPE, device=DEVICE),
    ])
    sobol_unit = (
        draw_sobol_samples(bounds=unit_bounds, n=n_init, q=1, seed=int(bo.get("torch_seed", 0)))
        .squeeze(1)
    )
    train_X = from_unit(sobol_unit).to(dtype=DTYPE, device=DEVICE)

    # ── Batch rotation + noisy GP (optional, default off) ─────────────────────
    # When `bo.batch_rotation: true` each BO evaluation samples a fresh random
    # subset of `batches_per_eval` batch indices. The GP receives the per-eval
    # estimator's variance via `train_Yvar` and models the underlying full-file
    # loss correctly (qLogNoisyExpectedImprovement is the natural acquisition).
    # See "Batch rotation across BO evals + noisy GP" notes in the project doc.
    batch_rotation     = bool(bo.get("batch_rotation", False))
    batches_per_eval   = int(bo.get("batches_per_eval", min(20, len(session.dataset))))
    yvar_source        = str(bo.get("yvar_source", "calibrate")).lower()
    yvar_value         = float(bo.get("yvar_value", 1.0e-4))
    yvar_calibration_n = int(bo.get("yvar_calibration_n", 10))

    # noisy_gp controls whether train_Yvar is passed to SingleTaskGP. It is
    # ORTHOGONAL to batch_rotation: rotation produces noise, noisy_gp tells
    # the GP about it. Default is to follow batch_rotation (the usual choice),
    # but they can be set independently:
    #   - rotation=true,  noisy_gp=true  (typical)  : GP models rotation noise correctly
    #   - rotation=false, noisy_gp=false (typical)  : fully deterministic, vanilla GP
    #   - rotation=false, noisy_gp=true             : "regularization" — GP smooths over a
    #                                                 fixed nominal Yvar (yvar_value);
    #                                                 useful if you suspect residual numerical
    #                                                 noise in the simulator
    #   - rotation=true,  noisy_gp=false (BAD)      : GP fits rotation noise as if it were
    #                                                 signal → lengthscale collapse; warned.
    noisy_gp_cfg = bo.get("noisy_gp", None)   # None | True | False
    noisy_gp = bool(batch_rotation) if noisy_gp_cfg is None else bool(noisy_gp_cfg)
    if noisy_gp and not batch_rotation and yvar_source == "calibrate":
        logger.warning(
            "noisy_gp: true with batch_rotation: false — Yvar calibration needs "
            "rotation. Falling back to yvar_source='fixed' (yvar_value=%g).",
            yvar_value,
        )
        yvar_source = "fixed"
    if batch_rotation and not noisy_gp:
        logger.warning(
            "batch_rotation: true with noisy_gp: false — the GP will treat noisy "
            "rotation observations as deterministic. Lengthscales typically collapse "
            "and BO progress stalls. Set noisy_gp: true unless you know what you're doing."
        )

    n_total_batches = len(session.dataset)
    if batch_rotation:
        if batches_per_eval > n_total_batches:
            raise ValueError(
                f"bo.batches_per_eval={batches_per_eval} exceeds available batches "
                f"({n_total_batches}). Raise sim.max_nbatch or lower batches_per_eval."
            )
        rotation_rng = np.random.default_rng(seed=int(bo.get("torch_seed", 0)))
        def draw_subset():
            return rotation_rng.choice(
                n_total_batches, size=batches_per_eval, replace=False
            ).tolist()
        logger.info(
            "Batch rotation ON: each eval uses %d/%d random batches.",
            batches_per_eval, n_total_batches,
        )
    else:
        draw_subset = None

    # ── Yvar calibration (only when noisy_gp is on) ──────────────────────────
    yvar: Optional[float] = None
    if noisy_gp:
        if yvar_source == "calibrate":
            logger.info(
                "Calibrating Yvar at nominal θ (n_repeats=%d, %d batches per repeat)...",
                yvar_calibration_n, batches_per_eval,
            )
            cal_overrides = {n: float(nominal[i].item()) for i, n in enumerate(names)}
            cal_losses = []
            for k in range(yvar_calibration_n):
                subset = draw_subset() if draw_subset is not None else None
                loss_val = session.evaluate(cal_overrides, targets, batches_to_use=subset)
                cal_losses.append(loss_val)
                logger.info("  cal %2d/%d: loss=%.4e", k + 1, yvar_calibration_n, loss_val)
            yvar = float(np.var(cal_losses, ddof=1))
            logger.info(
                "Yvar calibration done. mean=%.4e, std=%.4e, var (Yvar)=%.4e",
                float(np.mean(cal_losses)),
                float(np.std(cal_losses, ddof=1)),
                yvar,
            )
        elif yvar_source == "fixed":
            yvar = yvar_value
            logger.info("Using fixed Yvar = %.4e (skipping calibration).", yvar)
        else:
            raise ValueError(
                f"Unknown bo.yvar_source={yvar_source!r}. Must be 'calibrate' or 'fixed'."
            )

    # ── Sobol-seed evaluations (with rotation if enabled) ─────────────────────
    train_Y = _evaluate_batch(session, targets, train_X, names, draw_subset=draw_subset)
    train_Yvar = (
        torch.full_like(train_Y, float(yvar)) if (noisy_gp and yvar is not None) else None
    )
    logger.info(
        "Sobol init complete: best -loss = %.6e. Acquisition: %s. %s",
        float(train_Y.max()), bo.get("acquisition", "qLogEI"),
        "Noisy GP (train_Yvar set)" if train_Yvar is not None else "Deterministic GP",
    )

    diagnostics_on = bool(bo.get("diagnostics", True))
    early_stop          = bool(bo.get("early_stop", True))
    early_stop_window   = int(bo.get("early_stop_window", 30))
    early_stop_acq_tol  = float(bo.get("early_stop_acq_tol", 0.1))
    early_stop_diff_tol = float(bo.get("early_stop_diff_tol", 0.01))
    if early_stop and not diagnostics_on:
        logger.warning(
            "bo.early_stop requires bo.diagnostics; auto-enabling diagnostics."
        )
        diagnostics_on = True

    history: List[float] = []
    acq_history: List[float] = []          # populated only when diagnostics_on
    diff_history: List[float] = []
    lengthscale_history: List[List[float]] = []
    early_stopped_at: int = -1             # -1 means ran to completion

    import contextlib
    import warnings as _warnings

    for it in range(n_iter):
        # Capture BoTorch/scipy/gpytorch warnings when diagnostics are on so they
        # land in the slurm log instead of being silently deduplicated by Python's
        # default warning filter. When off, fall through to the default behavior.
        if diagnostics_on:
            warn_ctx = _warnings.catch_warnings(record=True)
            caught = warn_ctx.__enter__()
            _warnings.simplefilter("always")
        else:
            warn_ctx = contextlib.nullcontext()
            caught = None
            warn_ctx.__enter__()

        try:
            # train_Yvar (when batch rotation is on) puts SingleTaskGP into
            # fixed-noise mode: the GP uses these per-observation variances
            # instead of fitting a homoscedastic noise term.
            gp_kwargs = dict(
                input_transform=input_transform,
                outcome_transform=Standardize(m=1),
            )
            if train_Yvar is not None:
                gp_kwargs["train_Yvar"] = train_Yvar
            model = SingleTaskGP(train_X, train_Y, **gp_kwargs)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acq_name = str(bo.get("acquisition", "qLogEI")).lower()
            if acq_name in ("qlognei", "qlognoisyei", "qlognoisyexpectedimprovement"):
                acq = qLogNoisyExpectedImprovement(
                    model=model, X_baseline=train_X, prune_baseline=True,
                )
            elif acq_name in ("qlogei", "qlogexpectedimprovement"):
                acq = qLogExpectedImprovement(model=model, best_f=train_Y.max())
            else:
                raise ValueError(
                    f"Unknown bo.acquisition={acq_name!r}. "
                    "Must be one of: qLogEI, qLogNEI."
                )
            candidate, acq_value = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=q,
                num_restarts=int(bo.get("num_restarts", 10)),
                raw_samples=int(bo.get("raw_samples", 256)),
            )
        finally:
            warn_ctx.__exit__(None, None, None)

        if diagnostics_on:
            for w in (caught or []):
                logger.warning("iter %d | %s: %s",
                               it + 1, w.category.__name__, str(w.message).strip())
            # ARD lengthscales — accessor differs between BoTorch versions.
            # Older: SingleTaskGP wraps the kernel in ScaleKernel, so the
            #        Matérn/RBF base kernel (which owns `lengthscale`) is at
            #        `model.covar_module.base_kernel`.
            # Newer: `covar_module` IS the bare RBFKernel/MaternKernel, with
            #        `lengthscale` as a direct attribute. Probe and fall back.
            covar = model.covar_module
            base_kernel = covar.base_kernel if hasattr(covar, "base_kernel") else covar
            ls = base_kernel.lengthscale.detach().squeeze().cpu().tolist()
            lengthscale_history.append(ls if isinstance(ls, list) else [float(ls)])
            # ||Δx||_2 in NORMALIZED space — uses to_unit so log axes are compared
            # in their log-scaled coordinates, matching what the GP sees.
            prev = train_X[-1]
            diff_norm = float((to_unit(candidate[0]) - to_unit(prev)).norm().item())
            diff_history.append(diff_norm)
            acq_history.append(float(acq_value))

        new_y = _evaluate_batch(session, targets, candidate, names, draw_subset=draw_subset)
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_y], dim=0)
        if train_Yvar is not None:
            train_Yvar = torch.cat(
                [train_Yvar, torch.full_like(new_y, float(yvar))], dim=0
            )

        # Best-so-far selection:
        #   - Deterministic mode: pick the lowest observed loss (current behavior).
        #   - Noisy (rotation) mode: pick the θ the GP THINKS is best, not the
        #     luckiest observation. argmax of the posterior MEAN at train_X is
        #     the standard noisy-BO convention.
        if train_Yvar is None:
            best_idx = int(torch.argmax(train_Y).item())
            best_loss = float(-train_Y[best_idx].item())
        else:
            with torch.no_grad():
                post_mean = model.posterior(train_X).mean.squeeze(-1)
            best_idx = int(torch.argmax(post_mean).item())
            best_loss = float(-post_mean[best_idx].item())
        history.append(best_loss)

        if diagnostics_on:
            flag = ""
            if diff_norm < 1e-3:
                flag += " STUCK_dx"
            if len(acq_history) >= 10 and float(np.std(acq_history[-10:])) < 0.05:
                flag += " ACQ_PLATEAU"
            logger.info(
                "iter %3d/%d | loss=%.4e | best=%.4e | acq=%+.3e | Δx=%.2e%s | x=%s",
                it + 1, n_iter, float(-new_y.min()), best_loss,
                float(acq_value), diff_norm, flag,
                ",".join(f"{v: .4e}" for v in candidate[0].tolist()),
            )
        else:
            logger.info(
                "iter %3d/%d | loss=%.6e | best=%.6e | x=%s",
                it + 1, n_iter, float(-new_y.min()), best_loss,
                ",".join(f"{v: .4e}" for v in candidate[0].tolist()),
            )

        # Early stop: both the acquisition value's recent variability and the
        # average step size have plateaued. Requires diagnostics (otherwise the
        # histories aren't populated). Triggers at most once per run.
        if (
            early_stop
            and diagnostics_on
            and len(acq_history) >= early_stop_window
        ):
            acq_tail_std   = float(np.std(acq_history[-early_stop_window:]))
            diff_tail_mean = float(np.mean(diff_history[-early_stop_window:]))
            if acq_tail_std < early_stop_acq_tol and diff_tail_mean < early_stop_diff_tol:
                logger.info(
                    "early-stop at iter %d/%d (window=%d: acq std=%.3e < %.3e, "
                    "mean Δx=%.3e < %.3e)",
                    it + 1, n_iter, early_stop_window,
                    acq_tail_std, early_stop_acq_tol,
                    diff_tail_mean, early_stop_diff_tol,
                )
                early_stopped_at = it + 1
                break

    # Optional L-BFGS-B polish.
    # IMPORTANT: polish always evaluates on the FULL dataset (no rotation), so the
    # finite-difference gradients used by L-BFGS-B are consistent and the polished
    # `best_loss` is the deterministic loss at the final θ — directly comparable
    # to the existing summary.json convention regardless of train_Yvar.
    polish_record = {}
    if train_Yvar is None:
        best_idx = int(torch.argmax(train_Y).item())
        best_x = train_X[best_idx].clone()
        best_loss_pre = float(-train_Y[best_idx].item())
    else:
        # Pick the best θ by GP posterior mean, not by the noisiest-best
        # observation. Then re-evaluate it on the FULL dataset to get a clean
        # baseline loss for the polish comparison.
        with torch.no_grad():
            post_mean = model.posterior(train_X).mean.squeeze(-1)
        best_idx = int(torch.argmax(post_mean).item())
        best_x = train_X[best_idx].clone()
        best_overrides = _to_overrides(best_x, names)
        best_loss_pre = float(session.evaluate(best_overrides, targets))
        logger.info(
            "Best θ by GP posterior mean: deterministic full-batch loss=%.4e "
            "(GP est=%.4e at this θ).",
            best_loss_pre, float(-post_mean[best_idx].item()),
        )
    best_loss = best_loss_pre
    if bo.get("polish", True):
        scipy_bounds = list(zip(bounds[0].tolist(), bounds[1].tolist()))

        def _loss_np(p: np.ndarray) -> float:
            overrides = {name: float(v) for name, v in zip(names, p)}
            # Polish always uses the full dataset (batches_to_use=None) so the
            # FD gradient is consistent across L-BFGS-B steps.
            return float(session.evaluate(overrides, targets))

        res = minimize(
            _loss_np,
            x0=best_x.cpu().numpy(),
            method="L-BFGS-B",
            bounds=scipy_bounds,
            options={"maxiter": 50},
        )
        polished = torch.tensor(res.x, dtype=DTYPE, device=DEVICE)
        polished_loss = float(res.fun)
        if polished_loss < best_loss:
            best_x = polished
            best_loss = polished_loss
        polish_record = {
            "x": polished.tolist(),
            "loss": polished_loss,
            "success": bool(res.success),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
        }
        logger.info(
            "polish: loss %.6e -> %.6e (%s, %d iters)",
            best_loss_pre, polished_loss, res.success, res.nit,
        )

    summary = {
        "best_x": best_x.tolist(),
        "best_loss": best_loss,
        "names": names,
        "nominal": nominal.tolist(),
        "polish": polish_record,
        "n_iter_completed": len(history),
        "early_stopped_at": int(early_stopped_at),  # -1 if ran to completion
    }
    diagnostics = {
        "enabled": diagnostics_on,
        "acq_history": acq_history,
        "diff_history": diff_history,
        "lengthscale_history": lengthscale_history,
        "n_init": n_init,
    }
    return train_X, train_Y, history, summary, diagnostics


def _save_run(out_dir: str, train_X, train_Y, history, summary, diagnostics) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, f"run_{stamp}")
    os.makedirs(run_dir, exist_ok=True)
    arrays = {
        "train_X": train_X.cpu().numpy(),
        "train_Y": train_Y.cpu().numpy(),
        "history": np.asarray(history),
    }
    if diagnostics.get("enabled", False):
        arrays.update(
            acq_history=np.asarray(diagnostics["acq_history"], dtype=np.float64),
            diff_history=np.asarray(diagnostics["diff_history"], dtype=np.float64),
            lengthscale_history=np.asarray(diagnostics["lengthscale_history"], dtype=np.float64),
            n_init=np.int64(diagnostics["n_init"]),
        )
    np.savez(os.path.join(run_dir, "trace.npz"), **arrays)
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to configs/bo_detector.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_X, train_Y, history, summary, diagnostics = run_bo(cfg)
    run_dir = _save_run(cfg["output_dir"], train_X, train_Y, history, summary, diagnostics)
    logger.info("Saved run to %s", run_dir)
    logger.info("Best params: %s", dict(zip(summary["names"], summary["best_x"])))
    logger.info("Best loss : %.6e", summary["best_loss"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

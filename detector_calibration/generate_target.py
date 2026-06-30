"""Generate the BO target by running the forward sim once at nominal params."""

from __future__ import annotations

import argparse
import logging
import os

import yaml

from . import forward

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _make_session_from_cfg(cfg: dict) -> forward.ForwardSession:
    sim_kwargs = dict(cfg["sim"])
    if "target_seed" in cfg and cfg["target_seed"] is not None:
        sim_kwargs["seed"] = int(cfg["target_seed"])
    if "target_seed_strategy" in cfg and cfg["target_seed_strategy"] is not None:
        sim_kwargs["sim_seed_strategy"] = str(cfg["target_seed_strategy"])
    sim_cfg = forward.SimConfig(input_file=cfg["input_file"], **sim_kwargs)
    loss_cfg = forward.LossConfig(**cfg.get("loss", {}))
    return forward.ForwardSession(
        sim_cfg=sim_cfg,
        loss_cfg=loss_cfg,
        larndsim_repo=cfg.get("larndsim_repo", ""),
        tunable_params=tuple(p["name"] for p in cfg.get("params", [])),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to bo_detector.yaml")
    parser.add_argument(
        "--output",
        default=None,
        help="Override target_file in the config",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Honor target_file_auto_suffix: e.g. target_nominal__nbatch100_maxlen200_*.npz
    out_path = args.output or forward.resolved_target_path(cfg)
    if out_path != cfg.get("target_file") and not args.output:
        logger.info("target_file_auto_suffix: %s -> %s", cfg["target_file"], out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    session = _make_session_from_cfg(cfg)
    logger.info(
        "Loaded %d batches from %s (sim_seed_strategy=%s)",
        len(session.dataset), session.sim_cfg.input_file, session.sim_cfg.sim_seed_strategy,
    )
    targets = session.build_target()
    meta = session.target_meta()
    forward.save_target(targets, out_path, meta=meta)
    per_batch_sizes = [int(t["adcs"].size) for t in targets]
    logger.info(
        "Wrote target to %s (%d batch(es), per-batch hits: %s, total: %d, "
        "tpc_borders shape: %s)",
        out_path, len(targets), per_batch_sizes, sum(per_batch_sizes),
        tuple(meta["tpc_borders"].shape),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

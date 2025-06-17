#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
from omegaconf import OmegaConf

from mpl_tf.data.mnist import BaseDataset, MNISTDataset
from mpl_tf.models.cnn import CNN
from mpl_tf.training.trainer import MPLTrainer


def _parse_args() -> argparse.Namespace:
    """Return parsed CLI args."""
    p = argparse.ArgumentParser(prog="mpl-tf", description="TensorFlow MPL reference")

    sub = p.add_subparsers(dest="command", required=True)

    # -- train ---------------------------------------------------------------
    t = sub.add_parser("train", help="Train student & teacher with MPL")
    t.add_argument("-c", "--config", type=Path, required=True,
                   help="Path to YAML or JSON config file")
    t.add_argument("--resume", action="store_true",
                   help="Resume from last checkpoint in cfg['trainer']['output_dir']")

    # -- eval ----------------------------------------------------------------
    e = sub.add_parser("eval", help="Evaluate a trained checkpoint")
    e.add_argument("--ckpt", type=Path, required=True, help="Path to .h5 or SavedModel dir")
    e.add_argument("--dataset", default="mnist", choices=["mnist"],
                   help="Which evaluation dataset to use")
    e.add_argument("--batch-size", type=int, default=128)

    # -- predict -------------------------------------------------------------
    p_ = sub.add_parser("predict", help="Run inference over a directory of images")
    p_.add_argument("--ckpt", type=Path, required=True)
    p_.add_argument("--input-dir", type=Path, required=True)
    p_.add_argument("--out", type=Path, default=Path("predictions.json"))

    return p.parse_args()


def _load_cfg(path: Path) -> Dict[str, Any]:
    """Load Ω-Conf YAML or a raw JSON config into a plain dict."""
    if path.suffix in {".yml", ".yaml"}:
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if path.suffix == ".json":
        return json.loads(path.read_text())
    raise ValueError(f"Unsupported config format: {path}")


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_cfg(args.config)

    tf.random.set_seed(cfg.get("seed", 42))

    # 1. Data
    if cfg["data"]["dataset"] == "mnist":
        dataset = MNISTDataset(**cfg["data"]["params"])
    else:
        raise ValueError(f"Unrecognized dataset: {cfg['data']['dataset']}")
    
    # 2. Models
    arch = cfg["model"]["model"]
    if arch == "cnn":
        student = CNN(**cfg["model"]["params"]).build()
        teacher = CNN(**cfg["model"]["params"]).build()

    # 3. MPL training loop
    trainer = MPLTrainer(student, teacher, dataset, **cfg["train"])
    trainer.train()


def main() -> None:  # this is what your setuptools entry-point will call
    args = _parse_args()
    match args.command:
        case "train":
            _cmd_train(args)
        # case "eval":
        #     _cmd_eval(args)
        # case "predict":
        #     _cmd_predict(args)
        case _:
            raise RuntimeError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":  
    # mpl-tf train --config configs/mnist.yml
    main()

"""Export a resolved, fingerprinted PPO-STGNN experiment configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from cecoppo.config import TrainConfig
from cecoppo.config_io import save_train_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON file to create")
    parser.add_argument("--seed", type=int, default=42, help="training and environment seed")
    parser.add_argument("--device", default="cpu", help="resolved training device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(device=args.device)
    config.env.seed = args.seed
    fingerprint = save_train_config(config, args.output)
    print(f"Saved {args.output} (config {fingerprint})")


if __name__ == "__main__":
    main()

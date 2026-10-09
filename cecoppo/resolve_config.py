"""Validate and write a fully resolved experiment configuration without torch."""

from __future__ import annotations

import argparse
import json

from .config_io import ExperimentConfigError, load_train_config, save_train_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="partial or complete input JSON")
    parser.add_argument("destination", help="canonical resolved JSON output")
    args = parser.parse_args()
    try:
        config = load_train_config(args.source)
        fingerprint = save_train_config(config, args.destination)
    except ExperimentConfigError as error:
        parser.error(str(error))
    print(json.dumps({
        "synthetic": True,
        "scope": "configuration resolution only; no training or benchmark",
        "fingerprint": fingerprint,
        "output": args.destination,
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

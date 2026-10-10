"""Materialize a deterministic multi-seed experiment plan without torch.

Run: python -m cecoppo.plan_runs config.json plans/ --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from pathlib import Path

from .config_io import ExperimentConfigError, config_fingerprint, load_train_config, save_train_config


def parse_seeds(raw: str) -> list[int]:
    values = []
    for token in raw.split(','):
        token = token.strip()
        if not token or not token.isdigit():
            raise ExperimentConfigError("seeds must be comma-separated unsigned integers")
        seed = int(token)
        if seed >= 2**32:
            raise ExperimentConfigError("seeds must be unsigned 32-bit integers")
        values.append(seed)
    if not values or len(set(values)) != len(values):
        raise ExperimentConfigError("seeds must contain at least one unique value")
    return values


def write_atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    except OSError as error:
        raise ExperimentConfigError(f"Unable to save experiment plan: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def materialize_seed_plan(config_path: str | Path, destination: str | Path, seeds: list[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ExperimentConfigError("seeds must contain at least one unique value")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ExperimentConfigError("seeds must be unsigned 32-bit integers")
    config = load_train_config(config_path)
    output = Path(destination)
    entries = []
    for seed in seeds:
        resolved = copy.deepcopy(config)
        resolved.env.seed = seed
        filename = f"seed-{seed:010d}.json"
        target = output / filename
        fingerprint = save_train_config(resolved, target)
        entries.append({"seed": seed, "file": filename, "fingerprint": fingerprint})
    manifest = {"schema_version": 1, "source": str(Path(config_path)), "seeds": entries}
    write_atomic_json(output / 'seed-plan.json', manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--seeds', required=True, help='comma-separated unsigned 32-bit seeds')
    args = parser.parse_args()
    try:
        manifest = materialize_seed_plan(args.source, args.destination, parse_seeds(args.seeds))
    except ExperimentConfigError as error:
        parser.error(str(error))
    print(json.dumps({"synthetic": True, "scope": "configuration plan only; no training or benchmark", "output": str(Path(args.destination) / 'seed-plan.json'), "seeds": manifest['seeds']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

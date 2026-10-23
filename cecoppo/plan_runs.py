"""Materialize a deterministic multi-seed experiment plan without torch.

Run: python -m cecoppo.plan_runs config.json plans/ --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import copy
import hashlib
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


def _plan_fingerprint(entries: list[dict]) -> str:
    canonical = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]


def materialize_seed_plan(config_path: str | Path, destination: str | Path, seeds: list[int]) -> dict:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ExperimentConfigError("seeds must contain at least one unique value")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32 for seed in seeds):
        raise ExperimentConfigError("seeds must be unsigned 32-bit integers")
    source = Path(config_path)
    config = load_train_config(source)
    output = Path(destination)
    entries = []
    for seed in seeds:
        resolved = copy.deepcopy(config)
        resolved.env.seed = seed
        filename = f"seed-{seed:010d}.json"
        target = output / filename
        fingerprint = save_train_config(resolved, target)
        entries.append({"seed": seed, "file": filename, "fingerprint": fingerprint})
    manifest = {
        "schema_version": 1,
        "source": source.name,
        "source_fingerprint": config_fingerprint(config),
        "plan_fingerprint": _plan_fingerprint(entries),
        "seed_count": len(entries),
        "seeds": entries,
    }
    write_atomic_json(output / 'seed-plan.json', manifest)
    return manifest


def verify_seed_plan(destination: str | Path) -> dict:
    """Check that a materialized plan still matches each resolved config file."""
    directory = Path(destination)
    try:
        manifest = json.loads((directory / 'seed-plan.json').read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ExperimentConfigError(f"Unable to read seed plan: {directory}") from error
    source_fingerprint = manifest.get('source_fingerprint') if isinstance(manifest, dict) else None
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 1
            or not isinstance(manifest.get('seeds'), list)
            or manifest.get('seed_count') != len(manifest.get('seeds', []))
            or not isinstance(manifest.get('source'), str)
            or Path(manifest.get('source', '')).name != manifest.get('source')
            or not isinstance(manifest.get('plan_fingerprint'), str)
            or len(manifest.get('plan_fingerprint', '')) != 16
            or any(char not in '0123456789abcdef' for char in manifest.get('plan_fingerprint', ''))
            or not isinstance(source_fingerprint, str) or len(source_fingerprint) != 12
            or any(char not in '0123456789abcdef' for char in source_fingerprint)):
        raise ExperimentConfigError('Unsupported seed plan schema')
    source_path = directory / manifest['source']
    if source_path.exists():
        try:
            actual_source_fingerprint = config_fingerprint(load_train_config(source_path))
        except ExperimentConfigError as error:
            raise ExperimentConfigError('Seed plan source config is invalid') from error
        if actual_source_fingerprint != source_fingerprint:
            raise ExperimentConfigError('Seed plan source fingerprint mismatch')
    verified = []
    seen = set()
    for entry in manifest['seeds']:
        if not isinstance(entry, dict) or not isinstance(entry.get('seed'), int) or entry['seed'] in seen:
            raise ExperimentConfigError('Seed plan contains invalid or duplicate seeds')
        filename, expected = entry.get('file'), entry.get('fingerprint')
        if not isinstance(filename, str) or Path(filename).name != filename or not isinstance(expected, str):
            raise ExperimentConfigError('Seed plan contains an unsafe file reference')
        expected_filename = f"seed-{entry['seed']:010d}.json"
        if filename != expected_filename:
            raise ExperimentConfigError(f'Seed plan filename is not canonical: {filename}')
        resolved = load_train_config(directory / filename)
        if resolved.env.seed != entry['seed']:
            raise ExperimentConfigError(f'Seed plan entry does not match resolved config: {filename}')
        actual = config_fingerprint(resolved)
        if actual != expected:
            raise ExperimentConfigError(f'Seed plan fingerprint mismatch: {filename}')
        seen.add(entry['seed'])
        verified.append({'seed': entry['seed'], 'file': filename, 'fingerprint': actual})
    expected_files = {entry['file'] for entry in verified}
    unexpected_files = sorted(
        path.name for path in directory.glob('seed-*.json')
        if path.name != 'seed-plan.json' and path.name not in expected_files
    )
    if unexpected_files:
        raise ExperimentConfigError(
            f'Seed plan contains unexpected materialized files: {", ".join(unexpected_files)}'
        )
    if [entry['seed'] for entry in verified] != sorted(entry['seed'] for entry in verified):
        raise ExperimentConfigError('Seed plan entries must be sorted by seed')
    if manifest.get('plan_fingerprint') != _plan_fingerprint(verified):
        raise ExperimentConfigError('Seed plan manifest fingerprint mismatch')
    return {'schema_version': 1, 'source_fingerprint': manifest.get('source_fingerprint'), 'plan_fingerprint': manifest['plan_fingerprint'], 'seeds': verified}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', nargs='?')
    parser.add_argument('destination', nargs='?')
    parser.add_argument('--seeds', help='comma-separated unsigned 32-bit seeds')
    parser.add_argument('--verify', action='store_true', help='verify an existing plan in destination')
    args = parser.parse_args()
    try:
        if args.verify:
            destination = args.destination or args.source
            if destination is None or args.destination is not None:
                parser.error('--verify accepts exactly one destination')
            manifest = verify_seed_plan(destination)
        else:
            if args.source is None or args.destination is None or args.seeds is None:
                parser.error('source and destination are required unless --verify is used')
            manifest = materialize_seed_plan(args.source, args.destination, parse_seeds(args.seeds))
    except ExperimentConfigError as error:
        parser.error(str(error))
    output = args.destination or args.source
    print(json.dumps({"synthetic": True, "scope": "configuration plan only; no training or benchmark", "output": str(Path(output) / 'seed-plan.json'), "seeds": manifest['seeds']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

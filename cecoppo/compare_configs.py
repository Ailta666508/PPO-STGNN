"""Compare saved experiment settings without torch or private datasets.

Exit codes: 0 compatible, 1 unapproved change, 2 invalid input.
"""
from __future__ import annotations
import argparse
import json
from collections.abc import Mapping
from .config_io import config_changes, config_fingerprint, load_train_config


def _leaf_paths(value: object, prefix: str = "") -> set[str]:
    if isinstance(value, Mapping):
        paths: set[str] = set()
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_leaf_paths(child, path))
        return paths
    return {prefix}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    parser.add_argument(
        '--allow', action='append', default=[], metavar='FIELD',
        help='approve a changed leaf path (repeatable, for example env.seed)',
    )
    args = parser.parse_args()
    try:
        reference = load_train_config(args.reference)
        candidate = load_train_config(args.candidate)
        changes = config_changes(reference, candidate)
        known_paths = _leaf_paths(reference.to_dict())
        unknown_allow = sorted(set(args.allow) - known_paths)
        if unknown_allow:
            parser.error(f"unknown --allow field: {', '.join(unknown_allow)}")
        approved = {key: value for key, value in changes.items() if key in args.allow}
        unapproved = {key: value for key, value in changes.items() if key not in args.allow}
        encode = lambda items: {
            key: {'before': before, 'after': after}
            for key, (before, after) in items.items()
        }
        report = {
            'reference_fingerprint': config_fingerprint(reference),
            'candidate_fingerprint': config_fingerprint(candidate),
            'identical': not changes,
            'compatible': not unapproved,
            'approved_changes': encode(approved),
            'unapproved_changes': encode(unapproved),
            'changes': encode(changes),
        }
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return int(bool(unapproved))


if __name__ == '__main__':
    raise SystemExit(main())

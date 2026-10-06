"""Compare saved experiment settings without torch or private datasets.

Exit codes: 0 identical, 1 changed, 2 invalid input.
"""
from __future__ import annotations
import argparse
import json
from .config_io import config_changes, config_fingerprint, load_train_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference')
    parser.add_argument('candidate')
    args = parser.parse_args()
    try:
        reference = load_train_config(args.reference)
        candidate = load_train_config(args.candidate)
        changes = config_changes(reference, candidate)
        report = {
            'reference_fingerprint': config_fingerprint(reference),
            'candidate_fingerprint': config_fingerprint(candidate),
            'identical': not changes,
            'changes': {key: {'before': before, 'after': after}
                        for key, (before, after) in changes.items()},
        }
        print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False))
    except (ValueError, TypeError) as error:
        parser.error(str(error))
    return int(bool(changes))


if __name__ == '__main__':
    raise SystemExit(main())

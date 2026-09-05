"""Verify packaged source, paper assets, and Python syntax without execution."""

import ast
import hashlib
import json
from pathlib import Path


PUBLICATION_ASSET_SHA256 = {
    "docs/paper.pdf": "95da53a10ceeeee60272388cf80e9346543371772ebf86ff4d0f3541f856ea67",
    "docs/assets/figure-1-system-model.jpg": "65623fee4b2498d1cee9ba2153bbbd58caa0f3ab224fd251b2089c75fd564d9b",
    "docs/assets/figure-2-ppo-stgnn-framework.jpg": "8e7b942e34a9dd992f193be8d9ccbdbb14978be0295c0802254655f9cc9e8a4d",
    "docs/assets/figure-3-baseline-comparison.png": "f89e00698ec5b82ac5467f6e5c613ee79b1f4f1b6a2f19bc9b61e994bb48c5f2",
    "docs/assets/figure-4-encoder-comparison.png": "e53f2def37684e02734483b4db5169d19e6850906fe9f8f7deb6f82e46e89090",
}


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/source-manifest.json").read_text(encoding="utf-8"))
    errors = []
    python_files = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            errors.append(f"Missing: {entry['path']}")
            continue
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            errors.append(f"Changed: {entry['path']}")
        if path.suffix == ".py":
            try:
                ast.parse(content, filename=entry["path"])
                python_files += 1
            except SyntaxError as error:
                errors.append(f"Syntax error: {entry['path']}: {error}")

    for relative_path, expected_sha256 in PUBLICATION_ASSET_SHA256.items():
        path = root / relative_path
        if not path.is_file():
            errors.append(f"Missing publication asset: {relative_path}")
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            errors.append(f"Changed publication asset: {relative_path}")

    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} maintained files; parsed {python_files} Python files.")
    print(f"Verified the paper and {len(PUBLICATION_ASSET_SHA256) - 1} embedded paper figures.")
    print("This checks packaged-file integrity, not experiment reproduction or historical provenance.")


if __name__ == "__main__":
    main()

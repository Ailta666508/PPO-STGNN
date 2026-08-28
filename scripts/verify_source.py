"""Verify the original packaged files and parse Python sources without execution."""

import ast
import hashlib
import json
from pathlib import Path


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
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest['files'])} original files; parsed {python_files} Python files.")
    print("This checks packaged-file integrity, not experiment reproduction or historical provenance.")


if __name__ == "__main__":
    main()

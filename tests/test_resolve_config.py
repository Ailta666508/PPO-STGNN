import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cecoppo.config_io import load_train_config


class ResolveConfigCliTests(unittest.TestCase):
    def test_resolves_defaults_and_writes_a_reusable_canonical_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "partial.json"
            output = Path(directory) / "resolved.json"
            source.write_text('{"env":{"seed":17},"ppo":{"lr":0.0002}}')
            result = subprocess.run(
                [sys.executable, "-m", "cecoppo.resolve_config", str(source), str(output)],
                capture_output=True, text=True, check=True,
            )
            report = json.loads(result.stdout)
            resolved = load_train_config(output)
            self.assertTrue(report["synthetic"])
            self.assertEqual(len(report["fingerprint"]), 12)
            self.assertEqual(resolved.env.seed, 17)
            self.assertEqual(resolved.ppo.lr, 0.0002)
            self.assertIn("max_nodes", json.loads(output.read_text())["env"])

    def test_invalid_config_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"
            output = Path(directory) / "resolved.json"
            source.write_text('{"env":{"seed":-1}}')
            result = subprocess.run(
                [sys.executable, "-m", "cecoppo.resolve_config", str(source), str(output)],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())
            self.assertNotIn("Traceback", result.stderr)

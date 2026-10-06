import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class CompareConfigCliTests(unittest.TestCase):
    def run_cli(self, left, right):
        return subprocess.run([sys.executable, '-m', 'cecoppo.compare_configs', str(left), str(right)], capture_output=True, text=True)

    def test_resolved_defaults_and_changed_values(self):
        with tempfile.TemporaryDirectory() as d:
            left, right = Path(d)/'left.json', Path(d)/'right.json'
            left.write_text('{}'); right.write_text('{"env":{"seed":42}}')
            result = self.run_cli(left, right)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(json.loads(result.stdout)['identical'])
            right.write_text('{"env":{"seed":17},"ppo":{"lr":0.002}}')
            result = self.run_cli(left, right)
            self.assertEqual(result.returncode, 1, result.stderr)
            changes = json.loads(result.stdout)['changes']
            self.assertEqual(changes['env.seed'], {'before':42, 'after':17})
            self.assertEqual(set(changes), {'env.seed', 'ppo.lr'})

    def test_bad_input_exits_two_without_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'bad.json'
            for raw in ['{', '{"env":{"sed":3}}', '{"ppo":{"lr":NaN}}']:
                path.write_text(raw)
                result = self.run_cli(path, path)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn('Traceback', result.stderr)
            result = self.run_cli(path, Path(d)/'missing.json')
            self.assertEqual(result.returncode, 2)

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cecoppo.config_io import ExperimentConfigError
from cecoppo.plan_runs import materialize_seed_plan, parse_seeds


class SeedPlanTests(unittest.TestCase):
    def test_materializes_independent_resolved_configs_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            source.write_text(json.dumps({'env': {'seed': 9}, 'ppo': {'lr': 0.0002}}))
            manifest = materialize_seed_plan(source, Path(directory) / 'plans', [42, 43])
            self.assertEqual([entry['seed'] for entry in manifest['seeds']], [42, 43])
            self.assertEqual(json.loads((Path(directory) / 'plans/seed-0000000042.json').read_text())['env']['seed'], 42)
            self.assertEqual(json.loads((Path(directory) / 'plans/seed-0000000043.json').read_text())['env']['seed'], 43)
            self.assertEqual(json.loads((Path(directory) / 'plans/seed-plan.json').read_text()), manifest)
            self.assertNotEqual(manifest['seeds'][0]['fingerprint'], manifest['seeds'][1]['fingerprint'])

    def test_cli_reports_data_free_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            source.write_text('{}')
            result = subprocess.run([sys.executable, '-m', 'cecoppo.plan_runs', str(source), str(Path(directory) / 'plans'), '--seeds', '1,2'], capture_output=True, text=True, check=True)
            report = json.loads(result.stdout)
            self.assertTrue(report['synthetic'])
            self.assertEqual([entry['seed'] for entry in report['seeds']], [1, 2])

    def test_rejects_duplicate_malformed_and_out_of_range_seeds(self):
        with self.assertRaisesRegex(ExperimentConfigError, 'unique'):
            parse_seeds('1,1')
        with self.assertRaisesRegex(ExperimentConfigError, 'unsigned'):
            parse_seeds('-1')
        with self.assertRaisesRegex(ExperimentConfigError, 'unsigned'):
            parse_seeds('4294967296')

    def test_rejects_duplicate_seed_plan_before_creating_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ExperimentConfigError, 'unique'):
                materialize_seed_plan(Path(directory) / 'missing.json', Path(directory) / 'plans', [1, 1])
            self.assertFalse((Path(directory) / 'plans').exists())

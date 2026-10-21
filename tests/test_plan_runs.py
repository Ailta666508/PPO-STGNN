import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cecoppo.config_io import ExperimentConfigError
from cecoppo.plan_runs import materialize_seed_plan, parse_seeds, verify_seed_plan


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
            self.assertIsInstance(manifest['source_fingerprint'], str)
            self.assertIsInstance(manifest['plan_fingerprint'], str)
            self.assertEqual(manifest['source'], 'source.json')
            self.assertEqual(manifest['seed_count'], 2)
            self.assertNotIn(directory, json.dumps(manifest))

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

    def test_verifies_plan_and_rejects_tampered_resolved_config(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, Path(directory) / 'plans', [42])
            report = verify_seed_plan(Path(directory) / 'plans')
            self.assertEqual(report['seeds'][0]['seed'], 42)
            target = Path(directory) / 'plans/seed-0000000042.json'
            target.write_text(json.dumps({'env': {'seed': 42}, 'ppo': {'lr': 0.9}}))
            with self.assertRaisesRegex(ExperimentConfigError, 'fingerprint mismatch'):
                verify_seed_plan(Path(directory) / 'plans')

    def test_cli_verifies_plan_using_single_positional_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            result = subprocess.run([sys.executable, '-m', 'cecoppo.plan_runs', '--verify', str(plans)], capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout)['seeds'][0]['seed'], 42)

    def test_verifier_rejects_manifest_seed_that_disagrees_with_config(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['seeds'][0]['seed'] = 43
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'does not match'):
                verify_seed_plan(plans)

    def test_verifier_rejects_tampered_plan_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'
            manifest = json.loads(manifest_path.read_text())
            manifest['plan_fingerprint'] = 'tampered'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_missing_source_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'
            plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'
            manifest = json.loads(manifest_path.read_text())
            manifest.pop('source_fingerprint')
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_seed_count_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text()); manifest['seed_count'] = 2
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_reordered_seed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42, 43])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['seeds'].reverse(); manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'sorted'):
                verify_seed_plan(plans)

    def test_verifier_rejects_unsafe_source_and_noncanonical_plan_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = Path(directory) / 'plans'; plans.mkdir()
            (plans / 'seed-plan.json').write_text(json.dumps({
                'schema_version': 1, 'source': '../config.json',
                'source_fingerprint': '0' * 12, 'plan_fingerprint': '0' * 15,
                'seed_count': 0, 'seeds': [],
            }))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_detects_tampered_source_when_it_is_available(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            (plans / 'source.json').write_text(json.dumps({'env': {'seed': 10}}))
            with self.assertRaisesRegex(ExperimentConfigError, 'source fingerprint'):
                verify_seed_plan(plans)

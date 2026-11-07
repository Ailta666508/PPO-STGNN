import json
import os
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
        with self.assertRaisesRegex(ExperimentConfigError, 'unsigned'):
            parse_seeds('４２')
        self.assertEqual(parse_seeds('43,2,42'), [2, 42, 43])

    def test_materializer_canonicalizes_unsorted_programmatic_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            manifest = materialize_seed_plan(source, plans, [43, 42])
            self.assertEqual([entry['seed'] for entry in manifest['seeds']], [42, 43])
            self.assertEqual(verify_seed_plan(plans)['seeds'], manifest['seeds'])

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
            (plans / 'seed-0000000042.json').rename(plans / 'seed-0000000043.json')
            manifest['seeds'][0]['file'] = 'seed-0000000043.json'
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

    def test_verifier_rejects_empty_seed_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            plans = Path(directory) / 'plans'; plans.mkdir()
            (plans / 'seed-plan.json').write_text(json.dumps({
                'schema_version': 1, 'source': 'source.json',
                'source_fingerprint': '0' * 12, 'plan_fingerprint': '0' * 16,
                'seed_count': 0, 'seeds': [],
            }))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_unrecognized_manifest_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['unexpected'] = 'unverified metadata'; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_boolean_schema_version(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['schema_version'] = True; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_boolean_seed_count(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['seed_count'] = True; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_empty_source_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['source'] = ''; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'Unsupported seed plan schema'):
                verify_seed_plan(plans)

    def test_verifier_rejects_boolean_and_out_of_range_manifest_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['seeds'][0]['seed'] = True; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'invalid or duplicate'):
                verify_seed_plan(plans)
            manifest['seeds'][0]['seed'] = 2**32; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'invalid or duplicate'):
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

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symbolic links unavailable')
    def test_verifier_rejects_symbolic_link_seed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            target = plans / 'seed-0000000042.json'
            real = plans / 'seed-real.json'
            real.write_text(target.read_text())
            target.unlink()
            try:
                target.symlink_to(real.name)
            except OSError as error:
                self.skipTest(f'symbolic links unavailable: {error}')
            with self.assertRaisesRegex(ExperimentConfigError, 'symbolic link'):
                verify_seed_plan(plans)

    def test_verifier_rejects_noncanonical_seed_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['seeds'][0]['file'] = 'seed-42.json'
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'canonical'):
                verify_seed_plan(plans)

    def test_verifier_rejects_noncanonical_entry_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}})); materialize_seed_plan(source, plans, [42])
            manifest_path = plans / 'seed-plan.json'; manifest = json.loads(manifest_path.read_text())
            manifest['seeds'][0]['fingerprint'] = 'not-a-fingerprint'; manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ExperimentConfigError, 'unsafe file reference'):
                verify_seed_plan(plans)

    def test_verifier_rejects_stale_materialized_seed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            materialize_seed_plan(source, plans, [42])
            (plans / 'seed-0000000043.json').write_text((plans / 'seed-0000000042.json').read_text())
            with self.assertRaisesRegex(ExperimentConfigError, 'unexpected materialized files'):
                verify_seed_plan(plans)

    def test_materializer_refuses_to_extend_a_directory_with_stale_seed_files(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; plans = Path(directory) / 'plans'
            source.write_text(json.dumps({'env': {'seed': 9}}))
            plans.mkdir()
            stale = plans / 'seed-0000000043.json'
            stale.write_text('{}')
            with self.assertRaisesRegex(ExperimentConfigError, 'stale seed files'):
                materialize_seed_plan(source, plans, [42])
            self.assertEqual(stale.read_text(), '{}')
            self.assertFalse((plans / 'seed-0000000042.json').exists())

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symbolic links unavailable')
    def test_materializer_rejects_symbolic_link_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.json'; source.write_text(json.dumps({'env': {'seed': 9}}))
            plans = Path(directory) / 'plans'; real = Path(directory) / 'real-plans'; real.mkdir()
            try:
                plans.symlink_to(real, target_is_directory=True)
            except OSError as error:
                self.skipTest(f'symbolic links unavailable: {error}')
            with self.assertRaisesRegex(ExperimentConfigError, 'destination'):
                materialize_seed_plan(source, plans, [42])

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symbolic links unavailable')
    def test_materializer_rejects_symbolic_link_source(self):
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / 'real.json'; real.write_text(json.dumps({'env': {'seed': 9}}))
            source = Path(directory) / 'source.json'
            try:
                source.symlink_to(real)
            except OSError as error:
                self.skipTest(f'symbolic links unavailable: {error}')
            with self.assertRaisesRegex(ExperimentConfigError, 'source'):
                materialize_seed_plan(source, Path(directory) / 'plans', [42])

    @unittest.skipUnless(hasattr(os, 'symlink'), 'symbolic links unavailable')
    def test_materializer_rejects_symlinked_parent_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'source.json'; source.write_text(json.dumps({'env': {'seed': 9}}))
            real = root / 'real'; real.mkdir(); redirected = root / 'redirected'
            try:
                redirected.symlink_to(real, target_is_directory=True)
            except OSError as error:
                self.skipTest(f'symbolic links unavailable: {error}')
            with self.assertRaisesRegex(ExperimentConfigError, 'traverse'):
                materialize_seed_plan(source, redirected / 'plans', [42])

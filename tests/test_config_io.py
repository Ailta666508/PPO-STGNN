import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cecoppo.config import TrainConfig
from cecoppo.config_io import (
    ExperimentConfigError,
    config_fingerprint,
    config_changes,
    load_train_config,
    save_train_config,
    train_config_from_dict,
    validate_train_config,
)


class ExperimentConfigIoTests(unittest.TestCase):
    def test_rejects_semantically_unsafe_training_settings(self):
        config = TrainConfig()
        for field, value, message in [
            ("seed", -1, "env.seed"),
            ("split", "production", "env.split"),
            ("reward_mode", "unknown", "env.reward_mode"),
        ]:
            with self.subTest(field=field):
                setattr(config.env, field, value)
                with self.assertRaisesRegex(ExperimentConfigError, message):
                    validate_train_config(config)
                config = TrainConfig()
        config.ppo.lr = 0.0
        with self.assertRaisesRegex(ExperimentConfigError, "ppo.lr"):
            validate_train_config(config)

    def test_rejects_invalid_checkpoint_schedule_and_directory(self):
        for field, value, message in [
            ("eval_episodes", 0, "eval_episodes"),
            ("save_every", 0, "save_every"),
            ("checkpoint_dir", "", "checkpoint_dir"),
        ]:
            with self.subTest(field=field):
                config = TrainConfig()
                setattr(config, field, value)
                with self.assertRaisesRegex(ExperimentConfigError, message):
                    validate_train_config(config)

    def test_config_changes_reports_actionable_leaf_paths(self):
        reference = TrainConfig()
        candidate = TrainConfig()
        candidate.env.seed = 17
        candidate.ppo.lr = 0.0002
        self.assertEqual(
            config_changes(reference, candidate),
            {"env.seed": (42, 17), "ppo.lr": (0.0001, 0.0002)},
        )

    def test_config_changes_is_empty_for_equivalent_configs(self):
        self.assertEqual(config_changes(TrainConfig(), TrainConfig()), {})

    def test_failed_replace_preserves_existing_config_and_removes_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_train_config(TrainConfig(), path)
            original = path.read_bytes()
            with patch("cecoppo.config_io.os.replace", side_effect=OSError("synthetic failure")):
                with self.assertRaisesRegex(ExperimentConfigError, "Unable to save"):
                    save_train_config(TrainConfig(eval_episodes=1), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_failed_sync_preserves_existing_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_train_config(TrainConfig(), path)
            original = path.read_bytes()
            with patch("cecoppo.config_io.os.fsync", side_effect=OSError("synthetic failure")):
                with self.assertRaises(ExperimentConfigError):
                    save_train_config(TrainConfig(eval_episodes=1), path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_nonfinite_config_does_not_overwrite_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            save_train_config(TrainConfig(), path)
            original = path.read_bytes()
            invalid = TrainConfig()
            invalid.ppo.lr = float("nan")
            with self.assertRaises(ValueError):
                save_train_config(invalid, path)
            self.assertEqual(path.read_bytes(), original)

    def test_invalid_utf8_reports_config_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_bytes(b"\xff")
            with self.assertRaises(ExperimentConfigError):
                load_train_config(path)

    def test_round_trip_preserves_resolved_settings_and_fingerprint(self):
        config = TrainConfig(device="cpu", eval_episodes=3)
        config.env.seed = 17
        config.ppo.hidden_dim = 64

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run" / "config.json"
            written_fingerprint = save_train_config(config, path)
            restored = load_train_config(path)

        self.assertEqual(restored.to_dict(), config.to_dict())
        self.assertEqual(written_fingerprint, config_fingerprint(restored))
        self.assertEqual(len(written_fingerprint), 12)

    def test_rejects_unknown_fields_instead_of_ignoring_typos(self):
        payload = TrainConfig().to_dict()
        payload["env"]["sed"] = payload["env"].pop("seed")

        with self.assertRaisesRegex(ExperimentConfigError, "Unknown env fields: sed"):
            train_config_from_dict(payload)

    def test_reports_malformed_json_with_source_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps({"env": {}})[:-1], encoding="utf-8")
            with self.assertRaisesRegex(ExperimentConfigError, str(path)):
                load_train_config(path)


if __name__ == "__main__":
    unittest.main()

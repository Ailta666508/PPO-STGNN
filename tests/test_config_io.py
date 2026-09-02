import json
import tempfile
import unittest
from pathlib import Path

from cecoppo.config import TrainConfig
from cecoppo.config_io import (
    ExperimentConfigError,
    config_fingerprint,
    load_train_config,
    save_train_config,
    train_config_from_dict,
)


class ExperimentConfigIoTests(unittest.TestCase):
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

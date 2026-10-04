import json
import subprocess
import sys

import pytest
from cecoppo.synthetic_run import run_synthetic


@pytest.mark.parametrize("encoder", ["stgnn", "static_gnn", "mlp"])
def test_quick_run_repeats_exactly_and_updates_policy(encoder):
    first = run_synthetic(seed=17, encoder=encoder)
    second = run_synthetic(seed=17, encoder=encoder)
    assert first == second
    assert first["synthetic"] and first["parameters_updated"] and first["checkpoint_round_trip"]
    assert all(action % 3 != 1 for action in first["actions"])


def test_cli_runs_without_private_data():
    result = subprocess.run([sys.executable, "-m", "cecoppo.synthetic_run", "--encoder", "mlp"], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)["checkpoint_round_trip"]


@pytest.mark.parametrize("kwargs", [{"steps": 1}, {"steps": True}, {"seed": -1}, {"encoder": "unknown"}])
def test_invalid_options(kwargs):
    with pytest.raises(ValueError):
        run_synthetic(**kwargs)

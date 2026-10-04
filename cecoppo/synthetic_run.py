"""Data-free PPO interface exercise; not scheduling or research results.

Run: python -m cecoppo.synthetic_run --encoder stgnn --seed 42 --steps 8
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from .config import PPOConfig, TrainConfig
from .config_io import config_fingerprint
from .ppo_agent import PPOAgent
from .utils import set_seed


def synthetic_observation(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shapes = {
        "resource_x": (5, 38, 14), "resource_edge_attr": (5, 38, 38, 4),
        "resource_time_attr": (5, 38, 3), "dag_x": (48, 10),
        "current_task_x": (10,), "interaction_x": (38, 3),
        "ready_task_x": (4, 10), "pair_interaction_x": (4, 38, 3),
        "global_x": (8,),
    }
    obs = {key: rng.random(shape).astype(np.float32) for key, shape in shapes.items()}
    obs["resource_adj"] = np.repeat(np.eye(38, dtype=np.float32)[None], 5, axis=0)
    obs["dag_adj"] = np.eye(48, dtype=np.float32)
    obs["ready_task_mask"] = np.ones(4, dtype=np.float32)
    # Deliberately mask several actions to exercise the actual policy boundary.
    obs["action_mask"] = np.ones(153, dtype=np.float32)
    obs["action_mask"][1::3] = 0
    return obs


def run_synthetic(seed: int = 42, encoder: str = "stgnn", steps: int = 8) -> dict:
    if encoder not in {"stgnn", "static_gnn", "mlp"}:
        raise ValueError("unsupported encoder")
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 128:
        raise ValueError("steps must be an integer between 2 and 128")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be a 32-bit unsigned integer")
    set_seed(seed)
    torch.set_num_threads(1)
    obs = synthetic_observation(seed)
    config = TrainConfig(ppo=PPOConfig(train_iters=1, minibatch_size=8, hidden_dim=32))
    config.env.seed = seed
    config.env.max_ready_tasks = 4
    agent = PPOAgent(obs, len(obs["action_mask"]), config.ppo.hidden_dim, config.ppo, encoder_type=encoder)
    before = [p.detach().clone() for p in agent.model.parameters()]
    actions = []
    for step in range(steps):
        action, log_prob, value = agent.act(obs)
        if not obs["action_mask"][action]:
            raise RuntimeError("policy selected a masked action")
        actions.append(action)
        # Synthetic alternating rewards exercise GAE and PPO; no private trace input.
        agent.store(obs, action, float(step % 2), step == steps - 1, log_prob, value)
    losses = agent.update(last_value=0.0)
    if not all(np.isfinite(v) for v in losses.values()):
        raise RuntimeError("nonfinite PPO loss")
    changed = any(not torch.equal(a, b) for a, b in zip(before, agent.model.parameters()))
    if not changed:
        raise RuntimeError("PPO update did not change model parameters")
    expected, _ = agent.action_distribution(obs)
    with tempfile.TemporaryDirectory(prefix="ppo-synthetic-") as directory:
        checkpoint = str(Path(directory) / "synthetic-policy.pt")
        agent.save(checkpoint)
        restored = PPOAgent(obs, len(obs["action_mask"]), config.ppo.hidden_dim, config.ppo, encoder_type=encoder)
        restored.load(checkpoint)
        actual, _ = restored.action_distribution(obs)
        np.testing.assert_array_equal(actual, expected)
    return {
        "synthetic": True,
        "scope": "PPO tensor-interface validation; not simulator evaluation or research results",
        "seed": seed, "encoder": encoder, "steps": steps,
        "config_fingerprint": config_fingerprint(config), "actions": actions,
        "losses": losses, "parameters_updated": changed,
        "checkpoint_round_trip": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--encoder", choices=["stgnn", "static_gnn", "mlp"], default="stgnn")
    args = parser.parse_args()
    try:
        result = run_synthetic(args.seed, args.encoder, args.steps)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

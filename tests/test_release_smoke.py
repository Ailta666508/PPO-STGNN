"""Small CPU checks for the released data, simulator, and learning interfaces.

These are software smoke tests, not a reproduction of the paper's experiments.
"""

import json
import os
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

from cecoppo.baselines import FCFSPolicy, GreedyPolicy, HEFTPolicy, LeastLoadPolicy
from cecoppo.config import EnvConfig, PPOConfig
from cecoppo.env_cec_dag import CloudEdgeDagEnv
from cecoppo.ppo_agent import PPOAgent


DATA_DIR = Path(os.environ.get(
    "PPO_STGNN_DATA_DIR", Path(__file__).resolve().parents[1] / "new500DAG"
))


def read_jobs(split):
    with (DATA_DIR / f"jobs_{split}.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_dataset_splits_and_dag_integrity():
    if not (DATA_DIR / "meta.json").is_file():
        pytest.skip("Processed data is not redistributed; set PPO_STGNN_DATA_DIR to a local copy.")
    meta = json.loads((DATA_DIR / "meta.json").read_text(encoding="utf-8"))
    jobs = {split: read_jobs(split) for split in ("all", "train", "val", "test")}
    by_id = {}
    for split, rows in jobs.items():
        by_id[split] = {row["job_name"]: row for row in rows}
        assert len(by_id[split]) == len(rows) == meta[f"num_jobs_{split}"]
    for left, right in combinations(("train", "val", "test"), 2):
        assert by_id[left].keys().isdisjoint(by_id[right])
    merged = {**by_id["train"], **by_id["val"], **by_id["test"]}
    assert merged == by_id["all"]

    for job in jobs["all"]:
        names = {node["task_name"] for node in job["nodes"]}
        assert len(names) == len(job["nodes"]) == job["num_nodes"]
        graph = nx.DiGraph()
        graph.add_nodes_from(names)
        for source, target, weight in job["edges"]:
            assert source in names and target in names
            assert np.isfinite(weight) and weight >= 0
            graph.add_edge(source, target)
        assert nx.is_directed_acyclic_graph(graph)
        order = {name: i for i, name in enumerate(job["topo_order"])}
        assert set(order) == names
        assert all(order[source] < order[target] for source, target in graph.edges)


@pytest.fixture(scope="module")
def env():
    if not (DATA_DIR / "meta.json").is_file():
        pytest.skip("Processed data is not redistributed; set PPO_STGNN_DATA_DIR to a local copy.")
    torch.set_num_threads(1)
    config = EnvConfig(
        split="train", seed=42, episode_jobs=2,
        max_ready_tasks=4, max_steps_per_episode=64,
    )
    return CloudEdgeDagEnv(DATA_DIR, config)


@pytest.mark.parametrize("policy_type", [FCFSPolicy, LeastLoadPolicy, HEFTPolicy, GreedyPolicy])
def test_baselines_select_valid_actions_and_step(env, policy_type):
    obs = env.reset()
    policy = policy_type()
    for _ in range(16):
        assert all(np.isfinite(value).all() for value in obs.values())
        action = int(policy.select_action(obs, env=env))
        assert 0 <= action < env.action_dim
        assert obs["action_mask"][action] > 0
        obs, reward, done, info = env.step(action)
        assert np.isfinite(reward)
        assert not info.get("invalid_action", False)
        if done:
            break


def synthetic_observation():
    """Build non-sensitive synthetic tensors matching the simulator interface."""
    rng = np.random.default_rng(42)
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
    obs["action_mask"] = np.ones(153, dtype=np.float32)
    return obs


@pytest.mark.parametrize("encoder", ["stgnn", "static_gnn", "mlp"])
def test_encoder_masking_and_ppo_update(encoder):
    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    obs = synthetic_observation()
    action_dim = len(obs["action_mask"])
    config = PPOConfig(train_iters=1, minibatch_size=8, hidden_dim=32)
    agent = PPOAgent(obs, action_dim, config.hidden_dim, config, encoder_type=encoder)

    # A single allowed action must have all probability mass for every encoder.
    restricted_obs = {key: value.copy() for key, value in obs.items()}
    allowed = int(np.flatnonzero(obs["action_mask"] > 0)[0])
    restricted_obs["action_mask"][:] = 0
    restricted_obs["action_mask"][allowed] = 1
    probs, _ = agent.action_distribution(restricted_obs)
    assert probs.shape == (action_dim,)
    assert probs[allowed] == pytest.approx(1.0)
    assert np.count_nonzero(probs) == 1

    before = [parameter.detach().clone() for parameter in agent.model.parameters()]
    for step in range(8):
        action, log_prob, value = agent.act(obs)
        assert obs["action_mask"][action] > 0
        assert np.isfinite([log_prob, value]).all()
        agent.store(obs, action, float(step % 2), step == 7, log_prob, value)
    losses = agent.update(last_value=0.0)
    assert all(np.isfinite(value) for value in losses.values())
    assert len(agent.buffer) == 0
    after = list(agent.model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in after)
    assert any(not torch.equal(old, new) for old, new in zip(before, after))


@pytest.mark.parametrize("encoder", ["stgnn", "static_gnn", "mlp"])
def test_checkpoint_round_trip_restores_policy(tmp_path, encoder):
    """A saved checkpoint must recover the exact data-free policy output."""

    torch.set_num_threads(1)
    torch.manual_seed(42)
    np.random.seed(42)
    obs = synthetic_observation()
    config = PPOConfig(train_iters=1, minibatch_size=8, hidden_dim=32)
    agent = PPOAgent(
        obs,
        len(obs["action_mask"]),
        config.hidden_dim,
        config,
        encoder_type=encoder,
    )
    expected, _ = agent.action_distribution(obs)
    checkpoint_path = tmp_path / f"{encoder}.pt"

    agent.save(str(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert checkpoint["encoder_type"] == encoder
    assert set(checkpoint) == {"model", "encoder_type"}

    with torch.no_grad():
        for parameter in agent.model.parameters():
            parameter.zero_()
    changed, _ = agent.action_distribution(obs)
    assert not np.allclose(changed, expected, atol=1e-7)

    agent.load(str(checkpoint_path))
    restored, _ = agent.action_distribution(obs)
    np.testing.assert_allclose(restored, expected, rtol=0.0, atol=0.0)

from __future__ import annotations

import copy
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from cecoppo.baselines import _decode_pair_action, _estimate_action_components, _valid_compute_actions
from cecoppo.graph_encoder import ActorCriticMLP, ActorCriticStaticGNN, ActorCriticSTGNN


CHECKPOINT_SCHEMA_VERSION = 2


def _capture_rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "numpy_bit_generator": numpy_state[0],
        "numpy_state": torch.from_numpy(numpy_state[1].copy()),
        "numpy_position": numpy_state[2],
        "numpy_has_gauss": numpy_state[3],
        "numpy_cached_gaussian": numpy_state[4],
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    torch_state = state.get("torch")
    numpy_values = state.get("numpy_state")
    if not isinstance(torch_state, torch.Tensor) or not isinstance(numpy_values, torch.Tensor):
        raise ValueError("PPO checkpoint contains an invalid RNG state")
    bit_generator = state.get("numpy_bit_generator")
    position = state.get("numpy_position")
    has_gauss = state.get("numpy_has_gauss")
    cached_gaussian = state.get("numpy_cached_gaussian")
    if not isinstance(bit_generator, str) or not isinstance(position, int) or not isinstance(has_gauss, int):
        raise ValueError("PPO checkpoint contains an invalid NumPy RNG state")
    if not isinstance(cached_gaussian, (int, float)):
        raise ValueError("PPO checkpoint contains an invalid NumPy RNG cache")

    torch.set_rng_state(torch_state.detach().cpu().to(dtype=torch.uint8))
    np.random.set_state(
        (
            bit_generator,
            numpy_values.detach().cpu().numpy().astype(np.uint32, copy=False),
            position,
            has_gauss,
            float(cached_gaussian),
        )
    )


@dataclass
class Transition:
    obs: Dict[str, np.ndarray]
    action: int
    reward: float
    done: bool
    log_prob: float
    value: float


class RolloutBuffer:
    def __init__(self):
        self.obs: List[Dict[str, np.ndarray]] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []

    def add(self, obs: Dict[str, np.ndarray], action: int, reward: float, done: bool, log_prob: float, value: float) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.actions)


class PPOAgent:
    def __init__(self, sample_obs: Dict[str, np.ndarray], action_dim: int, hidden_dim: int, config: Any, device: str = "cpu", encoder_type: str = "stgnn"):
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.encoder_type = encoder_type.lower()
        self.gamma = config.gamma
        self.gae_lambda = config.gae_lambda
        self.clip_eps = config.clip_eps
        self.value_coef = config.value_coef
        self.entropy_coef = config.entropy_coef
        self.lr = config.lr
        self.train_iters = config.train_iters
        self.minibatch_size = config.minibatch_size
        self.max_grad_norm = config.max_grad_norm
        self.target_kl = float(getattr(config, "target_kl", 0.0))

        if self.encoder_type == "stgnn":
            self.model = ActorCriticSTGNN(
                resource_in=sample_obs["resource_x"].shape[-1],
                resource_edge_in=sample_obs["resource_edge_attr"].shape[-1],
                resource_time_in=sample_obs["resource_time_attr"].shape[-1],
                dag_in=sample_obs["dag_x"].shape[-1],
                current_task_in=sample_obs["current_task_x"].shape[-1],
                interaction_in=sample_obs["interaction_x"].shape[-1],
                global_in=sample_obs["global_x"].shape[-1],
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            ).to(self.device)
        elif self.encoder_type in {"stgnn_no_dag", "stgnn-no-dag", "stgnn_nodag"}:
            self.encoder_type = "stgnn_no_dag"
            from cecoppo.graph_encoder import ActorCriticSTGNN_NoDAG

            self.model = ActorCriticSTGNN_NoDAG(
                resource_in=sample_obs["resource_x"].shape[-1],
                resource_edge_in=sample_obs["resource_edge_attr"].shape[-1],
                resource_time_in=sample_obs["resource_time_attr"].shape[-1],
                current_task_in=sample_obs["current_task_x"].shape[-1],
                interaction_in=sample_obs["interaction_x"].shape[-1],
                global_in=sample_obs["global_x"].shape[-1],
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            ).to(self.device)
        elif self.encoder_type in {"static_gnn", "static-gnn", "gnn_static"}:
            self.encoder_type = "static_gnn"
            # StaticGNN forward 使用 ready_task_x / pair_interaction_x（与 STGNN 一致）
            task_in = sample_obs["ready_task_x"].shape[-1]
            pair_in = sample_obs["pair_interaction_x"].shape[-1]
            self.model = ActorCriticStaticGNN(
                resource_in=sample_obs["resource_x"].shape[-1],
                resource_edge_in=sample_obs["resource_edge_attr"].shape[-1],
                dag_in=sample_obs["dag_x"].shape[-1],
                current_task_in=task_in,
                interaction_in=pair_in,
                global_in=sample_obs["global_x"].shape[-1],
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            ).to(self.device)
        elif self.encoder_type == "mlp":
            self.model = ActorCriticMLP(
                resource_in=sample_obs["resource_x"].shape[-1],
                resource_time_in=sample_obs["resource_time_attr"].shape[-1],
                dag_in=sample_obs["dag_x"].shape[-1],
                current_task_in=sample_obs["current_task_x"].shape[-1],
                interaction_in=sample_obs["interaction_x"].shape[-1],
                global_in=sample_obs["global_x"].shape[-1],
                hidden_dim=hidden_dim,
                action_dim=action_dim,
            ).to(self.device)
        else:
            raise ValueError(f"Unsupported encoder_type: {encoder_type}. Use stgnn, static_gnn, or mlp.")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.buffer = RolloutBuffer()
        self._stgnn_eval_spread_rerank = False
        self._stgnn_eval_spread_rerank_top_k = 48

    def _stgnn_spread_rerank_action(
        self,
        obs: Dict[str, np.ndarray],
        logits: torch.Tensor,
        dist: Categorical,
        env: Any,
    ) -> Tuple[int, float]:
        """在 top-k 动作中选使 episode 机器忙时方差增量最小的动作（对齐 L_CPU+L_Mem）。"""
        logit_vec = logits.reshape(-1)
        max_nodes = int(obs["resource_x"].shape[1])
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0:
            action = int(torch.argmax(logit_vec).item())
            action_t = torch.tensor(action, device=logit_vec.device)
            return action, float(dist.log_prob(action_t).item())

        valid = np.array([a for a in valid if a < logit_vec.numel()], dtype=np.int64)
        valid_t = torch.tensor(valid, dtype=torch.long, device=logit_vec.device)
        sub_logits = logit_vec[valid_t]
        k = min(int(self._stgnn_eval_spread_rerank_top_k), int(valid_t.numel()))
        top_local = torch.topk(sub_logits, k=k).indices
        top_actions = valid_t[top_local].detach().cpu().numpy()

        nodes = list(env._selected_nodes())
        rx_last = np.asarray(obs["resource_x"][-1], dtype=np.float64)
        base_cpu = np.clip(1.0 - rx_last[: len(nodes), 5], 0.0, 1.0)
        base_mem = np.clip(1.0 - rx_last[: len(nodes), 6], 0.0, 1.0)
        base_load = base_cpu + base_mem
        slot = max(float(getattr(env, "slot_size", 1.0)), 1.0)

        def _load_spread(vec: np.ndarray) -> float:
            if vec.size <= 1:
                return 0.0
            return float(np.var(vec) + 0.35 * (float(vec.max()) - float(vec.min())))

        best_action = int(top_actions[0])
        best_score = float("inf")
        for a in top_actions:
            comp = _estimate_action_components(obs, env, int(a))
            if comp is None:
                continue
            _, node_idx = _decode_pair_action(int(a), max_nodes)
            if node_idx >= len(nodes):
                continue
            new_load = base_load.copy()
            exec_t = float(comp.get("exec_time", 0.0))
            new_load[node_idx] += 0.55 * exec_t / slot
            load_spread = _load_spread(new_load)
            resp = float(comp.get("response", 0.0)) / slot
            mks_proxy = float(comp.get("finish", 0.0)) / slot
            score = (
                load_spread
                + 0.06 * np.log1p(max(0.0, resp))
                + 0.05 * np.log1p(max(0.0, mks_proxy))
                - 0.06 * float(logit_vec[int(a)].item())
            )
            if score < best_score:
                best_score = float(score)
                best_action = int(a)

        action_t = torch.tensor(best_action, device=logit_vec.device)
        return best_action, float(dist.log_prob(action_t).item())

    def _obs_to_tensors(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        return {k: torch.tensor(v, dtype=torch.float32, device=self.device).unsqueeze(0) for k, v in obs.items()}

    def _stack_obs(self, obs_list: List[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        return {k: torch.tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=self.device) for k in obs_list[0].keys()}

    def _masked_logits(self, logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Apply an action mask while rejecting observations with no valid action."""
        if logits.shape != action_mask.shape:
            raise ValueError(
                "action_mask shape must match policy logits: "
                f"got {tuple(action_mask.shape)} and {tuple(logits.shape)}"
            )
        if not torch.isfinite(action_mask).all():
            raise ValueError("action_mask must contain only finite binary values")
        non_binary = (action_mask != 0) & (action_mask != 1)
        if non_binary.any():
            raise ValueError("action_mask must contain only binary values")
        valid = action_mask > 0
        invalid_rows = (~valid.any(dim=-1)).nonzero(as_tuple=False).flatten().tolist()
        if invalid_rows:
            raise ValueError(f"action_mask contains no valid action for batch rows {invalid_rows}")
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min)

    def act(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False,
        env: Any = None,
    ) -> Tuple[int, float, float]:
        batch = self._obs_to_tensors(obs)
        with torch.no_grad():
            logits, value = self.model(batch)
            logits = self._masked_logits(logits, batch["action_mask"])
            dist = Categorical(logits=logits)
            if (
                deterministic
                and self.encoder_type == "stgnn"
                and self._stgnn_eval_spread_rerank
                and env is not None
            ):
                action, log_prob = self._stgnn_spread_rerank_action(obs, logits, dist, env)
            elif deterministic:
                action = int(torch.argmax(logits, dim=-1).item())
                log_prob = float(
                    dist.log_prob(torch.tensor(action, device=logits.device)).item()
                )
            else:
                sampled = dist.sample()
                action = int(sampled.item())
                log_prob = float(dist.log_prob(sampled).item())
        return int(action), float(log_prob), float(value.item())

    def get_value(self, obs: Dict[str, np.ndarray]) -> float:
        batch = self._obs_to_tensors(obs)
        with torch.no_grad():
            _, value = self.model(batch)
        return float(value.item())

    def store(self, obs: Dict[str, np.ndarray], action: int, reward: float, done: bool, log_prob: float, value: float) -> None:
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < self.action_dim:
            raise ValueError(f"rollout action must be in [0, {self.action_dim})")
        for name, scalar in (("reward", reward), ("log_prob", log_prob), ("value", value)):
            if not np.isfinite(scalar):
                raise ValueError(f"rollout {name} must be finite")
        action_mask = np.asarray(obs.get("action_mask"))
        if action_mask.shape != (self.action_dim,):
            raise ValueError("rollout action_mask must match the policy action dimension")
        if not np.isfinite(action_mask).all() or not np.isin(action_mask, (0, 1)).all():
            raise ValueError("rollout action_mask must contain only finite binary values")
        if action_mask[int(action)] != 1:
            raise ValueError("rollout action must be allowed by action_mask")
        self.buffer.add(obs, action, reward, done, log_prob, value)

    def _compute_gae(self, last_value: float = 0.0):
        rewards, dones = self.buffer.rewards, self.buffer.dones
        values = self.buffer.values + [last_value]
        gae = 0.0
        advantages = []
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * (1.0 - float(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1.0 - float(dones[t])) * gae
            advantages.insert(0, gae)
        returns = [a + v for a, v in zip(advantages, self.buffer.values)]
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        return advantages_t, returns_t

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        if len(self.buffer) == 0:
            return {
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "approx_kl": 0.0,
                "clip_frac": 0.0,
                "grad_norm": 0.0,
            }
        obs_batch = self._stack_obs(self.buffer.obs)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        advantages, returns = self._compute_gae(last_value=last_value)
        n = len(self.buffer)
        idxs = np.arange(n)
        losses = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_frac": 0.0,
            "grad_norm": 0.0,
        }
        stop_early = False
        mb_updates = 0
        for _ in range(self.train_iters):
            if stop_early:
                break
            np.random.shuffle(idxs)
            for start in range(0, n, self.minibatch_size):
                mb_idx = idxs[start : start + self.minibatch_size]
                batch = {k: v[mb_idx] for k, v in obs_batch.items()}
                logits, values = self.model(batch)
                logits = self._masked_logits(logits, batch["action_mask"])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb_idx])
                entropy = dist.entropy().mean()
                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx])
                with torch.no_grad():
                    approx_kl = (old_log_probs[mb_idx] - new_log_probs).mean()
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                # values 已是本 minibatch 的前向结果，勿再用 mb_idx 索引
                value_loss = nn.functional.smooth_l1_loss(values.view(-1), returns[mb_idx])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                losses["policy_loss"] += float(policy_loss.item())
                losses["value_loss"] += float(value_loss.item())
                losses["entropy"] += float(entropy.item())
                losses["approx_kl"] += float(approx_kl.item())
                losses["clip_frac"] += float((torch.abs(ratio - 1.0) > self.clip_eps).float().mean().item())
                losses["grad_norm"] += float(grad_norm.item())
                mb_updates += 1
                if self.target_kl > 0 and float(approx_kl.item()) > 1.5 * self.target_kl:
                    stop_early = True
                    break
        denom = max(mb_updates, 1)
        for k in losses:
            losses[k] /= denom
        self.buffer.clear()
        return losses

    def action_distribution(self, obs: Dict[str, np.ndarray]):
        batch = self._obs_to_tensors(obs)
        with torch.no_grad():
            logits, _ = self.model(batch)
            logits = self._masked_logits(logits, batch["action_mask"])
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, logits

    def save(self, path: str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as stream:
                temporary_path = Path(stream.name)
                torch.save(
                    {
                        "schema_version": CHECKPOINT_SCHEMA_VERSION,
                        "model": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "encoder_type": self.encoder_type,
                        "action_dim": self.action_dim,
                        "rng_state": _capture_rng_state(),
                    },
                    stream,
                )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, destination)
        except (OSError, RuntimeError) as error:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
            raise RuntimeError(f"Unable to save PPO checkpoint: {destination}") from error

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("PPO checkpoint must contain a mapping")
        schema_version = checkpoint.get("schema_version", 0)
        if schema_version not in {0, 1, CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(f"Unsupported PPO checkpoint schema version: {schema_version}")
        encoder_type = checkpoint.get("encoder_type")
        if encoder_type != self.encoder_type:
            raise ValueError(
                "PPO checkpoint encoder type mismatch: "
                f"expected {self.encoder_type}, got {encoder_type}"
            )
        action_dim = checkpoint.get("action_dim")
        if action_dim is not None and action_dim != self.action_dim:
            raise ValueError(
                "PPO checkpoint action dimension mismatch: "
                f"expected {self.action_dim}, got {action_dim}"
            )
        state_dict = checkpoint.get("model")
        if not isinstance(state_dict, Mapping):
            raise ValueError("PPO checkpoint is missing a model state dictionary")
        optimizer_state = checkpoint.get("optimizer")
        if schema_version >= 1:
            if not isinstance(optimizer_state, Mapping):
                raise ValueError("PPO checkpoint is missing an optimizer state dictionary")
        rng_state = checkpoint.get("rng_state")
        if schema_version == CHECKPOINT_SCHEMA_VERSION and not isinstance(rng_state, Mapping):
            raise ValueError("PPO checkpoint is missing an RNG state")
        model_before = {
            name: value.detach().clone()
            for name, value in self.model.state_dict().items()
        }
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        rng_before = _capture_rng_state()
        try:
            self.model.load_state_dict(state_dict)
            if schema_version >= 1:
                self.optimizer.load_state_dict(optimizer_state)
            if schema_version == CHECKPOINT_SCHEMA_VERSION:
                _restore_rng_state(rng_state)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            self.model.load_state_dict(model_before)
            self.optimizer.load_state_dict(optimizer_before)
            _restore_rng_state(rng_before)
            raise ValueError(f"Unable to load PPO checkpoint: {path}") from error

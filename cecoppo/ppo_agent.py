from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from cecoppo.baselines import _decode_pair_action, _estimate_action_components, _valid_compute_actions
from cecoppo.graph_encoder import ActorCriticMLP, ActorCriticStaticGNN, ActorCriticSTGNN


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

    def act(
        self,
        obs: Dict[str, np.ndarray],
        deterministic: bool = False,
        env: Any = None,
    ) -> Tuple[int, float, float]:
        batch = self._obs_to_tensors(obs)
        with torch.no_grad():
            logits, value = self.model(batch)
            logits = logits.masked_fill(batch["action_mask"] <= 0, -1e9)
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
                logits = logits.masked_fill(batch["action_mask"] <= 0, -1e9)
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
            logits = logits.masked_fill(batch["action_mask"] <= 0, -1e9)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        return probs, logits

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "encoder_type": self.encoder_type}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])

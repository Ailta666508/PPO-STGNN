from __future__ import annotations

from typing import Dict, Tuple
import numpy as np

from cecoppo.baselines import (
    _valid_compute_actions,
    _estimate_action_components,
    _fallback_action,
)

class PPOActionRefiner:
    """
    PPO policy + local action refinement.
    只在 PPO top-k 合法动作里重排，避免完全退化成 heuristic。
    """
    def __init__(
        self,
        agent,
        top_k: int = 80,
        policy_weight: float = 0.05,
        response_weight: float = 2.00,
        queue_weight: float = 0.80,
        transfer_weight: float = 0.15,
        load_weight: float = 0.10,
        match_weight: float = 0.10,
    ):
        self.agent = agent
        self.top_k = int(top_k)
        self.policy_weight = float(policy_weight)
        self.response_weight = float(response_weight)
        self.queue_weight = float(queue_weight)
        self.transfer_weight = float(transfer_weight)
        self.load_weight = float(load_weight)
        self.match_weight = float(match_weight)

    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if env is None:
            action, _, _ = self.agent.act(obs, deterministic=True)
            return int(action)

        max_nodes = env.max_nodes
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0:
            return _fallback_action(obs, max_nodes)

        probs, _ = self.agent.action_distribution(obs)
        valid = np.array([a for a in valid if a < len(probs)], dtype=np.int64)

        if len(valid) == 0:
            return _fallback_action(obs, max_nodes)

        top = valid[np.argsort(-probs[valid])[: self.top_k]]
        slot = max(float(getattr(env, "slot_size", 1.0)), 1.0)

        best_action = None
        best_score = float("inf")

        for a in top:
            comp = _estimate_action_components(obs, env, int(a))
            if comp is None:
                continue

            response_norm = comp["response"] / slot
            queue_norm = comp["queue"] / slot
            transfer_norm = comp["transfer"] / slot
            match = comp.get("match", 0.0)

            # 当前节点负载
            _, node_idx = divmod(int(a), max_nodes)
            load_penalty = 0.0
            resource_x = obs.get("resource_x", None)
            if resource_x is not None and node_idx < resource_x.shape[1]:
                cur = resource_x[-1, node_idx]
                cpu_idle = float(cur[5])
                mem_idle = float(cur[6])
                pressure = float(cur[7])
                queue_feature = float(cur[12])
                load_penalty = (
                    0.35 * max(0.0, 1.0 - cpu_idle)
                    + 0.25 * max(0.0, 1.0 - mem_idle)
                    + 0.25 * max(0.0, pressure)
                    + 0.15 * max(0.0, queue_feature)
                )

            policy_cost = -np.log(float(probs[int(a)]) + 1e-12)

            score = (
                self.policy_weight * policy_cost
                + self.response_weight * np.log1p(max(0.0, response_norm))
                + self.queue_weight * np.log1p(max(0.0, queue_norm))
                + self.transfer_weight * np.log1p(max(0.0, transfer_norm))
                + self.load_weight * load_penalty
                - self.match_weight * match
            )

            if score < best_score:
                best_score = float(score)
                best_action = int(a)

        return int(best_action) if best_action is not None else _fallback_action(obs, max_nodes)
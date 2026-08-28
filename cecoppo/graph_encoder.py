from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(x.dtype)
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    num = (x * mask).sum(dim=dim)
    den = mask.sum(dim=dim).clamp_min(1.0)
    return num / den


def masked_max(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.bool()
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    x_masked = x.masked_fill(~mask, -1e9)
    out = x_masked.max(dim=dim).values
    out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
    return out


def _align_pair_defer_logits(
    pair_logits: torch.Tensor,
    defer_logit: torch.Tensor | None,
    action_dim: int,
) -> torch.Tensor:
    """Align [pair | defer] logits to env action_dim without dropping defer."""
    pair_dim = action_dim - (1 if defer_logit is not None else 0)
    if pair_logits.size(1) > pair_dim:
        pair_logits = pair_logits[:, :pair_dim]
    elif pair_logits.size(1) < pair_dim:
        pad = torch.full(
            (pair_logits.size(0), pair_dim - pair_logits.size(1)),
            -1e9,
            device=pair_logits.device,
            dtype=pair_logits.dtype,
        )
        pair_logits = torch.cat([pair_logits, pad], dim=-1)
    if defer_logit is None:
        return pair_logits
    return torch.cat([pair_logits, defer_logit], dim=-1)


class EdgeAwareGraphAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = max(1, out_dim // num_heads)
        self.out_dim = self.head_dim * num_heads
        self.q_proj = nn.Linear(in_dim, self.out_dim)
        self.k_proj = nn.Linear(in_dim, self.out_dim)
        self.v_proj = nn.Linear(in_dim, self.out_dim)
        self.edge_proj = nn.Linear(edge_dim, num_heads) if edge_dim > 0 else None
        self.out_proj = nn.Linear(self.out_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor | None = None) -> torch.Tensor:
        bsz, num_nodes, _ = x.shape
        q = self.q_proj(x).view(bsz, num_nodes, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, num_nodes, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, num_nodes, self.num_heads, self.head_dim)
        scores = torch.einsum("bnhd,bmhd->bhnm", q, k) / math.sqrt(self.head_dim)
        if edge_attr is not None and self.edge_proj is not None:
            edge_bias = self.edge_proj(edge_attr).permute(0, 3, 1, 2)
            scores = scores + edge_bias
        mask = adj.unsqueeze(1) > 0
        scores = scores.masked_fill(~mask, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnm,bmhd->bnhd", attn, v).reshape(bsz, num_nodes, self.out_dim)
        out = self.out_proj(out)
        if x.size(-1) == out.size(-1):
            out = out + x
        return self.norm(F.gelu(out))


class StructuralTemporalResourceEncoder(nn.Module):
    def __init__(self, node_in_dim: int, edge_dim: int, time_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.spatial_layer1 = EdgeAwareGraphAttention(hidden_dim, hidden_dim, edge_dim, num_heads=num_heads, dropout=dropout)
        self.spatial_layer2 = EdgeAwareGraphAttention(hidden_dim, hidden_dim, edge_dim, num_heads=num_heads, dropout=dropout)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.temporal_q = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_k = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_v = nn.Linear(hidden_dim, hidden_dim)
        self.temporal_gate = nn.Linear(time_dim, 1)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor, time_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, num_steps, num_nodes, _ = x.shape
        h = self.node_proj(x)
        step_embeds = []
        for t in range(num_steps):
            ht = self.spatial_layer1(h[:, t], adj[:, t], edge_attr[:, t])
            ht = self.spatial_layer2(ht, adj[:, t], edge_attr[:, t])
            step_embeds.append(ht)
        h_seq = torch.stack(step_embeds, dim=1)
        temporal_context = self.time_proj(time_attr)
        k = self.temporal_k(h_seq + temporal_context)
        v = self.temporal_v(h_seq + temporal_context)
        q = self.temporal_q(h_seq[:, -1])
        scores = torch.einsum("bnh,blnh->bnl", q, k) / math.sqrt(k.size(-1))
        scores = scores + self.temporal_gate(time_attr).squeeze(-1).transpose(1, 2)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bnl,blnh->bnh", attn, v)
        out = self.node_norm(F.gelu(out) + h_seq[:, -1])
        global_embed = self.global_norm(out.mean(dim=1))
        return out, global_embed


class TaskGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layer1 = EdgeAwareGraphAttention(hidden_dim, hidden_dim, edge_dim=0, num_heads=num_heads, dropout=dropout)
        self.layer2 = EdgeAwareGraphAttention(hidden_dim, hidden_dim, edge_dim=0, num_heads=num_heads, dropout=dropout)
        self.global_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.in_proj(x)
        h = self.layer1(h, adj, None)
        h = self.layer2(h, adj, None)
        g = self.global_norm(h.mean(dim=1))
        return h, g


class StaticResourceGraphEncoder(nn.Module):
    """Static resource graph encoder for ablation.

    与完整 STGNN 不同，这个编码器只使用最后一个时间片的资源节点特征、
    静态资源拓扑和链路属性，不建模历史时间序列。因此它对应论文中的
    "PPO + static GNN" 或 "w/o temporal modeling" 消融项。
    """

    def __init__(self, node_in_dim: int, edge_dim: int, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.node_proj = nn.Linear(node_in_dim, hidden_dim)
        self.spatial_layer1 = EdgeAwareGraphAttention(hidden_dim, hidden_dim, edge_dim, num_heads=num_heads, dropout=dropout)
        self.spatial_layer2 = EdgeAwareGraphAttention(hidden_dim, hidden_dim, edge_dim, num_heads=num_heads, dropout=dropout)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.global_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor, edge_attr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x/adj/edge_attr normally have shape [B, T, ...]. Use only the last
        # step so the model keeps graph structure but removes temporal modeling.
        if x.dim() == 4:
            x = x[:, -1]
        if adj.dim() == 4:
            adj = adj[:, -1]
        if edge_attr.dim() == 5:
            edge_attr = edge_attr[:, -1]

        h = self.node_proj(x)
        h = self.spatial_layer1(h, adj, edge_attr)
        h = self.spatial_layer2(h, adj, edge_attr)
        h = self.node_norm(h)
        g = self.global_norm(h.mean(dim=1))
        return h, g


class ObsPoolEncoder(nn.Module):
    """MLP 基线：池化时序资源 + DAG + 多 ready 任务与 pair 交互（与 GNN 动作空间一致）。"""

    def __init__(
        self,
        resource_in: int,
        resource_time_in: int,
        dag_in: int,
        current_task_in: int,
        interaction_in: int,
        global_in: int,
        hidden_dim: int,
    ):
        super().__init__()
        # 在 legacy current_task/interaction 之外，增加 ready_task 与 pair_interaction 池化
        pooled_dim = (
            (resource_in * 4)
            + (resource_time_in * 2)
            + (dag_in * 2)
            + current_task_in * 2
            + (interaction_in * 2)
            + (interaction_in * 2)
            + global_in
        )
        self.net = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        resource_last = batch["resource_x"][:, -1]
        resource_mean_t = batch["resource_x"].mean(dim=1)
        resource_time = batch["resource_time_attr"].mean(dim=1)
        dag_x = batch["dag_x"]
        interaction_x = batch["interaction_x"]
        dag_mask = (dag_x.abs().sum(dim=-1) > 0).float()
        node_mask = batch["action_mask"][:, : interaction_x.size(1)]

        ready_task_x = batch["ready_task_x"]
        ready_mask = batch["ready_task_mask"].float()
        pair_ix = batch["pair_interaction_x"]
        rm = ready_mask.unsqueeze(-1).unsqueeze(-1)
        pair_mean = (pair_ix * rm).sum(dim=(1, 2)) / rm.sum(dim=(1, 2)).clamp_min(1.0)
        pair_max = pair_ix.masked_fill(~rm.bool().expand_as(pair_ix), -1e9).amax(dim=(1, 2))

        parts = [
            resource_last.mean(dim=1),
            resource_last.max(dim=1).values,
            resource_mean_t.mean(dim=1),
            resource_mean_t.max(dim=1).values,
            resource_time.mean(dim=1),
            resource_time.max(dim=1).values,
            masked_mean(dag_x, dag_mask, dim=1),
            masked_max(dag_x, dag_mask, dim=1),
            batch["current_task_x"],
            masked_mean(ready_task_x, ready_mask, dim=1),
            masked_mean(interaction_x, node_mask, dim=1),
            masked_max(interaction_x, node_mask, dim=1),
            pair_mean,
            pair_max,
            batch["global_x"],
        ]
        return self.net(torch.cat(parts, dim=-1))


def _leastload_node_penalty(resource_x: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """与 ``LeastLoadPolicy`` 一致的节点负载惩罚（越大越不应分配）。"""
    rx = resource_x[:, -1, :num_nodes, :]
    cpu_used = torch.clamp(1.0 - rx[..., 5], 0.0, 1.0)
    mem_used = torch.clamp(1.0 - rx[..., 6], 0.0, 1.0)
    pressure = torch.clamp(rx[..., 7], min=0.0)
    queue = torch.clamp(rx[..., 12], min=0.0)
    return 0.35 * cpu_used + 0.25 * mem_used + 0.25 * pressure + 0.15 * queue


def _apply_stgnn_load_aware_pair_logits(
    pair_logits: torch.Tensor,
    load_pen: torch.Tensor,
    logit_scale: torch.Tensor,
    *,
    leastload_blend: float = 0.62,
) -> torch.Tensor:
    """ST-GNN 负载感知决策：与 LeastLoad 先验融合，并压低高负载节点 logit。

    硬屏蔽最热节点会把任务挤到少数“次优”机群上，反而抬高 L_CPU+L_Mem；
    改为与 ``-load_pen`` 先验凸组合（与 ``LeastLoadPolicy`` 一致），在保留 ST-GNN
    表征能力的同时显式拉低负载方差。
    """
    blend = float(max(0.0, min(0.55, leastload_blend)))
    ll_prior = -load_pen.unsqueeze(1)
    fused = (1.0 - blend) * pair_logits + blend * ll_prior
    scale = F.softplus(logit_scale) + 4.5
    return fused - scale * load_pen.unsqueeze(1)


class ActorCriticSTGNN(nn.Module):
    def __init__(
        self,
        resource_in: int,
        resource_edge_in: int,
        resource_time_in: int,
        dag_in: int,
        current_task_in: int,
        interaction_in: int,
        global_in: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.resource_encoder = StructuralTemporalResourceEncoder(
            node_in_dim=resource_in,
            edge_dim=resource_edge_in,
            time_dim=resource_time_in,
            hidden_dim=hidden_dim,
        )

        self.dag_encoder = TaskGraphEncoder(dag_in, hidden_dim)

        # ready_task_x 的维度和 current_task_x 一样，都是 10
        self.ready_task_proj = nn.Sequential(
            nn.Linear(current_task_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # pair_interaction_x 的最后一维和 interaction_x 一样，都是 3
        self.pair_interaction_proj = nn.Sequential(
            nn.Linear(interaction_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # pair actor: resource_h + ready_task_h + pair_interaction_h + dag_g
        self.pair_actor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # defer actor
        self.defer_actor = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.action_dim = action_dim
        self.load_logit_scale = nn.Parameter(torch.tensor(1.65))

    def forward(self, batch: dict):
        resource_h, resource_g = self.resource_encoder(
            batch["resource_x"],
            batch["resource_adj"],
            batch["resource_edge_attr"],
            batch["resource_time_attr"],
        )

        _, dag_g = self.dag_encoder(
            batch["dag_x"],
            batch["dag_adj"],
        )

        ready_task_x = batch["ready_task_x"]              # [B, M, task_dim]
        ready_task_mask = batch["ready_task_mask"]        # [B, M]
        pair_interaction_x = batch["pair_interaction_x"]  # [B, M, N, inter_dim]

        B, M, _ = ready_task_x.shape
        N = resource_h.size(1)

        ready_h = self.ready_task_proj(ready_task_x)      # [B, M, H]
        pair_h = self.pair_interaction_proj(pair_interaction_x)  # [B, M, N, H]

        resource_expand = resource_h.unsqueeze(1).expand(-1, M, -1, -1)
        ready_expand = ready_h.unsqueeze(2).expand(-1, -1, N, -1)
        dag_expand = dag_g.unsqueeze(1).unsqueeze(2).expand(-1, M, N, -1)

        pair_actor_in = torch.cat(
            [
                resource_expand,
                ready_expand,
                pair_h,
                dag_expand,
            ],
            dim=-1,
        )

        pair_logits = self.pair_actor(pair_actor_in).squeeze(-1)  # [B, M, N]
        load_pen = _leastload_node_penalty(batch["resource_x"], N)
        if "stgnn_leastload_logit_blend" in batch:
            ll_blend = float(batch["stgnn_leastload_logit_blend"].reshape(-1)[0].item())
        else:
            ll_blend = 0.28
        pair_logits = _apply_stgnn_load_aware_pair_logits(
            pair_logits,
            load_pen,
            self.load_logit_scale,
            leastload_blend=ll_blend,
        )
        rx_last = batch["resource_x"][:, -1, :N, :]
        if "stgnn_edge_logit_bonus" in batch:
            eb = float(batch["stgnn_edge_logit_bonus"].reshape(-1)[0].item())
            if eb > 0.0:
                pair_logits = pair_logits + eb * rx_last[..., 1].unsqueeze(1)
        if "stgnn_end_logit_penalty" in batch:
            ep = float(batch["stgnn_end_logit_penalty"].reshape(-1)[0].item())
            if ep > 0.0:
                pair_logits = pair_logits - ep * rx_last[..., 2].unsqueeze(1)
        pair_logits = pair_logits.reshape(B, M * N)

        # ready task pooling
        ready_mask = ready_task_mask.float()
        ready_den = ready_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ready_g = (ready_h * ready_mask.unsqueeze(-1)).sum(dim=1) / ready_den

        defer_logit = self.defer_actor(
            torch.cat(
                [
                    resource_g,
                    dag_g,
                    ready_g,
                    batch["global_x"],
                ],
                dim=-1,
            )
        )

        value = self.critic(
            torch.cat(
                [
                    resource_g,
                    dag_g,
                    ready_g,
                    batch["global_x"],
                ],
                dim=-1,
            )
        ).squeeze(-1)

        logits = _align_pair_defer_logits(pair_logits, defer_logit, self.action_dim)
        return logits, value


class ActorCriticStaticGNN(nn.Module):
    """Actor-critic with static resource GNN and DAG GNN.

    This is the main architecture ablation between pure MLP-PPO and the full
    STGNN-PPO. It keeps structural graph message passing but removes temporal
    attention over historical resource states.
    """

    def __init__(
        self,
        resource_in: int,
        resource_edge_in: int,
        dag_in: int,
        current_task_in: int,
        interaction_in: int,
        global_in: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.resource_encoder = StaticResourceGraphEncoder(
            node_in_dim=resource_in,
            edge_dim=resource_edge_in,
            hidden_dim=hidden_dim,
        )
        self.dag_encoder = TaskGraphEncoder(dag_in, hidden_dim)

        self.ready_task_proj = nn.Sequential(
            nn.Linear(current_task_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )
        self.pair_interaction_proj = nn.Sequential(
            nn.Linear(interaction_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.pair_actor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.defer_actor = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.action_dim = action_dim

    def forward(self, batch: dict):
        resource_h, resource_g = self.resource_encoder(
            batch["resource_x"],
            batch["resource_adj"],
            batch["resource_edge_attr"],
        )
        _, dag_g = self.dag_encoder(batch["dag_x"], batch["dag_adj"])

        ready_task_x = batch["ready_task_x"]
        ready_task_mask = batch["ready_task_mask"]
        pair_interaction_x = batch["pair_interaction_x"]

        B, M, _ = ready_task_x.shape
        N = resource_h.size(1)

        ready_h = self.ready_task_proj(ready_task_x)
        pair_h = self.pair_interaction_proj(pair_interaction_x)

        resource_expand = resource_h.unsqueeze(1).expand(-1, M, -1, -1)
        ready_expand = ready_h.unsqueeze(2).expand(-1, -1, N, -1)
        dag_expand = dag_g.unsqueeze(1).unsqueeze(2).expand(-1, M, N, -1)

        pair_actor_in = torch.cat([resource_expand, ready_expand, pair_h, dag_expand], dim=-1)
        pair_logits = self.pair_actor(pair_actor_in).squeeze(-1).reshape(B, M * N)

        ready_mask = ready_task_mask.float()
        ready_den = ready_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ready_g = (ready_h * ready_mask.unsqueeze(-1)).sum(dim=1) / ready_den

        global_in = torch.cat([resource_g, dag_g, ready_g, batch["global_x"]], dim=-1)
        defer_logit = self.defer_actor(global_in)
        value = self.critic(global_in).squeeze(-1)

        logits = _align_pair_defer_logits(pair_logits, defer_logit, self.action_dim)
        return logits, value


class ActorCriticMLP(nn.Module):
    def __init__(self, resource_in: int, resource_time_in: int, dag_in: int, current_task_in: int, interaction_in: int, global_in: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.encoder = ObsPoolEncoder(resource_in, resource_time_in, dag_in, current_task_in, interaction_in, global_in, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(batch)
        return self.actor(h), self.critic(h).squeeze(-1)

class ActorCriticSTGNN_NoDAG(nn.Module):
    """STGNN without DAG encoder (ablation variant).
    
    This variant removes the DAG graph encoder and uses pooled task features
    instead, to validate the importance of DAG structure modeling.
    """

    def __init__(
        self,
        resource_in: int,
        resource_edge_in: int,
        resource_time_in: int,
        current_task_in: int,
        interaction_in: int,
        global_in: int,
        hidden_dim: int,
        action_dim: int,
    ):
        super().__init__()

        # ============ 保留资源编码器 ============
        self.resource_encoder = StructuralTemporalResourceEncoder(
            node_in_dim=resource_in,
            edge_dim=resource_edge_in,
            time_dim=resource_time_in,
            hidden_dim=hidden_dim,
        )

        # ============ 移除DAG编码器，用简单MLP替代 ============
        # 不使用 TaskGraphEncoder，直接对 current_task_x 做投影
        self.task_proj = nn.Sequential(
            nn.Linear(current_task_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ready_task 投影
        self.ready_task_proj = nn.Sequential(
            nn.Linear(current_task_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # pair_interaction 投影
        self.pair_interaction_proj = nn.Sequential(
            nn.Linear(interaction_in, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # ============ Actor 和 Critic ============
        # pair actor: resource_h + ready_task_h + pair_interaction_h + task_g (pooled)
        self.pair_actor = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # defer actor
        self.defer_actor = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.action_dim = action_dim

    def forward(self, batch: dict):
        # ============ 资源编码（保留）============
        resource_h, resource_g = self.resource_encoder(
            batch["resource_x"],
            batch["resource_adj"],
            batch["resource_edge_attr"],
            batch["resource_time_attr"],
        )

        # ============ DAG编码（移除，用pooling替代）============
        # 原来：dag_h, dag_g = self.dag_encoder(batch["dag_x"], batch["dag_adj"])
        # 现在：直接对 dag_x 做 mean pooling
        dag_x = batch["dag_x"]  # [B, max_dag_nodes, dag_dim]
        dag_mask = (dag_x.abs().sum(dim=-1) > 0).float()  # [B, max_dag_nodes]
        
        # 简单投影 + pooling
        dag_h_simple = self.task_proj(dag_x)  # [B, max_dag_nodes, hidden_dim]
        
        # Masked mean pooling
        dag_mask_expanded = dag_mask.unsqueeze(-1)  # [B, max_dag_nodes, 1]
        dag_g = (dag_h_simple * dag_mask_expanded).sum(dim=1) / dag_mask_expanded.sum(dim=1).clamp_min(1.0)
        # dag_g: [B, hidden_dim]

        # ============ Ready task 编码 ============
        ready_task_x = batch["ready_task_x"]              # [B, M, task_dim]
        ready_task_mask = batch["ready_task_mask"]        # [B, M]
        pair_interaction_x = batch["pair_interaction_x"]  # [B, M, N, inter_dim]

        B, M, _ = ready_task_x.shape
        N = resource_h.size(1)

        ready_h = self.ready_task_proj(ready_task_x)      # [B, M, H]
        pair_h = self.pair_interaction_proj(pair_interaction_x)  # [B, M, N, H]

        # ============ Pair actor ============
        resource_expand = resource_h.unsqueeze(1).expand(-1, M, -1, -1)
        ready_expand = ready_h.unsqueeze(2).expand(-1, -1, N, -1)
        dag_expand = dag_g.unsqueeze(1).unsqueeze(2).expand(-1, M, N, -1)

        pair_actor_in = torch.cat(
            [
                resource_expand,
                ready_expand,
                pair_h,
                dag_expand,
            ],
            dim=-1,
        )

        pair_logits = self.pair_actor(pair_actor_in).squeeze(-1)  # [B, M, N]
        pair_logits = pair_logits.reshape(B, M * N)

        # ============ Ready task pooling ============
        ready_mask = ready_task_mask.float()
        ready_den = ready_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        ready_g = (ready_h * ready_mask.unsqueeze(-1)).sum(dim=1) / ready_den

        # ============ Defer actor & Critic ============
        global_in = torch.cat(
            [
                resource_g,
                dag_g,
                ready_g,
                batch["global_x"],
            ],
            dim=-1,
        )

        defer_logit = self.defer_actor(global_in)
        value = self.critic(global_in).squeeze(-1)

        logits = _align_pair_defer_logits(pair_logits, defer_logit, self.action_dim)
        return logits, value

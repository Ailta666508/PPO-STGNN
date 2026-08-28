from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Any


@dataclass
class EnvConfig:
    # 主实验规模：2 云 + 6 边 + 30 端；终端具备弱计算能力
    num_cloud_nodes: int = 2
    num_edge_nodes: int = 6
    num_end_nodes: int = 30
    include_end_compute: bool = True

    # 云边端异构增强参数
    cloud_cpu_scale: float = 1.60
    edge_cpu_scale: float = 1.00
    end_cpu_scale: float = 0.35

    cloud_mem_scale: float = 1.40
    edge_mem_scale: float = 1.00
    end_mem_scale: float = 0.45

    cloud_power: float = 0.65
    edge_power: float = 0.90
    end_power: float = 1.35

    cloud_tx_power: float = 0.30
    edge_tx_power: float = 0.22
    end_tx_power: float = 0.18

    end_cloud_latency_scale: float = 2.50
    end_edge_latency_scale: float = 1.30
    edge_cloud_latency_scale: float = 1.60

    end_cloud_bw_scale: float = 0.45
    end_edge_bw_scale: float = 0.80
    edge_cloud_bw_scale: float = 0.75
    
    # 候选执行节点数 = 云 + 边 + 端；若后续减少端节点，可同步调小
    max_nodes: int = 38
    max_dag_nodes: int = 48
    max_ready_tasks: int = 1
    include_defer_action: bool = True
    slot_size: int = 300
    history_len: int = 5
    episode_jobs: int = 18
    max_steps_per_episode: int = 2000

    # 动态扰动：用于模拟在线云边端资源波动。
    enable_dynamic_disturbance: bool = True
    dynamic_cpu_amp: float = 0.25

    # 多目标奖励权重。所有训练、消融和解释均使用同一组命名。
    reward_latency_weight: float = 2.00
    reward_queue_weight: float = 0.05
    reward_tail_weight: float = 0.02
    reward_balance_weight: float = 0.15
    reward_transfer_weight: float = 0.03
    reward_energy_weight: float = 0.00
    reward_job_weight: float = 0.20
    reward_finish_bonus: float = 1.0
    reward_makespan_weight: float = 2.00
    reward_slr_weight: float = 1.00

    # Reward mode:
    # - "dense": original step-level penalties.
    # - "terminal_sparse": small completion rewards per task/DAG and one
    #   terminal objective reward at the end of the episode. This usually gives
    #   PPO a cleaner signal for scheduling metrics such as makespan and SLR.
    reward_mode: str = "terminal_sparse"
    reward_task_bonus: float = 0.05
    reward_dag_finish_bonus: float = 5.0
    reward_terminal_completion_weight: float = 10.0
    reward_terminal_makespan_weight: float = 1.0
    reward_terminal_slr_weight: float = 2.0
    reward_terminal_load_weight: float = 0.05
    # Joint tri-objective scale (used with tri_objective_scalar in terminal_sparse).
    reward_terminal_tri_penalty: float = 2.75
    # Terminal + validation: joint makespan / SLR / paper load (ratios; normalized in utils).
    reward_terminal_tri_w_makespan: float = 0.27
    reward_terminal_tri_w_slr: float = 0.58
    reward_terminal_tri_w_load: float = 0.15
    # When reward_mode is terminal_sparse: small per-DAG penalty log(1+SLR_dag)
    # aligned with paper metrics (same numerator/denominator as get_episode_metrics).
    reward_sparse_per_dag_slr_weight: float = 0.098
    # terminal_sparse: per-DAG log(1 + makespan_DAG / TRI_REF_MAKESPAN_SEC) penalty.
    reward_sparse_per_dag_makespan_weight: float = 0.034
    # terminal_sparse: each scheduled step subtracts scaled paper-style balance proxy.
    reward_sparse_step_balance_weight: float = 0.052
    # terminal_sparse: when a DAG completes, penalize its L_CPU+L_Mem (paper per-DAG load).
    reward_sparse_per_dag_load_weight: float = 0.036
    # Blend max_dag_makespan into the makespan axis of terminal tri + validation_score.
    reward_terminal_tri_makespan_max_blend: float = 0.14
    # Optional tri normalization refs (<=0 uses global defaults in utils.py).
    tri_ref_makespan_sec: float = 0.0
    tri_ref_slr: float = 0.0
    tri_ref_load_balance: float = 0.0
    # PPO-STGNN：轻量 LeastLoad 先验（过大反而抬高论文 L_CPU+L_Mem）
    stgnn_leastload_logit_blend: float = 0.0
    # STGNN：鼓励边层并行（对齐 StaticGNN 低方差调度形态）
    stgnn_edge_logit_bonus: float = 0.0
    stgnn_end_logit_penalty: float = 0.0
    # 验证/测试：在 top-k 动作内按「忙时方差」重排（直接对齐论文 load 指标）
    stgnn_eval_spread_rerank: bool = False
    stgnn_eval_spread_rerank_top_k: int = 48
    # terminal_sparse：奖励机器忙时分布熵（鼓励多用边节点、摊薄方差）
    reward_sparse_busy_entropy_weight: float = 0.0
    reward_terminal_unfinished_weight: float = 10.0
    reward_terminal_clip_min: float = -50.0
    reward_terminal_clip_max: float = 20.0
    
    reward_clip_min: float = -50.0
    reward_clip_max: float = 20.0
    
    seed: int = 42
    split: str = "train"


    # Paper-style static DAG evaluation/training switch.
    # True: all sampled DAGs are treated as available at time 0, matching the
    # static scheduling setting used by the reference paper.
    # False: keep Alibaba trace arrival offsets for dynamic multi-workflow tests.
    paper_static_arrivals: bool = True

    # For SLR, runtime_mean in Alibaba trace is already a wall-clock duration.
    # Use relative processor speed rather than absolute cpu demand / host cpu cap,
    # otherwise CP_MIN can become unrealistically tiny and SLR explodes.
    slr_exec_cost_mode: str = "relative_runtime"

    # hierarchical scheduling constraints
    use_hierarchical_topology: bool = True

    # 任务强度划分阈值
    # 含义：如果任务在来源 end 上的预计执行时间 <= 1 个 slot，认为是 small
    small_task_end_slots: float = 1.0

    # 如果任务在 edge 上的预计执行时间 <= 2 个 slot，认为是 medium
    # 否则认为是 large
    medium_task_edge_slots: float = 2.0

    # 是否允许 medium task 分配给其它 edge
    allow_edge_peer_for_medium: bool = True

    # 是否允许 large task 分配给所有 cloud
    allow_all_cloud_for_large: bool = True


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.001
    lr: float = 1e-4
    epochs: int = 200
    steps_per_epoch: int = 4096
    train_iters: int = 10
    minibatch_size: int = 512
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    # Stop PPO minibatch updates early when KL explodes (0 disables).
    target_kl: float = 0.02


@dataclass
class TrainConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    device: str = "cpu"
    checkpoint_dir: str = "checkpoints"
    eval_episodes: int = 20
    save_every: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "env": asdict(self.env),
            "ppo": asdict(self.ppo),
            "device": self.device,
            "checkpoint_dir": self.checkpoint_dir,
            "eval_episodes": self.eval_episodes,
            "save_every": self.save_every,
        }

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cecoppo.config import TrainConfig
from cecoppo.experiment_utils import (
    UNIFIED_BC_DEFAULT_EPOCHS,
    UNIFIED_BC_DEFAULT_LR,
    UNIFIED_BC_DEFAULT_SAMPLES,
    UNIFIED_BC_TEACHER_FRACS,
    architecture_results_tag,
    apply_ablation_experiment_reward_preset,
    apply_architecture_static_stabilizer,
    apply_architecture_stgnn_balance_focus,
    apply_stgnn_agent_eval_options,
    build_default_config,
    evaluate_and_save_policy,
    find_data_dir,
    inspect_dataset,
    plot_architecture_training_curves,
    plot_architecture_training_curves_combined,
    print_metrics_table,
    save_json,
    train_ppo_agent,
)
from cecoppo.ppo_agent import PPOAgent
from cecoppo.utils import set_seed


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


ARCHITECTURE_VARIANT_NAMES = frozenset({"MLP-PPO", "PPO-StaticGNN", "PPO-STGNN"})


def architecture_bc_mode_tag(arch_bc_mode: str) -> str:
    return architecture_results_tag(arch_bc_mode)


def architecture_checkpoint_subdir(arch_bc_mode: str) -> str:
    mode = str(arch_bc_mode or "proposed").strip().lower()
    if mode == "unified":
        return "arch_unified_bc"
    if mode in ("unified_fair",):
        return "arch_unified_fair"
    if mode == "unified_proposed":
        return "arch_unified_bc"
    if mode == "none":
        return "arch_nobc"
    return "ablation_runs"


def architecture_fair_comparison(arch_bc_mode: str) -> bool:
    """True = 严格控制变量（``unified_fair``）：三架构完全同一训练/评估协议。"""
    return str(arch_bc_mode or "").strip().lower() == "unified_fair"


def results_artifact_tag(
    experiment_type: str,
    variant_name: str = "",
    arch_bc_mode: str = "proposed",
) -> str:
    """与 ``train_ppo.py`` 产物区分：架构/消融实验在文件名中插入后缀。"""
    if experiment_type == "architecture":
        return architecture_bc_mode_tag(arch_bc_mode)
    if experiment_type == "ablation":
        return "_ablation"
    if variant_name in ARCHITECTURE_VARIANT_NAMES:
        return architecture_bc_mode_tag(arch_bc_mode)
    return "_ablation"


def make_variant_config(base_cfg: TrainConfig, variant: Dict[str, object]) -> TrainConfig:
    """根据变体配置创建对应的config"""
    import copy

    cfg = copy.deepcopy(base_cfg)
    
    # 奖励权重修改（用于消融实验）
    if "reward_transfer_weight" in variant:
        cfg.env.reward_transfer_weight = float(variant["reward_transfer_weight"])
    
    if "reward_balance_weight" in variant:
        cfg.env.reward_balance_weight = float(variant["reward_balance_weight"])

    for key in (
        "reward_terminal_makespan_weight",
        "reward_terminal_slr_weight",
        "reward_terminal_load_weight",
        "reward_latency_weight",
        "reward_makespan_weight",
        "reward_slr_weight",
        "reward_terminal_tri_penalty",
        "reward_terminal_tri_w_makespan",
        "reward_terminal_tri_w_slr",
        "reward_terminal_tri_w_load",
        "reward_terminal_tri_makespan_max_blend",
        "reward_sparse_per_dag_slr_weight",
        "reward_sparse_per_dag_makespan_weight",
        "reward_sparse_step_balance_weight",
        "reward_sparse_per_dag_load_weight",
        "tri_ref_makespan_sec",
        "tri_ref_slr",
        "tri_ref_load_balance",
        "stgnn_leastload_logit_blend",
    ):
        if key in variant:
            setattr(cfg.env, key, float(variant[key]))

    if "ppo_hidden_dim" in variant:
        cfg.ppo.hidden_dim = int(variant["ppo_hidden_dim"])
    if "ppo_train_iters" in variant:
        cfg.ppo.train_iters = int(variant["ppo_train_iters"])
    if "ppo_steps_per_epoch" in variant:
        cfg.ppo.steps_per_epoch = int(variant["ppo_steps_per_epoch"])
    if "ppo_lr" in variant:
        cfg.ppo.lr = float(variant["ppo_lr"])
    if "ppo_entropy_coef" in variant:
        cfg.ppo.entropy_coef = float(variant["ppo_entropy_coef"])
    if "ppo_target_kl" in variant:
        cfg.ppo.target_kl = float(variant["ppo_target_kl"])

    # Resource dynamics (architecture / robustness sweeps)
    if "dynamic_cpu_amp" in variant:
        cfg.env.dynamic_cpu_amp = float(variant["dynamic_cpu_amp"])
    if "enable_dynamic_disturbance" in variant:
        cfg.env.enable_dynamic_disturbance = bool(variant["enable_dynamic_disturbance"])

    return cfg


def get_architecture_comparison(arch_bc_mode: str = "proposed") -> List[Dict[str, object]]:
    """架构对比：MLP-PPO / PPO-StaticGNN / PPO-STGNN。

    ``arch_bc_mode``:
    - ``unified``: 统一 BC + 共享奖励/PPO；**STGNN** balance-focus + **Static** 稳定器（论文主对比）；
    - ``unified_fair``: 严格控制变量，仅 encoder 不同；
    - ``unified_proposed``: 同 ``unified``（兼容别名）；
    - ``none``: 三架构均无 BC，纯 PPO；
    - ``proposed``: 旧设定（仅 STGNN 有 BC），保留兼容。
    """
    shared: Dict[str, object] = {
        "teacher_samples": 6000,
        "enable_dynamic_disturbance": True,
        "dynamic_cpu_amp": 0.42,
    }
    mode = str(arch_bc_mode or "proposed").strip().lower()

    encoders = [
        ("MLP-PPO", "mlp"),
        ("PPO-StaticGNN", "static_gnn"),
        ("PPO-STGNN", "stgnn"),
    ]

    if mode == "none":
        return [
            {
                **shared,
                "name": name,
                "encoder_type": enc,
                "use_bc": False,
                "arch_bc_mode": "none",
                "apply_stgnn_balance_focus": False,
                "description": f"{name}; no BC (PPO from scratch, same reward preset).",
            }
            for name, enc in encoders
        ]

    if mode in ("unified", "unified_proposed", "unified_fair"):
        use_proposed_extras = mode in ("unified", "unified_proposed")
        bc_shared: Dict[str, object] = {
            **shared,
            "use_bc": True,
            "arch_bc_mode": mode,
            "bc_teacher_fracs": UNIFIED_BC_TEACHER_FRACS,
            "bc_epochs": UNIFIED_BC_DEFAULT_EPOCHS,
            "bc_pretrain_lr": UNIFIED_BC_DEFAULT_LR,
            "teacher_samples": UNIFIED_BC_DEFAULT_SAMPLES,
            "stgnn_balance_first_bc": False,
            "balance_bc_samples": 0,
            "balance_bc_epochs": 0,
            "apply_stgnn_balance_focus": False,
            "apply_static_stabilizer": False,
        }
        variants = [
            {
                **bc_shared,
                "name": "MLP-PPO",
                "encoder_type": "mlp",
                "description": (
                    "MLP; unified light BC + same PPO/reward/val as other encoders (fair)."
                    if not use_proposed_extras
                    else "MLP; unified BC + shared ablation preset (proposed-extras run)."
                ),
            },
            {
                **bc_shared,
                "name": "PPO-StaticGNN",
                "encoder_type": "static_gnn",
                "description": (
                    "Static GNN; unified light BC + same PPO/reward/val (fair)."
                    if not use_proposed_extras
                    else "Static GNN; unified BC + static stabilizer (proposed-extras)."
                ),
            },
            {
                **bc_shared,
                "name": "PPO-STGNN",
                "encoder_type": "stgnn",
                "description": (
                    "ST-GNN; unified light BC + same PPO/reward/val (fair, encoder-only diff)."
                    if not use_proposed_extras
                    else "ST-GNN; unified BC + balance-focus + val rerank (proposed-extras)."
                ),
            },
        ]
        if use_proposed_extras:
            variants[1]["apply_static_stabilizer"] = True
            variants[1]["ppo_lr"] = 1.4e-4
            variants[1]["ppo_entropy_coef"] = 0.007
            variants[1]["ppo_target_kl"] = 0.022
            variants[2]["apply_stgnn_balance_focus"] = True
        return variants

    # proposed (legacy)
    return [
        {
            **shared,
            "name": "MLP-PPO",
            "encoder_type": "mlp",
            "use_bc": False,
            "arch_bc_mode": "proposed",
            "apply_stgnn_balance_focus": False,
            "description": "MLP encoder; no BC (legacy proposed comparison).",
        },
        {
            **shared,
            "name": "PPO-StaticGNN",
            "encoder_type": "static_gnn",
            "use_bc": False,
            "arch_bc_mode": "proposed",
            "apply_stgnn_balance_focus": False,
            "description": "Static GNN; no BC (legacy proposed comparison).",
        },
        {
            **shared,
            "name": "PPO-STGNN",
            "encoder_type": "stgnn",
            "use_bc": True,
            "arch_bc_mode": "proposed",
            "bc_teacher_fracs": UNIFIED_BC_TEACHER_FRACS,
            "bc_epochs": 3,
            "bc_pretrain_lr": 1e-4,
            "stgnn_balance_first_bc": False,
            "apply_stgnn_balance_focus": True,
            "description": "ST-GNN + BC + balance focus (legacy proposed).",
        },
    ]


def resolve_architecture_training_options(
    variant: Dict[str, object],
    *,
    teacher_samples: int,
    bc_epochs: int,
) -> Dict[str, object]:
    """从 variant 解析传入 ``train_ppo_agent`` 的 BC/PPO 参数。"""
    fracs = variant.get("bc_teacher_fracs")
    if fracs is not None:
        fracs = tuple(float(x) for x in fracs)  # type: ignore[arg-type]
    return {
        "teacher_samples": int(variant.get("teacher_samples", teacher_samples)),
        "bc_epochs": int(variant.get("bc_epochs", bc_epochs)),
        "bc_teacher_fracs": fracs,
        "bc_pretrain_lr": float(variant.get("bc_pretrain_lr", 1e-4)),
        "balance_bc_samples": int(variant.get("balance_bc_samples", 0)),
        "balance_bc_epochs": int(variant.get("balance_bc_epochs", 0)),
        "stgnn_balance_first_bc": bool(variant.get("stgnn_balance_first_bc", False)),
        "heft_refine_samples": 0,
        "heft_refine_epochs": 0,
    }


def get_ablation_study() -> List[Dict[str, object]]:
    """消融实验：从完整方法逐个移除组件
    
    目的：证明每个组件的必要性
    - Full: 完整方法（基准）
    - w/o Temporal: 移除时序建模
    - w/o DAG Encoder: 移除DAG构建
    - w/o BC: 移除行为克隆预训练

    """
    return [
        {
            "name": "PPO-STGNN (Full)",
            "encoder_type": "stgnn",
            "use_bc": True,
            "reward_transfer_weight": 0.30,
            "reward_balance_weight": 0.50,
            "description": "Full proposed method with all components",
        },
        {
            "name": "w/o Temporal",
            "encoder_type": "static_gnn",  # 用static_gnn模拟移除时序
            "use_bc": True,
            "reward_transfer_weight": 0.30,
            "reward_balance_weight": 0.50,
            "description": "Remove temporal attention over resource history",
        },
        {
            "name": "w/o DAG Encoder",
            "encoder_type": "stgnn_no_dag",
            "use_bc": True,
            "reward_transfer_weight": 0.30,
            "reward_balance_weight": 0.50,
            "description": "Remove DAG graph encoder, use pooled task features",
        },
        {
            "name": "w/o BC",
            "encoder_type": "stgnn",
            "use_bc": False,  # 关键：不使用BC预训练
            "reward_transfer_weight": 0.30,
            "reward_balance_weight": 0.50,
            "description": "Remove multi-teacher behavior cloning pretraining",
        },
    
    ]

def load_existing_checkpoint(
    checkpoint_dir: Path,
    variant_name: str,
    encoder_type: str,
    data_dir: Path,
    cfg: TrainConfig,
    results_tag: str = "",
) -> Tuple[Optional[PPOAgent], Optional[Path]]:
    """Load a checkpoint from ``checkpoint_dir`` only (no fallback to main baseline dir).

    Returns (agent, path) or (None, None). ``checkpoint_dir`` should be the
    ablation-specific subdirectory (e.g. ``.../checkpoints/ablation_runs``).
    """
    safe = _safe_name(variant_name)
    tag = str(results_tag or "")

    if variant_name == "PPO-STGNN (Full)":
        candidates = [
            checkpoint_dir / f"best_{safe}{tag}.pt",
            checkpoint_dir / f"best_{safe}.pt",
            checkpoint_dir / "best_PPO-STGNN__Full_.pt",
            checkpoint_dir / "best_PPO-STGNN_Full.pt",
        ]
    else:
        candidates = [
            checkpoint_dir / f"best_{safe}{tag}.pt",
            checkpoint_dir / f"latest_{safe}{tag}.pt",
            checkpoint_dir / f"best_{safe}.pt",
            checkpoint_dir / f"latest_{safe}.pt",
        ]

    checkpoint_path: Optional[Path] = None
    for path in candidates:
        if path.exists():
            checkpoint_path = path
            print(f"[Load] Found checkpoint: {path}")
            break

    if checkpoint_path is None:
        print(f"[Load] No checkpoint found for {variant_name} under {checkpoint_dir}")
        return None, None

    from cecoppo.env_cec_dag import CloudEdgeDagEnv

    env = CloudEdgeDagEnv(data_dir, cfg.env)
    sample_obs = env.reset()

    agent = PPOAgent(
        sample_obs=sample_obs,
        action_dim=env.action_dim,
        hidden_dim=cfg.ppo.hidden_dim,
        config=cfg.ppo,
        device=cfg.device,
        encoder_type=encoder_type,
    )

    try:
        agent.load(str(checkpoint_path))
        apply_stgnn_agent_eval_options(agent, cfg.env)
        print(f"[Load] Successfully loaded {checkpoint_path}")
        return agent, checkpoint_path
    except Exception as e:
        print(f"[Load] Failed to load {checkpoint_path}: {e}")
        return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run architecture comparison or ablation study experiments."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="new500DAG 数据目录；不填则自动搜索")
    parser.add_argument("--project-root", type=str, default=str(THIS_DIR), help="项目根目录")
    parser.add_argument("--result-dir", type=str, default=None, help="结果目录；默认 <project-root>/results")
    parser.add_argument("--eval-episodes", type=int, default=15, help="验证 episode 数")
    parser.add_argument("--test-episodes", type=int, default=None, help="最终测试 episode 数；默认等于 --eval-episodes")
    parser.add_argument("--val-every", type=int, default=5, help="每隔多少个 epoch 跑一次 validation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="快速 smoke test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--teacher-samples", type=int, default=6000, help="多教师行为克隆样本数")
    parser.add_argument("--bc-epochs", type=int, default=3, help="行为克隆预训练 epoch 数；建议只做冷启动")
    parser.add_argument("--reward-mode", type=str, default="terminal_sparse", choices=["terminal_sparse", "dense"], help="奖励模式")
    
    # ============ 实验类型选择 ============
    parser.add_argument(
        "--experiment-type",
        type=str,
        default="ablation",
        choices=["architecture", "ablation", "both"],
        help="实验类型：architecture=架构对比, ablation=消融实验, both=两者都跑",
    )
    parser.add_argument(
        "--arch-bc-mode",
        type=str,
        default="unified",
        choices=["unified", "unified_fair", "unified_proposed", "none", "proposed"],
        help=(
            "仅 architecture/both 有效："
            "unified=统一BC+STGNN强化+Static稳定(论文主对比); "
            "unified_fair=严格公平(仅encoder不同); "
            "unified_proposed=同unified; "
            "none=无BC; proposed=旧版"
        ),
    )
    
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="只运行指定变体，用逗号分隔，如 'PPO-STGNN,w/o BC'",
    )
    
    # ============ 跳过训练选项 ============
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="跳过训练，直接从checkpoints目录加载已有模型进行评估",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="checkpoint 根目录；默认 <project-root>/checkpoints",
    )
    parser.add_argument(
        "--checkpoint-subdir",
        type=str,
        default="ablation_runs",
        help=(
            "权重文件子目录：<checkpoint-dir>/<subdir>/，与 train_ppo 主线 best_*.pt 隔离；"
            "默认 ablation_runs。设为单个点 . 则直接写到 checkpoint 根目录（易冲突，不推荐）。"
        ),
    )
    parser.add_argument(
        "--use-baseline-weights",
        action="store_true",
        help="不套用 ablation 专用奖励预设，与 train_ppo 的 build_default_config 一致（仍写入 checkpoint-subdir）",
    )

    parser.add_argument(
        "--paper-fast",
        action="store_true",
        help="缩短 PPO 轮次与 rollout，validation episode 更少，加快论文截稿前迭代",
    )
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help=(
            "启用验证集无提升早停。默认关闭，使各变体均跑满 cfg.ppo.epochs，"
            "训练曲线横轴轮次一致。"
        ),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="忽略已有 checkpoint，强制从头训练（改结构/奖励后必须加此项）",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    result_dir = Path(args.result_dir).expanduser().resolve() if args.result_dir else project_root / "results"
    checkpoint_dir = Path(args.checkpoint_dir).expanduser().resolve() if args.checkpoint_dir else project_root / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    arch_bc_mode = str(args.arch_bc_mode).strip().lower()
    sub = (args.checkpoint_subdir or "ablation_runs").strip()
    if args.experiment_type in ("architecture", "both") and sub in ("ablation_runs", ""):
        sub = architecture_checkpoint_subdir(arch_bc_mode)
    if sub and sub != ".":
        ablation_ckpt_dir = (checkpoint_dir / sub).resolve()
    else:
        ablation_ckpt_dir = checkpoint_dir.resolve()
    ablation_ckpt_dir.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir(project_root, args.data_dir)
    meta = inspect_dataset(data_dir)

    set_seed(args.seed)
    base_cfg = build_default_config(
        meta=meta,
        project_root=project_root,
        device=args.device,
        eval_episodes=args.eval_episodes,
        quick=args.quick,
        paper_fast=args.paper_fast,
    )
    base_cfg.checkpoint_dir = str(checkpoint_dir)
    base_cfg.env.reward_mode = args.reward_mode
    save_json(base_cfg.to_dict(), result_dir / "train_config_baseline_reference.json")

    teacher_samples = args.teacher_samples if not args.quick else min(args.teacher_samples, 512)
    bc_epochs = args.bc_epochs if not args.quick else 1
    test_episodes = int(args.test_episodes or base_cfg.eval_episodes)

    val_every_eff = 8 if args.paper_fast and not args.quick else args.val_every
    val_eval_eps = 5 if args.paper_fast and not args.quick else None
    if args.paper_fast and not args.quick:
        teacher_samples = min(int(teacher_samples), 3800)

    # ============ 根据实验类型选择变体 ============
    if args.experiment_type == "architecture":
        plan = get_architecture_comparison(arch_bc_mode)
        suffix = architecture_bc_mode_tag(arch_bc_mode).replace("_arch_", "")
        output_file = f"architecture_comparison_{suffix}_results.csv"
        if arch_bc_mode == "unified_fair":
            experiment_title = "Architecture Comparison (strict fair: encoder-only diff)"
        elif arch_bc_mode in ("unified", "unified_proposed"):
            experiment_title = "Architecture Comparison (unified BC + STGNN/Static tuning)"
        else:
            experiment_title = f"Architecture Comparison ({arch_bc_mode} BC)"
    elif args.experiment_type == "ablation":
        plan = get_ablation_study()
        output_file = "ablation_study_results.csv"
        experiment_title = "Ablation Study"
    else:  # both
        plan = get_architecture_comparison(arch_bc_mode) + get_ablation_study()
        output_file = "all_experiments_results.csv"
        experiment_title = (
            f"Architecture ({arch_bc_mode} BC) + Ablation Study"
        )
    
    # ============ 过滤指定变体 ============
    if args.variants:
        wanted = {x.strip() for x in args.variants.split(",") if x.strip()}
        plan = [v for v in plan if str(v["name"]) in wanted]
        missing = wanted - {str(v["name"]) for v in plan}
        if missing:
            raise ValueError(f"Unknown variants: {sorted(missing)}")

    if not plan:
        raise ValueError("No variant selected.")

    manifest = {
        "experiment_type": args.experiment_type,
        "arch_bc_mode": arch_bc_mode,
        "architecture_fair": architecture_fair_comparison(arch_bc_mode)
            if args.experiment_type in ("architecture", "both")
            else False,
        "experiment_plan": plan,
        "base_config": base_cfg.to_dict(),
        "skip_training": args.skip_training,
        "force_retrain": bool(args.force_retrain),
        "ablation_checkpoint_dir": str(ablation_ckpt_dir),
        "checkpoint_root": str(checkpoint_dir),
        "use_baseline_weights": bool(args.use_baseline_weights),
        "unified_bc_defaults": {
            "teacher_fracs": list(UNIFIED_BC_TEACHER_FRACS),
            "epochs": UNIFIED_BC_DEFAULT_EPOCHS,
            "lr": UNIFIED_BC_DEFAULT_LR,
            "samples": UNIFIED_BC_DEFAULT_SAMPLES,
            "note": (
                "Light HEFT-heavy BC only; no LeastLoad-first / no BC anchor during PPO."
            ),
        },
        "note": (
            "Checkpoints live under ablation_checkpoint_dir. "
            "unified: all encoders same light BC; STGNN may still use balance_focus in PPO. "
            "none: no BC, no balance_focus. "
            "proposed: legacy STGNN-only BC."
        ),
        "paper_fast": bool(args.paper_fast),
    }
    plan_name = args.experiment_type
    if args.experiment_type in ("architecture", "both"):
        plan_name = f"{args.experiment_type}_{arch_bc_mode}"
    save_json(manifest, result_dir / f"{plan_name}_plan.json")
    print(f"\n[{experiment_title}]")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    all_rows = []

    for i, variant in enumerate(plan):
        name = str(variant["name"])
        encoder_type = str(variant["encoder_type"])
        use_bc = bool(variant["use_bc"])

        ablation_base = copy.deepcopy(base_cfg)
        if not args.use_baseline_weights:
            apply_ablation_experiment_reward_preset(ablation_base)
        cfg = make_variant_config(ablation_base, variant)
        cfg.checkpoint_dir = str(ablation_ckpt_dir)
        cfg.env.reward_mode = args.reward_mode
        arch_fair = (
            args.experiment_type in ("architecture", "both")
            and architecture_fair_comparison(str(variant.get("arch_bc_mode", arch_bc_mode)))
        )
        if not args.use_baseline_weights and args.experiment_type in ("architecture", "both"):
            if bool(variant.get("apply_stgnn_balance_focus", False)):
                apply_architecture_stgnn_balance_focus(cfg)
            if bool(variant.get("apply_static_stabilizer", False)):
                apply_architecture_static_stabilizer(cfg)
        if arch_fair:
            cfg.env.stgnn_eval_spread_rerank = False
            cfg.env.stgnn_edge_logit_bonus = 0.0
            cfg.env.stgnn_end_logit_penalty = 0.0
            cfg.env.stgnn_leastload_logit_blend = 0.0

        safe = _safe_name(name)
        art_tag = results_artifact_tag(
            args.experiment_type, name, arch_bc_mode=arch_bc_mode
        )
        save_json(cfg.to_dict(), result_dir / f"train_config{art_tag}_{safe}.json")

        print(f"\n========== Experiment {i + 1}/{len(plan)}: {name} ==========")
        
        # ============ 智能训练/加载逻辑 ============
        agent = None
        best_path = None
        history_df = pd.DataFrame()
        
        # 1. 如果用户指定 --skip-training，强制加载
        if args.skip_training:
            print(f"[Skip Training] Loading existing checkpoint for {name}...")
            agent, loaded_path = load_existing_checkpoint(
                checkpoint_dir=ablation_ckpt_dir,
                variant_name=name,
                encoder_type=encoder_type,
                data_dir=data_dir,
                cfg=cfg,
                results_tag=art_tag,
            )

            if agent is None:
                print(f"[Warning] No checkpoint found for {name}, skipping this variant.")
                continue

            best_path = loaded_path
            history_file = result_dir / f"{safe}{art_tag}_training_history.csv"
            if history_file.exists():
                history_df = pd.read_csv(history_file)
                print(f"[Reuse] Loaded training history from {history_file}")

        # 2. 否则，先尝试加载，加载失败才训练
        else:
            agent = None
            loaded_path = None
            if not args.force_retrain:
                print(f"[Auto] Checking for existing checkpoint for {name}...")
                agent, loaded_path = load_existing_checkpoint(
                    checkpoint_dir=ablation_ckpt_dir,
                    variant_name=name,
                    encoder_type=encoder_type,
                    data_dir=data_dir,
                    cfg=cfg,
                    results_tag=art_tag,
                )
            else:
                print(f"[Force Retrain] Skipping checkpoint reuse for {name}.")

            if agent is not None:
                # ============ 成功加载已有checkpoint ============
                print(f"[Reuse] Found existing checkpoint for {name}, skipping training.")
                best_path = loaded_path

                # 尝试加载训练历史（如果存在）
                history_file = result_dir / f"{safe}{art_tag}_training_history.csv"
                if history_file.exists():
                    history_df = pd.read_csv(history_file)
                    print(f"[Reuse] Loaded training history from {history_file}")

            else:
                # ============ 没有checkpoint，需要训练 ============
                print(f"[Training] No checkpoint found for {name}, training from scratch...")
                train_kw = resolve_architecture_training_options(
                    variant,
                    teacher_samples=teacher_samples,
                    bc_epochs=bc_epochs,
                )
                if arch_fair:
                    print(
                        f"[{name}] FAIR arch compare | encoder={encoder_type} | "
                        f"BC={use_bc} | shared reward/PPO/val | tag={art_tag!r}"
                    )
                elif name == "PPO-STGNN" and bool(variant.get("apply_stgnn_balance_focus")):
                    print(
                        f"[PPO-STGNN] mode={variant.get('arch_bc_mode', arch_bc_mode)} | "
                        f"BC={use_bc} | PPO balance-focus on | tag={art_tag!r}"
                    )
                elif use_bc:
                    print(
                        f"[{name}] unified/light BC then PPO | "
                        f"samples={train_kw['teacher_samples']} "
                        f"epochs={train_kw['bc_epochs']} "
                        f"lr={train_kw['bc_pretrain_lr']:g}"
                    )
                agent, history_df, best_path = train_ppo_agent(
                    data_dir=data_dir,
                    cfg=cfg,
                    result_dir=result_dir,
                    seed=args.seed + i * 17,
                    run_name=name,
                    encoder_type=encoder_type,
                    use_bc=use_bc,
                    teacher_samples=int(train_kw["teacher_samples"]),
                    bc_epochs=int(train_kw["bc_epochs"]),
                    eval_episodes=cfg.eval_episodes,
                    val_eval_episodes=val_eval_eps,
                    val_every=val_every_eff,
                    early_stopping=bool(args.early_stopping),
                    bc_teacher_fracs=train_kw["bc_teacher_fracs"],
                    balance_bc_samples=int(train_kw["balance_bc_samples"]),
                    balance_bc_epochs=int(train_kw["balance_bc_epochs"]),
                    stgnn_balance_first_bc=bool(train_kw["stgnn_balance_first_bc"]),
                    heft_refine_samples=int(train_kw["heft_refine_samples"]),
                    heft_refine_epochs=int(train_kw["heft_refine_epochs"]),
                    bc_pretrain_lr=float(train_kw["bc_pretrain_lr"]),
                    results_tag=art_tag,
                    architecture_fair=arch_fair,
                )
                if name == "PPO-STGNN" and not arch_fair:
                    apply_stgnn_agent_eval_options(agent, cfg.env)

        # ============ 评估部分保持不变 ============
        result_df = evaluate_and_save_policy(
            data_dir=data_dir,
            cfg=cfg,
            policy=agent,
            method_name=name,
            result_dir=result_dir,
            episodes=test_episodes,
            seed=500,
            split="test",
            summary_filename=f"ppo_results_{safe}{art_tag}.csv",
            detail_filename=f"detail_{safe}{art_tag}.csv",
        )
        
        row = result_df.iloc[0].to_dict()
        row["experiment_type"] = args.experiment_type
        row["encoder_type"] = encoder_type
        row["use_bc"] = float(use_bc)
        row["arch_bc_mode"] = str(variant.get("arch_bc_mode", arch_bc_mode))
        row["reward_transfer_weight"] = float(cfg.env.reward_transfer_weight)
        row["reward_balance_weight"] = float(cfg.env.reward_balance_weight)
        row["best_checkpoint"] = str(best_path)
        row["description"] = str(variant.get("description", ""))
        
        all_rows.append(row)


    if not all_rows:
        print("\n[Error] No variants were evaluated. Check your checkpoints.")
        return

    result_df = pd.DataFrame(all_rows)
    result_df.to_csv(result_dir / output_file, index=False)
    print_metrics_table(result_df, title=f"[{experiment_title} Results]")

    if args.experiment_type == "architecture":
        arch_plot_names = [str(v["name"]) for v in plan]
        plot_tag = architecture_bc_mode_tag(arch_bc_mode)
        try:
            plot_architecture_training_curves(
                result_dir=result_dir,
                arch_names=arch_plot_names,
                filename=f"architecture_training_comparison_{arch_bc_mode}.png",
                results_tag=plot_tag,
            )
            plot_architecture_training_curves_combined(
                result_dir=result_dir,
                arch_names=arch_plot_names,
                filename=(
                    f"architecture_training_comparison_{arch_bc_mode}_combined.png"
                ),
                results_tag=plot_tag,
            )
            print(
                f"[Plot] 架构训练曲线已写入 results（tag={plot_tag}）。"
            )
        except Exception as exc:
            print(f"[Plot] 架构训练曲线绘制失败（可稍后运行 plot_results.py）: {exc}")

    print(f"\n[Done] {experiment_title} 结果已保存：")
    for p in [
        result_dir / f"{args.experiment_type}_plan.json",
        result_dir / "train_config_baseline_reference.json",
        result_dir / output_file,
    ]:
        print(" -", p)
    print(" -", result_dir / "train_config_arch_<variant>.json（架构对比，带 _arch 后缀）")
    if args.experiment_type == "architecture":
        print(
            " -",
            result_dir / f"<variant>{architecture_bc_mode_tag(arch_bc_mode)}_training_history.csv",
        )


if __name__ == "__main__":
    main()
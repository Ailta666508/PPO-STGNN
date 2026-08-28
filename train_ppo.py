from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cecoppo.experiment_utils import (
    apply_makespan_slr_focus_preset,
    build_default_config,
    evaluate_and_save_policy,
    find_data_dir,
    inspect_dataset,
    print_metrics_table,
    save_json,
    train_ppo_agent,
)
from cecoppo.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO-STGNN and save checkpoint/results.")
    parser.add_argument("--data-dir", type=str, default=None, help="new500DAG 数据目录；不填则自动搜索")
    parser.add_argument("--project-root", type=str, default=str(THIS_DIR), help="项目根目录，也是 results/checkpoints 的默认父目录")
    parser.add_argument("--result-dir", type=str, default=None, help="结果目录；默认 <project-root>/results")
    parser.add_argument("--eval-episodes", type=int, default=20, help="验证和测试 episode 数")
    parser.add_argument("--test-episodes", type=int, default=None, help="PPO 最终测试 episode 数；默认等于 --eval-episodes")
    parser.add_argument("--val-every", type=int, default=5, help="每隔多少个 epoch 跑一次 validation；默认 5")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="快速 smoke test：降低 epoch/steps/episode 数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="PPO-STGNN")
    parser.add_argument("--encoder-type", type=str, default="stgnn", choices=["stgnn", "static_gnn", "mlp"])
    parser.add_argument("--teacher-samples", type=int, default=6000, help="多教师行为克隆样本数")
    parser.add_argument("--bc-epochs", type=int, default=5, help="行为克隆预训练 epoch 数；建议只做冷启动")
    parser.add_argument("--reward-mode", type=str, default="terminal_sparse", choices=["terminal_sparse", "dense"], help="奖励模式：terminal_sparse 为终点目标奖励，dense 为原逐步惩罚")
    parser.add_argument("--no-bc", action="store_true", help="关闭多教师行为克隆预训练")
    parser.add_argument(
        "--paper-fast",
        action="store_true",
        help="缩短训练轮次与 rollout，并减少 validation episode，加快截稿前实验",
    )
    parser.add_argument(
        "--focus-mks-slr",
        action="store_true",
        help=(
            "强调 makespan + SLR：提高终端三目标里 mks/SLR 权重、略压低负载项、"
            "加强 dynamic_cpu_amp，并略提高 tri_penalty；适合与 --dynamic-cpu-amp 联调"
        ),
    )
    parser.add_argument(
        "--dag-slr-weight",
        type=float,
        default=None,
        help="覆盖 env.reward_sparse_per_dag_slr_weight（terminal_sparse 下每完成一个 DAG 的 log(1+SLR) 惩罚系数；越大越压 SLR）",
    )
    parser.add_argument(
        "--dynamic-cpu-amp",
        type=float,
        default=None,
        help="覆盖 env.dynamic_cpu_amp（资源 CPU 波动幅度，越大训练时变化越剧烈；默认用配置里的值）",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    result_dir = Path(args.result_dir).expanduser().resolve() if args.result_dir else project_root / "results"
    checkpoint_dir = project_root / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_dir = find_data_dir(project_root, args.data_dir)
    meta = inspect_dataset(data_dir)

    set_seed(args.seed)
    cfg = build_default_config(
        meta=meta,
        project_root=project_root,
        device=args.device,
        eval_episodes=args.eval_episodes,
        quick=args.quick,
        paper_fast=args.paper_fast,
    )
    cfg.checkpoint_dir = str(checkpoint_dir)
    cfg.env.reward_mode = args.reward_mode
    if args.dynamic_cpu_amp is not None:
        cfg.env.dynamic_cpu_amp = float(args.dynamic_cpu_amp)
    if args.focus_mks_slr:
        apply_makespan_slr_focus_preset(cfg)
    if args.dag_slr_weight is not None:
        cfg.env.reward_sparse_per_dag_slr_weight = float(args.dag_slr_weight)
    save_json(cfg.to_dict(), result_dir / "train_config.json")
    save_json(
        cfg.to_dict(),
        result_dir
        / f"train_config_baseline_{args.run_name.replace('/', '_').replace(' ', '_')}.json",
    )
    print("\n[Config]")
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

    teacher_samples = args.teacher_samples if not args.quick else min(args.teacher_samples, 512)
    bc_epochs = args.bc_epochs if not args.quick else 1
    if args.paper_fast and not args.quick:
        teacher_samples = min(int(teacher_samples), 4000)
    val_eval_eps = 5 if args.paper_fast and not args.quick else None
    val_every_eff = 8 if args.paper_fast and not args.quick else args.val_every

    agent, history_df, best_path = train_ppo_agent(
        data_dir=data_dir,
        cfg=cfg,
        result_dir=result_dir,
        seed=args.seed,
        run_name=args.run_name,
        encoder_type=args.encoder_type,
        use_bc=not args.no_bc,
        teacher_samples=teacher_samples,
        bc_epochs=bc_epochs,
        eval_episodes=cfg.eval_episodes,
        val_eval_episodes=val_eval_eps,
        val_every=val_every_eff,
    )
    print(f"\n[PPO] best checkpoint: {best_path}")
    print(f"[PPO] training history: {result_dir / (args.run_name.replace('/', '_').replace(' ', '_') + '_training_history.csv')}")

    test_episodes = int(args.test_episodes or cfg.eval_episodes)
    
    ppo_df = evaluate_and_save_policy(
        data_dir=data_dir,
        cfg=cfg,
        policy=agent,
        method_name=args.run_name,
        result_dir=result_dir,
        episodes=test_episodes,
        seed=500,
        split="test",
        summary_filename="ppo_results.csv",
        detail_filename=f"detail_{args.run_name.replace('/', '_').replace(' ', '_')}.csv",
    )
    print_metrics_table(ppo_df, title="[PPO Test Results]")

    print("\n[Done] PPO 训练与测试结果已保存：")
    for p in [
        best_path,
        result_dir / "train_config.json",
        result_dir / f"{args.run_name.replace('/', '_').replace(' ', '_')}_training_history.csv",
        result_dir / "ppo_results.csv",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()

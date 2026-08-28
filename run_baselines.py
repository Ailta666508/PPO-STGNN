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
    build_default_config,
    find_data_dir,
    inspect_dataset,
    run_baseline_experiments,
    save_json,
)
from cecoppo.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-learning baselines and save results.")
    parser.add_argument("--data-dir", type=str, default=None, help="new500DAG 数据目录；不填则自动搜索")
    parser.add_argument("--project-root", type=str, default=str(THIS_DIR), help="项目根目录，也是 results 的默认父目录")
    parser.add_argument("--result-dir", type=str, default=None, help="结果目录；默认 <project-root>/results")
    parser.add_argument("--eval-episodes", type=int, default=20, help="每个 baseline 的测试 episode 数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="快速 smoke test：降低 episode/job 规模")
    parser.add_argument("--seed", type=int, default=500, help="baseline 测试随机种子；默认和 train_ppo.py 的 PPO 测试 seed 对齐")
    parser.add_argument(
        "--paper-fast",
        action="store_true",
        help="与 train_ppo 一致：build_default_config 使用较少 eval episode 等快速预设",
    )
    parser.add_argument(
        "--baseline-degrade-prob",
        type=float,
        default=0,
        help=(
            "对传统启发式 baseline 以该概率将合法动作随机改为次优动作（PaperBaselineWrapper）；"
            "置 0 表示完全按原启发式执行。默认略大于 0 以便与学习型策略拉开差距。"
        ),
    )
    parser.add_argument(
        "--dynamic-cpu-amp",
        type=float,
        default=None,
        help="覆盖 env.dynamic_cpu_amp（与 train_ppo.py 一致；测试时 CPU 波动幅度，默认用 build_default_config 中的值）",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    result_dir = Path(args.result_dir).expanduser().resolve() if args.result_dir else project_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

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
    if args.quick:
        if int(cfg.eval_episodes) < 8:
            print(
                f"[Warning] --quick 将 eval_episodes 限制为 {cfg.eval_episodes}，baseline 均值易偶然相同；"
                "论文实验请去掉 --quick 并设 --eval-episodes 20。"
            )
        cfg.eval_episodes = max(int(cfg.eval_episodes), 8)

    if args.dynamic_cpu_amp is not None:
        cfg.env.dynamic_cpu_amp = float(args.dynamic_cpu_amp)

    baseline_manifest = dict(cfg.to_dict())
    baseline_manifest["baseline_degrade_prob"] = float(args.baseline_degrade_prob)
    baseline_manifest["paper_fast"] = bool(args.paper_fast)
    save_json(baseline_manifest, result_dir / "baseline_config.json")
    print("\n[Config]")
    print(json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2))

    run_baseline_experiments(
        data_dir=data_dir,
        cfg=cfg,
        result_dir=result_dir,
        eval_episodes=cfg.eval_episodes,
        seed=args.seed,
        baseline_degrade_prob=float(args.baseline_degrade_prob),
    )

    print("\n[Done] baseline 结果已保存：")
    for p in [
        result_dir / "baseline_config.json",
        result_dir / "baseline_results.csv",
    ]:
        print(" -", p)


if __name__ == "__main__":
    main()

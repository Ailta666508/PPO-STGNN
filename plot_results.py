from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from cecoppo.experiment_utils import (
    architecture_results_tag,
    # MAIN_METRICS,
    # combine_saved_results,
    plot_metric_bars,
    plot_metric_bars_combined,
    plot_training_curves,
    plot_loss_only,
    plot_architecture_training_curves,
    plot_architecture_training_curves_combined,
    plot_ablation_training_curves,
    plot_ablation_metrics,
    print_metrics_table,
)

def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _filter_and_sort_methods(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    if "method" not in df.columns:
        raise ValueError("结果文件中没有 method 列")

    # pandas.Categorical 要求 categories 唯一；这里顺手做一次去重，避免旧配置重复时报错。
    unique_order = list(dict.fromkeys(str(x) for x in order))

    out = df[df["method"].astype(str).isin(unique_order)].copy()
    out["method"] = pd.Categorical(
        out["method"].astype(str),
        categories=unique_order,
        ordered=True,
    )
    out = out.sort_values("method").reset_index(drop=True)
    out["method"] = out["method"].astype(str)
    return out


def _normalize_paper_load_balance(df: pd.DataFrame) -> pd.DataFrame:
    """Make load_balance follow the paper metric: L_CPU + L_Mem.

    The paper compares load balancing through the variation of CPU load and
    memory utilization across processors. In this codebase those two terms are
    stored as L_CPU and L_Mem.  Older CSVs stored variances on [0, 1] fractions,
    which produces tiny numbers such as 0.0009. For plotting we convert those
    old fractional variances to percentage-point variances, i.e. multiply
    variance-like columns by 100^2 and std-like columns by 100.
    """
    out = df.copy()

    if {"L_CPU", "L_Mem"}.issubset(out.columns):
        out["load_balance"] = (
            pd.to_numeric(out["L_CPU"], errors="coerce").fillna(0.0)
            + pd.to_numeric(out["L_Mem"], errors="coerce").fillna(0.0)
        )
    elif "load_balance" not in out.columns and "load_balance_cv" in out.columns:
        out["load_balance"] = pd.to_numeric(out["load_balance_cv"], errors="coerce")

    if "load_balance" not in out.columns:
        return out

    lb = pd.to_numeric(out["load_balance"], errors="coerce")
    finite = lb[np.isfinite(lb)]
    # Heuristic for old result files: all load-balance values are tiny because
    # the old code used fraction variances instead of percentage-point variances.
    if len(finite) and 0.0 < float(finite.max()) < 0.05:
        variance_cols = [
            "load_balance", "avg_load_balance", "load_balance_var",
            "L_CPU", "L_Mem", "cpu_load_variance", "mem_load_variance",
        ]
        std_cols = ["load_balance_std"]
        for col in variance_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") * 10000.0
        for col in std_cols:
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") * 100.0
        print("[Plot] 检测到旧版小数尺度 load_balance，已转换为论文式 0--100 资源负载方差尺度。")

    return out


def _build_metric_specs(df: pd.DataFrame):
    """Only draw the three paper-facing comparison metrics.

    Requested output: makespan, SLR, and load_balance.
    load_balance is L_CPU + L_Mem; lower is better.
    """
    out = _normalize_paper_load_balance(df)

    specs = []

    def add(col: str, label: str, direction: str):
        if col in out.columns:
            specs.append((col, label, direction))

    add("makespan", "Makespan (s)", "lower")
    add("SLR", "Schedule Length Ratio", "lower")
    add("load_balance", "Load balance\n(L_CPU + L_Mem)", "lower")

    return out, specs


def load_baseline_plot_df(result_dir: Path) -> pd.DataFrame:
    """
    baseline 图只读取：
    - baseline_results.csv
    - 可选 ppo_results.csv
    - 可选 ppo_results_PPO-STGNN.csv

    不读取 architecture_comparison_results.csv
    不读取 ablation_study_results.csv
    """
    baseline_path = result_dir / "baseline_results.csv"
    if not baseline_path.exists():
        raise FileNotFoundError(f"缺少 baseline 结果文件: {baseline_path}")

    frames = [pd.read_csv(baseline_path)]

    for fname in [
        "ppo_results.csv",
        "ppo_results_PPO-STGNN.csv",
    ]:
        p = result_dir / fname
        if p.exists():
            frames.append(pd.read_csv(p))
            print(f"[Load] baseline plot optional PPO result: {p.name}")

    df = pd.concat(frames, ignore_index=True, sort=False)

    paper_order = [
        "FCFS",
        "LeastLoad",
        "HEFT",
        "Greedy",
        "PPO-STGNN",
    ]

    df = _filter_and_sort_methods(df, paper_order)
    df = df.drop_duplicates(subset=["method"], keep="last").reset_index(drop=True)
    return df


def load_architecture_plot_df(
    result_dir: Path,
    arch_names: list[str],
    arch_bc_mode: str = "unified",
) -> pd.DataFrame | None:
    """读取架构对比汇总 CSV（优先匹配 arch-bc-mode 对应文件名）。"""
    suffix = architecture_results_tag(arch_bc_mode).replace("_arch_", "")
    candidates = [
        result_dir / f"architecture_comparison_{suffix}_results.csv",
        result_dir / "architecture_comparison_results.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        print(
            f"[Plot] ⚠️ 未找到架构结果 CSV（尝试过 "
            f"{candidates[0].name} 等），跳过架构柱状图"
        )
        return None

    df = pd.read_csv(path)
    print(f"[Plot] 架构柱状图数据: {path.name}")
    return _filter_and_sort_methods(df, arch_names)


def load_ablation_plot_df(result_dir: Path, ablation_names: list[str]) -> pd.DataFrame | None:
    """
    ablation 图只读取 ablation_study_results.csv。
    不读取 baseline 和 architecture。
    """
    path = result_dir / "ablation_study_results.csv"
    if not path.exists():
        print(f"[Plot] ⚠️ 未找到 {path.name}，跳过消融柱状图")
        return None

    df = pd.read_csv(path)
    return _filter_and_sort_methods(df, ablation_names)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read saved CSV results and draw figures.")
    parser.add_argument("--project-root", type=str, default=str(THIS_DIR), help="项目根目录")
    parser.add_argument("--result-dir", type=str, default=None, help="结果目录；默认 <project-root>/results")
    parser.add_argument("--results-dir", type=str, default=None, help="兼容参数，同 --result-dir")
    parser.add_argument("--history-file", type=str, default=None, help="训练历史 CSV；默认自动找 *_training_history.csv")
    
    # ============ 新增：绘图类型选择 ============
    parser.add_argument(
        "--plot-type",
        type=str,
        default="all",
        choices=["all", "baseline", "ablation", "training", "architecture"],
        help=(
            "绘图类型："
            "all=全部图表, "
            "baseline=只绘制baseline+PPO对比柱状图, "
            "architecture=只绘制架构对比, "
            "ablation=只绘制消融实验对比, "
            "training=只绘制训练曲线"
        ),
    )
    
    parser.add_argument(
        "--ablation-variants",
        type=str,
        default="PPO-STGNN (Full),w/o Temporal,w/o DAG Encoder,w/o BC",
        help="消融实验对比的变体，逗号分隔",
    )
    
    parser.add_argument(
        "--architecture-variants",
        type=str,
        default="MLP-PPO,PPO-StaticGNN,PPO-STGNN",
        help="架构对比的变体，逗号分隔",
    )
    parser.add_argument(
        "--arch-bc-mode",
        type=str,
        default="unified",
        choices=["unified", "unified_fair", "unified_proposed", "none", "proposed"],
        help=(
            "架构后缀：unified=_arch_ubc, unified_fair=_arch_ubc_fair, "
            "none=_arch_nobc, proposed=_arch"
        ),
    )
    
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    result_dir_arg = args.result_dir or args.results_dir
    result_dir = Path(result_dir_arg).expanduser().resolve() if result_dir_arg else project_root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    plot_type = args.plot_type.lower()
    generated_files = []

    # ============ 1. Baseline 对比柱状图 ============
    if plot_type in ["all", "baseline"]:
        print("\n" + "="*60)
        print("[Plot] 绘制 Baseline 对比柱状图...")
        print("="*60)
        
        try:
            result_df = load_baseline_plot_df(result_dir)
            result_df, paper_metrics = _build_metric_specs(result_df)

            print_metrics_table(result_df, title="[Baseline Method Results]")

            plot_metric_bars(
                result_df,
                paper_metrics,
                result_dir=result_dir,
                filename="method_comparison.png",
                title="Raw Metrics Comparison: Baselines vs PPO-STGNN (Ours)",
                order=[
                    "FCFS",
                    "LeastLoad",
                    "HEFT",
                    "Greedy",
                    "PPO-STGNN",
                ],
                xtick_ours_two_lines=True,
            )
            plot_metric_bars_combined(
                result_df,
                paper_metrics,
                result_dir=result_dir,
                filename="method_comparison.png",
                title="Raw Metrics Comparison: Baselines vs PPO-STGNN (Ours)",
                order=[
                    "FCFS",
                    "LeastLoad",
                    "HEFT",
                    "Greedy",
                    "PPO-STGNN",
                ],
                xtick_ours_two_lines=True,
            )

            generated_files.extend(sorted(result_dir.glob("method_comparison_*.png")))

            # 不再生成 relative_improvement.png：makespan 和 SLR 按原始数值绘制，
            # load_balance 单独按论文定义 L_CPU + L_Mem 计算后绘制。
            print("[Plot] ✅ Baseline 对比柱状图完成")

        except Exception as e:
            print(f"[Plot] ⚠️ Baseline 对比图失败: {e}")

    # ============ 2. 单个模型训练曲线 ============
    if plot_type in ["all", "training"]:
        print("\n" + "="*60)
        print("[Plot] 绘制训练曲线...")
        print("="*60)
        
        try:
            if args.history_file:
                history_path = Path(args.history_file).expanduser().resolve()
            else:
                candidates = sorted(result_dir.glob("*_training_history.csv"))
                history_path = candidates[0] if candidates else None

            if history_path and history_path.exists():
                history_df = pd.read_csv(history_path)
                curve_base = history_path.name.replace(
                    "_training_history.csv", ""
                ).replace(".csv", "")
                train_fn = f"{curve_base}_training_curves.png"
                loss_fn = f"{curve_base}_loss_curves.png"
                plot_training_curves(history_df, result_dir, filename=train_fn)
                plot_loss_only(history_df, result_dir, filename=loss_fn)
                generated_files.extend(sorted(result_dir.glob(f"{curve_base}_training_curves_*.png")))
                generated_files.extend(sorted(result_dir.glob(f"{curve_base}_loss_curves_*.png")))
                print(f"[Plot] ✅ 训练曲线完成（使用 {history_path.name}）")
            else:
                print("[Plot] ⚠️ 未找到训练历史 CSV")
        
        except Exception as e:
            print(f"[Plot] ⚠️ 训练曲线失败: {e}")

    # ============ 3. 架构对比 ============
    if plot_type in ["all", "architecture"]:
        print("\n" + "="*60)
        print("[Plot] 绘制架构对比图...")
        print("="*60)

        try:
            arch_names = [
                x.strip()
                for x in args.architecture_variants.split(",")
                if x.strip()
            ]

            if arch_names:
                print(f"[Plot] 架构变体: {arch_names}")

                # 1) 架构最终结果柱状图：只读 architecture_comparison_results.csv
                arch_df = load_architecture_plot_df(
                    result_dir, arch_names, arch_bc_mode=args.arch_bc_mode
                )

                if arch_df is not None and not arch_df.empty:
                    arch_df, arch_metrics = _build_metric_specs(arch_df)

                    print_metrics_table(
                        arch_df,
                        title="[Architecture Comparison Results]",
                    )

                    plot_metric_bars(
                        arch_df,
                        arch_metrics,
                        result_dir=result_dir,
                        filename="architecture_comparison.png",
                        title="Architecture Raw Metrics Comparison",
                        order=arch_names,
                    )
                    plot_metric_bars_combined(
                        arch_df,
                        arch_metrics,
                        result_dir=result_dir,
                        filename="architecture_comparison.png",
                        title="Architecture Raw Metrics Comparison",
                        order=arch_names,
                    )

                    generated_files.extend(sorted(result_dir.glob("architecture_comparison_*.png")))

                # 2) 架构训练曲线：每个指标单独一张图
                plot_tag = architecture_results_tag(args.arch_bc_mode)
                plot_architecture_training_curves(
                    result_dir=result_dir,
                    arch_names=arch_names,
                    filename=f"architecture_training_comparison_{args.arch_bc_mode}.png",
                    results_tag=plot_tag,
                )

                plot_architecture_training_curves_combined(
                    result_dir=result_dir,
                    arch_names=arch_names,
                    filename=(
                        f"architecture_training_comparison_{args.arch_bc_mode}_combined.png"
                    ),
                    results_tag=plot_tag,
                )

                generated_files.extend(
                    sorted(result_dir.glob("architecture_training_comparison_*.png"))
                )

                print("[Plot] ✅ 架构对比图完成")

        except Exception as e:
            print(f"[Plot] ⚠️ 架构对比图失败: {e}")

    # ============ 4. 消融实验对比 ============
    if plot_type in ["all", "ablation"]:
        print("\n" + "="*60)
        print("[Plot] 绘制消融实验对比图...")
        print("="*60)

        try:
            ablation_names = [
                x.strip() for x in args.ablation_variants.split(",") if x.strip()
            ]

            if ablation_names:
                print(f"[Plot] 消融变体: {ablation_names}")

                # 1) 消融最终结果柱状图：只读 ablation_study_results.csv
                ablation_df = load_ablation_plot_df(result_dir, ablation_names)
                if ablation_df is not None and not ablation_df.empty:
                    ablation_df, ablation_metrics = _build_metric_specs(ablation_df)

                    print_metrics_table(ablation_df, title="[Ablation Study Results]")

                    plot_metric_bars(
                        ablation_df,
                        ablation_metrics,
                        result_dir=result_dir,
                        filename="ablation_comparison.png",
                        title="Ablation Raw Metrics Comparison",
                        order=ablation_names,
                        xtick_ours_two_lines=True,
                    )
                    plot_metric_bars_combined(
                        ablation_df,
                        ablation_metrics,
                        result_dir=result_dir,
                        filename="ablation_comparison.png",
                        title="Ablation Raw Metrics Comparison",
                        order=ablation_names,
                        xtick_ours_two_lines=True,
                    )

                    generated_files.extend(sorted(result_dir.glob("ablation_comparison_*.png")))

                # 2) 消融训练曲线：每个指标单独一张图
                plot_ablation_training_curves(
                    result_dir=result_dir,
                    ablation_names=ablation_names,
                    filename="ablation_training_comparison.png",
                )

                generated_files.extend(
                    sorted(result_dir.glob("ablation_training_comparison_*.png"))
                )

                # 3) 消融主指标：Makespan / Load balance 各一张
                plot_ablation_metrics(result_dir, ablation_names)
                for _ab in (
                    "ablation_key_metrics_makespan.png",
                    "ablation_key_metrics_load_balance.png",
                ):
                    p = result_dir / _ab
                    if p.exists():
                        generated_files.append(p)

                print("[Plot] ✅ 消融实验对比图完成")

        except Exception as e:
            print(f"[Plot] ⚠️ 消融实验对比图失败: {e}")

    # ============ 输出总结 ============
    print("\n" + "="*60)
    print("[Done] 图表生成完成")
    print("="*60)
    
    existing_files = [p for p in generated_files if p.exists()]
    
    if existing_files:
        print("\n已生成的图表文件：")
        for p in existing_files:
            print(f" ✅ {p}")
    else:
        print("\n⚠️ 没有生成任何图表文件，请检查数据是否存在。")


if __name__ == "__main__":
    main()
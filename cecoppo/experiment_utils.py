from __future__ import annotations

import copy
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from cecoppo.config import TrainConfig
from cecoppo.env_cec_dag import CloudEdgeDagEnv
from cecoppo.ppo_agent import PPOAgent
from cecoppo.baselines import (
    FCFSPolicy,
    LeastLoadPolicy,
    GreedyPolicy,
    HEFTPolicy,
)
from cecoppo.utils import (
    set_seed,
    tri_objective_scalar,
    tri_objective_weighted_scalar,
    tri_refs_from_env_config,
    resolve_tri_refs,
    effective_makespan_for_tri,
)

def _set_ieee_style() -> None:
    """统一设置IEEE论文绘图风格（兼容Linux无Times New Roman环境）"""
    plt.rcParams.update({
        # 字体：优先使用衬线字体，Linux上回退到DejaVu Serif
        'font.family':        'sans-serif',
        'font.sans-serif':    ['DejaVu Sans', 'Liberation Sans',
                               'Arial', 'sans-serif'],
        # 字号
        'font.size':          11,
        'axes.labelsize':     12,
        'axes.titlesize':     13,
        'xtick.labelsize':    10,
        'ytick.labelsize':    10,
        'legend.fontsize':    10,
        'figure.titlesize':   14,
        # 图形风格
        'axes.spines.top':    False,   # 去掉上边框
        'axes.spines.right':  False,   # 去掉右边框
        'axes.grid':          True,
        'grid.alpha':         0.3,
        'grid.linestyle':     '--',
        'grid.linewidth':     0.5,
        'lines.linewidth':    2.0,
        'patch.linewidth':    0.8,
        # 保存设置
        'savefig.dpi':        300,
 
        'figure.dpi':         100,
    })


def _set_ieee_transactions_style() -> None:
    """IEEE Transactions 风格：衬线体、四边框、弱网格、适合印刷与灰度稿。

    用于 ``plot_metric_bars``；训练曲线等仍可用 ``_set_ieee_style``。
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Nimbus Roman",
            "Times",
            "Liberation Serif",
            "DejaVu Serif",
            "serif",
        ],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8,
        "axes.labelsize": 7,
        "axes.titlesize": 7.5,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.9,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.55,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": ":",
        "grid.linewidth": 0.45,
        "grid.color": "#888888",
        "axes.axisbelow": True,
        "axes.edgecolor": "#000000",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "savefig.dpi": 300,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _ieee_bar_facecolor(method: str, index: int) -> str:
    """Solid fill color per method (colorblind-friendly palette, IEEE print)."""
    m = str(method)
    if m in METHOD_COLORS:
        return str(METHOD_COLORS[m])
    gray_cycle = ("#c7c7c7", "#adadad", "#949494", "#7c7c7c", "#656565")
    return gray_cycle[index % len(gray_cycle)]
METHOD_ORDER = [
    
    "FCFS",
    "LeastLoad",
    "Greedy",
    "HEFT",

    "MLP-PPO",
    "PPO-StaticGNN",
    "PPO-STGNN",

    "PPO-STGNN (Full)",
    "w/o Temporal",
    "w/o DAG Encoder",
    "w/o BC",
    "w/o Transfer Reward",
    "w/o Balance Reward",
]
# ============================================================
# 统一配色方案（IEEE论文标准）
# ============================================================
METHOD_COLORS = {
    # Traditional baselines: muted / colorblind-safe
    
    "Random":     "#999999",
    "FCFS":       "#56B4E9",
    "LeastLoad":  "#009E73",
    "HEFT":       "#0072B2",
    "Greedy":     "#E69F00",

    # PPO architecture variants
    "MLP-PPO":       "#BDBDBD",
    "PPO-StaticGNN": "#80B1D3",
    "PPO-STGNN":     "#D55E00",

    # Ablation
    "PPO-STGNN (Full)":    "#D55E00",
    "w/o Temporal":        "#0072B2",
    "w/o DAG Encoder":     "#009E73",
    "w/o BC":              "#CC79A7",
    "w/o Transfer Reward": "#E69F00",
    "w/o Balance Reward":  "#666666",
}


METHOD_LINE_STYLES = {
    # ── Baselines ──
  
    "Random":     ":",
    "FCFS":       "--",
    "LeastLoad":  "-.",
    "HEFT":       "-",     # 最强baseline用实线
    "Greedy":     "--",

    # ── Architecture Comparison ──
    "MLP-PPO":       ":",    # 虚线（最弱）
    "PPO-StaticGNN": "--",   # 中划线（中间）
    "PPO-STGNN":     "-",    # 实线（最强，提出方法）

    # ── Ablation Study ──
    "PPO-STGNN (Full)":      "-",    # 实线（完整方法）
    "w/o Temporal":          "--",   # 中划线
    "w/o DAG Encoder":       "-.",   # 点划线
    "w/o BC":                ":",    # 虚线
    "w/o Transfer Reward":   (0, (3, 1, 1, 1)),   # 自定义虚线
    "w/o Balance Reward":    (0, (5, 2, 1, 2)),   # 自定义虚线
}

METHOD_MARKERS = {
    # ── Baselines ──
 
    "Random":     "x",
    "FCFS":       "^",
    "LeastLoad":  "v",
    "HEFT":       "D",    # 菱形（重要baseline）
    "Greedy":     "s",

    # ── Architecture Comparison ──
    "MLP-PPO":       "o",
    "PPO-StaticGNN": "s",
    "PPO-STGNN":     "*",  # 星形（提出方法）

    # ── Ablation Study ──
    "PPO-STGNN (Full)":      "*",   # 星形
    "w/o Temporal":          "s",   # 方形
    "w/o DAG Encoder":       "^",   # 三角
    "w/o BC":                "o",   # 圆形
    "w/o Transfer Reward":   "D",   # 菱形
    "w/o Balance Reward":    "v",   # 倒三角
}


def method_plot_label(method: str, *, xtick_two_line_ours: bool = False) -> str:
    """Figure display name; CSV / internal ``method`` keys stay unchanged.

    When ``xtick_two_line_ours`` is True (baseline / crowded ablation bar charts),
    put ``(Ours)`` on the line below the method name so the x-axis stays readable.
    """
    m = str(method).strip()
    if m == "PPO-STGNN":
        return "PPO-STGNN\n(Ours)" if xtick_two_line_ours else "PPO-STGNN (Ours)"
    if m == "PPO-STGNN (Full)":
        return "PPO-STGNN (Full)\n(Ours)" if xtick_two_line_ours else "PPO-STGNN (Full) (Ours)"
    return m


MAIN_METRICS = [
    ("avg_job_completion_time", "Avg DAG completion time (s)", "lower"),
    ("makespan", "Completed-only makespan (s)", "lower"),
    ("SLR", "Schedule Length Ratio", "lower"),
    ("p95_task_response_time", "P95 task response time (s)", "lower"),
    ("load_balance", "Load balance: L_CPU + L_Mem", "lower"),
    ("resource_utilization", "Eligible resource utilization", "higher"),
    ("avg_energy", "Avg task energy", "lower"),
    ("avg_transfer_cost", "Avg transfer cost (s)", "lower"),
    ("completion_ratio", "Job completion ratio", "higher"),
]

TABLE_METRICS = [
    "method",
    "avg_job_completion_time", "avg_job_completion_time_std",
    "makespan", "makespan_std",
    "SLR", "SLR_std",
    "penalized_makespan", "penalized_makespan_std",
    "p95_task_response_time", "p95_task_response_time_std",
    "load_balance", "load_balance_std", "L_CPU", "L_Mem",
    "cpu_load_variance", "mem_load_variance",
    "resource_utilization", "resource_utilization_std",
    "avg_energy", "avg_energy_std",
    "avg_transfer_cost", "avg_transfer_cost_std",
    "completion_ratio", "completion_ratio_std",

    "cloud_ratio", "edge_ratio", "end_ratio",
    "small_task_ratio", "medium_task_ratio", "large_task_ratio",
    "small_to_end_rate", "medium_to_edge_rate", "large_to_cloud_rate",

    "reward_sum", "reward_mean",
    "invalid_rate", "defer_rate",
]

# ============================================================
# 1. Path and dataset helpers
# ============================================================

def find_data_dir(project_root: Path, user_data_dir: Optional[str] = None) -> Path:
    """Find the new500DAG dataset directory or zip file."""
    if user_data_dir:
        data_dir = Path(user_data_dir).expanduser().resolve()
        if not (data_dir / "nodes.csv").exists():
            raise FileNotFoundError(f"--data-dir={data_dir} 下没有 nodes.csv")
        return data_dir

    candidates = [
        Path.cwd() / "new500DAG",
        Path.cwd() / "new500DAG ",
        project_root / "new500DAG",
        Path("/kaggle/input/new500dag"),
        Path("/kaggle/input/datasets/moonstarry24/500dag-new"),
    ]
    zip_candidates = [
        Path.cwd() / "new500DAG .zip",
        Path.cwd() / "new500DAG.zip",
        Path("/mnt/data/new500DAG .zip"),
        Path("/mnt/data/new500DAG.zip"),
    ]

    for cand in candidates:
        if cand.exists() and (cand / "nodes.csv").exists():
            return cand.resolve()

    for z in zip_candidates:
        if z.exists():
            extract_dir = project_root / "new500DAG"
            if not (extract_dir / "nodes.csv").exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)
            return extract_dir.resolve()

    raise FileNotFoundError(
        "没有找到 new500DAG 数据。请通过 --data-dir 指定目录，或把 new500DAG 文件夹 / new500DAG.zip 放到当前目录。"
    )


def inspect_dataset(data_dir: Path) -> Dict[str, Any]:
    nodes_df = pd.read_csv(data_dir / "nodes.csv")
    dag_stats_df = pd.read_csv(data_dir / "dag_stats.csv")
    with open(data_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    print("\n[Dataset] node roles")
    print(nodes_df["role"].value_counts().to_string())

    print("\n[Dataset] DAG stats")
    cols = [c for c in ["num_tasks", "num_edges", "total_work"] if c in dag_stats_df.columns]
    print(dag_stats_df[cols].describe().to_string())

    print("\n[Dataset] meta")
    print(json.dumps(meta, ensure_ascii=False, indent=2)[:2000])
    return meta


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ============================================================
# 2. Env and evaluation helpers
# ============================================================

def clone_env_cfg(env_cfg: Any) -> Any:
    return copy.deepcopy(env_cfg)


def make_env(data_dir: Path, env_cfg: Any, split: str = "train", seed: int = 42) -> CloudEdgeDagEnv:
    local_cfg = clone_env_cfg(env_cfg)
    local_cfg.split = split
    local_cfg.seed = seed
    return CloudEdgeDagEnv(data_dir, local_cfg)


def select_action(
    policy: Any,
    obs: Dict[str, np.ndarray],
    env: Optional[CloudEdgeDagEnv] = None,
    deterministic: bool = True,
) -> Tuple[int, float, float]:
    """带有非法动作兜底防护的动作选择"""
    try:
        if hasattr(policy, "act"):
            if env is not None:
                action, log_prob, value = policy.act(
                    obs, deterministic=deterministic, env=env
                )
            else:
                action, log_prob, value = policy.act(obs, deterministic=deterministic)
            action = int(action)
        else:
            action = int(policy.select_action(obs, env=env))
            log_prob, value = 0.0, 0.0
            
        # 兜底：如果选出的动作越界或被mask掉，强制选择第一个有效动作
        mask = obs.get("action_mask", np.array([]))
        if len(mask) > 0 and (action < 0 or action >= len(mask) or mask[action] <= 0):
            valid_actions = np.where(mask > 0)[0]
            if len(valid_actions) > 0:
                action = int(valid_actions[0])
            else:
                action = 0
                
        return action, log_prob, value
    except Exception as e:
        print(f"[Warning] Action selection failed: {e}. Fallback to action 0.")
        return 0, 0.0, 0.0

def run_one_episode(
    env: CloudEdgeDagEnv,
    policy: Any,
    deterministic: bool = True,
    max_guard: int = 100000,
) -> Dict[str, float]:
    obs = env.reset()
    done = False

    rewards: List[float] = []
    raw_rewards: List[float] = []
    clip_low_flags: List[float] = []
    clip_high_flags: List[float] = []

    invalid_count = 0
    defer_count = 0
    steps = 0

    role_counts = {"cloud": 0, "edge": 0, "end": 0}
    demand_counts = {"small": 0, "medium": 0, "large": 0}
    demand_role_counts = {
        "small": {"cloud": 0, "edge": 0, "end": 0},
        "medium": {"cloud": 0, "edge": 0, "end": 0},
        "large": {"cloud": 0, "edge": 0, "end": 0},
    }
    while not done and steps < max_guard:
        action, _, _ = select_action(policy, obs, env=env, deterministic=deterministic)
        obs, reward, done, info = env.step(action)

        rewards.append(float(reward))
        raw_rewards.append(float(info.get("raw_reward", reward)))
        clip_low_flags.append(float(info.get("was_clipped_low", False)))
        clip_high_flags.append(float(info.get("was_clipped_high", False)))

        invalid_count += int(bool(info.get("invalid_action", False)))
        defer_count += int(bool(info.get("defer", False)))

        role = str(info.get("target_role", ""))
        if role in role_counts:
            role_counts[role] += 1

        demand = str(info.get("task_demand", ""))

        if demand in demand_counts:
            demand_counts[demand] += 1

            if role in demand_role_counts[demand]:
                demand_role_counts[demand][role] += 1

        steps += 1

    metrics = env.get_episode_metrics()

    metrics["reward_sum"] = float(np.sum(rewards)) if rewards else 0.0
    metrics["reward_mean"] = float(np.mean(rewards)) if rewards else 0.0
    metrics["raw_reward_mean"] = float(np.mean(raw_rewards)) if raw_rewards else metrics["reward_mean"]
    metrics["clip_low_rate"] = float(np.mean(clip_low_flags)) if clip_low_flags else 0.0
    metrics["clip_high_rate"] = float(np.mean(clip_high_flags)) if clip_high_flags else 0.0

    metrics["steps"] = float(steps)
    metrics["invalid_rate"] = float(invalid_count / max(steps, 1))
    metrics["defer_rate"] = float(defer_count / max(steps, 1))

    total_role_actions = max(sum(role_counts.values()), 1)
    metrics["cloud_ratio"] = float(role_counts["cloud"] / total_role_actions)
    metrics["edge_ratio"] = float(role_counts["edge"] / total_role_actions)
    metrics["end_ratio"] = float(role_counts["end"] / total_role_actions)

    small_total = max(demand_counts["small"], 1)
    medium_total = max(demand_counts["medium"], 1)
    large_total = max(demand_counts["large"], 1)

    metrics["small_task_ratio"] = float(demand_counts["small"] / max(steps, 1))
    metrics["medium_task_ratio"] = float(demand_counts["medium"] / max(steps, 1))
    metrics["large_task_ratio"] = float(demand_counts["large"] / max(steps, 1))

    metrics["small_to_end_rate"] = float(
        demand_role_counts["small"]["end"] / small_total
    )
    metrics["medium_to_edge_rate"] = float(
        demand_role_counts["medium"]["edge"] / medium_total
    )
    metrics["large_to_cloud_rate"] = float(
        demand_role_counts["large"]["cloud"] / large_total
    )
        
    return metrics


def evaluate_policy(
    data_dir: Path,
    env_cfg: Any,
    policy: Any,
    split: str = "test",
    episodes: int = 20,
    seed: int = 42,
    deterministic: bool = True,
    desc: Optional[str] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, float]] = []

    # Build the heavy environment once, then reseed/reset it per episode.
    # This avoids repeated CSV/JSON loading and lookup construction.
    env = make_env(data_dir, env_cfg, split=split, seed=seed)
    for ep in tqdm(range(episodes), desc=desc or f"eval-{split}", leave=False):
        ep_seed = seed + ep
        set_seed(ep_seed)
        if hasattr(env, "rng"):
            env.rng.seed(ep_seed)
        rows.append(run_one_episode(env, policy, deterministic=deterministic))

    detail_df = pd.DataFrame(rows)
    summary = detail_df.mean(numeric_only=True).to_dict()
    summary.update(detail_df.std(numeric_only=True).add_suffix("_std").to_dict())
    summary["episodes"] = float(episodes)
    return summary, detail_df


def evaluate_and_save_policy(
    data_dir: Path,
    cfg: TrainConfig,
    policy: Any,
    method_name: str,
    result_dir: Path,
    episodes: int,
    seed: int,
    split: str = "test",
    summary_filename: str = "ppo_results.csv",
    detail_filename: Optional[str] = None,
) -> pd.DataFrame:
    result_dir.mkdir(parents=True, exist_ok=True)
    summary, detail = evaluate_policy(
        data_dir=data_dir,
        env_cfg=cfg.env,
        policy=policy,
        split=split,
        episodes=episodes,
        seed=seed,
        deterministic=True,
        desc=f"Evaluate {method_name}",
    )
    summary["method"] = method_name
    result_df = pd.DataFrame([summary])
    result_df.to_csv(result_dir / summary_filename, index=False)

    if detail_filename is None:
        safe = method_name.replace("/", "_").replace(" ", "_")
        detail_filename = f"detail_{safe}.csv"
    detail.to_csv(result_dir / detail_filename, index=False)
    return result_df


def sort_by_method(df: pd.DataFrame, order: Optional[List[str]] = None) -> pd.DataFrame:
    out = df.copy()
    order = order or METHOD_ORDER
    if "method" in out.columns:
        # pandas.Categorical 要求 categories 必须唯一。
        # 这里先按出现顺序去重，同时把结果中存在但不在 METHOD_ORDER 里的方法追加到末尾，
        # 避免未知 method 被转成 NaN。
        methods = out["method"].astype(str)
        unique_order = list(dict.fromkeys(str(x) for x in order))
        full_order = unique_order + [m for m in methods.unique() if m not in unique_order]

        out["method"] = pd.Categorical(methods, categories=full_order, ordered=True)
        out = out.sort_values("method").reset_index(drop=True)
        out["method"] = out["method"].astype(str)
    return out


def print_metrics_table(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    cols = [c for c in TABLE_METRICS if c in df.columns]
    out = sort_by_method(df) if "method" in df.columns else df.copy()
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(out[cols].to_string(index=False))


# ============================================================
# 3. PPO training helpers
# ============================================================

def stack_obs_for_torch(obs_list: List[Dict[str, np.ndarray]], device: str) -> Dict[str, torch.Tensor]:
    return {
        k: torch.tensor(np.stack([o[k] for o in obs_list]), dtype=torch.float32, device=device)
        for k in obs_list[0].keys()
    }


def collect_teacher_dataset(
    data_dir: Path,
    env_cfg: Any,
    teacher_policy: Any,
    max_samples: int = 6000,
    seed: int = 42,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    env = make_env(data_dir, env_cfg, split="train", seed=seed)
    obs = env.reset()
    obs_list: List[Dict[str, np.ndarray]] = []
    action_list: List[int] = []

    for _ in tqdm(range(max_samples), desc="Collect HEFT teacher data", leave=False):
        action = teacher_policy.select_action(obs, env=env)
        if action < 0 or action >= len(obs["action_mask"]) or obs["action_mask"][action] <= 0:
            valid = np.where(obs["action_mask"] > 0)[0]
            action = int(valid[0]) if len(valid) else 0

        obs_list.append(obs)
        action_list.append(int(action))
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    return obs_list, np.array(action_list, dtype=np.int64)


def collect_multi_teacher_dataset(
    data_dir: Path,
    env_cfg: Any,
    max_samples: int = 6000,
    seed: int = 42,
    teacher_fracs: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """Collect behavior-cloning data from several strong heuristic teachers.

    Default ~82% HEFT / ~9% Greedy / ~9% LeastLoad. Pass ``teacher_fracs`` as
    ``(heft, greedy, least_load)`` summing to 1.0 (e.g. ST-GNN 架构对比用更多
    LeastLoad 标签以缓解 BC 冷启动偏 makespan/SLR)。
    """
    teachers = [
        ("HEFT", HEFTPolicy()),
        ("Greedy", GreedyPolicy()),
        ("LeastLoad", LeastLoadPolicy()),
    ]

    if teacher_fracs is None:
        fracs = (0.82, 0.09, 0.09)
    else:
        fracs = tuple(float(x) for x in teacher_fracs)
        total = sum(fracs)
        if total <= 0:
            raise ValueError("teacher_fracs must sum to a positive value")
        fracs = tuple(f / total for f in fracs)

    obs_list: List[Dict[str, np.ndarray]] = []
    action_list: List[int] = []

    n_heft = int(round(max_samples * fracs[0]))
    n_greedy = int(round(max_samples * fracs[1]))
    n_ll = max_samples - n_heft - n_greedy
    counts = [max(0, n_heft), max(0, n_greedy), max(0, n_ll)]

    for i, (teacher_name, teacher_policy) in enumerate(teachers):
        n_samples = counts[i]
        if n_samples <= 0:
            continue

        env = make_env(data_dir, env_cfg, split="train", seed=seed + i * 1000)
        obs = env.reset()

        for _ in tqdm(range(n_samples), desc=f"Collect {teacher_name} teacher data", leave=False):
            action = teacher_policy.select_action(obs, env=env)
            if action < 0 or action >= len(obs["action_mask"]) or obs["action_mask"][action] <= 0:
                valid = np.where(obs["action_mask"] > 0)[0]
                action = int(valid[0]) if len(valid) else 0

            obs_list.append(obs)
            action_list.append(int(action))
            obs, _, done, _ = env.step(action)
            if done:
                obs = env.reset()

    return obs_list, np.array(action_list, dtype=np.int64)


def collect_single_teacher_dataset(
    data_dir: Path,
    env_cfg: Any,
    teacher_name: str,
    max_samples: int = 1500,
    seed: int = 42,
) -> Tuple[List[Dict[str, np.ndarray]], np.ndarray]:
    """Collect BC data from one heuristic teacher (e.g. LeastLoad for load balance)."""
    name = teacher_name.strip().lower()
    if name in {"leastload", "least_load", "ll"}:
        policy = LeastLoadPolicy()
        label = "LeastLoad"
    elif name in {"heft"}:
        policy = HEFTPolicy()
        label = "HEFT"
    elif name in {"greedy"}:
        policy = GreedyPolicy()
        label = "Greedy"
    else:
        raise ValueError(f"Unknown teacher: {teacher_name}")

    obs_list: List[Dict[str, np.ndarray]] = []
    action_list: List[int] = []
    env = make_env(data_dir, env_cfg, split="train", seed=seed)
    obs = env.reset()
    for _ in tqdm(range(max_samples), desc=f"Collect {label} teacher data", leave=False):
        action = policy.select_action(obs, env=env)
        if action < 0 or action >= len(obs["action_mask"]) or obs["action_mask"][action] <= 0:
            valid = np.where(obs["action_mask"] > 0)[0]
            action = int(valid[0]) if len(valid) else 0
        obs_list.append(obs)
        action_list.append(int(action))
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()
    return obs_list, np.array(action_list, dtype=np.int64)


def behavior_clone_pretrain(
    agent: PPOAgent,
    obs_list: List[Dict[str, np.ndarray]],
    action_array: np.ndarray,
    device: str,
    epochs: int = 6,
    batch_size: int = 64,
    lr: float = 1e-4,
) -> None:
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    n = len(obs_list)
    indices = np.arange(n)

    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        losses: List[float] = []
        accs: List[float] = []
        for start in tqdm(range(0, n, batch_size), desc=f"BC pretrain {ep}/{epochs}", leave=False):
            mb_idx = indices[start:start + batch_size]
            batch = stack_obs_for_torch([obs_list[i] for i in mb_idx], device)
            target = torch.tensor(action_array[mb_idx], dtype=torch.long, device=device)

            logits, _ = agent.model(batch)
            logits = logits.masked_fill(batch["action_mask"] <= 0, -1e9)
            loss = torch.nn.functional.cross_entropy(logits, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == target).float().mean().item()
            losses.append(float(loss.item()))
            accs.append(float(acc))

        print(f"[BC {ep:02d}] loss={np.mean(losses):.4f}, teacher_acc={np.mean(accs):.3f}")


def bc_anchor_update(
    agent: PPOAgent,
    teacher_obs: List[Dict[str, np.ndarray]],
    teacher_actions: np.ndarray,
    device: str,
    batch_size: int = 64,
    n_batches: int = 4,
    lr: float = 2e-5,
) -> Dict[str, float]:
    if not teacher_obs:
        return {"bc_anchor_loss": 0.0, "bc_anchor_acc": 0.0}

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    n = len(teacher_obs)
    losses: List[float] = []
    accs: List[float] = []

    for _ in range(n_batches):
        mb_idx = np.random.choice(n, size=min(batch_size, n), replace=False)
        batch = stack_obs_for_torch([teacher_obs[i] for i in mb_idx], device)
        target = torch.tensor(teacher_actions[mb_idx], dtype=torch.long, device=device)

        logits, _ = agent.model(batch)
        logits = logits.masked_fill(batch["action_mask"] <= 0, -1e9)
        loss = torch.nn.functional.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=-1)
            acc = (pred == target).float().mean().item()
        losses.append(float(loss.item()))
        accs.append(float(acc))

    return {"bc_anchor_loss": float(np.mean(losses)), "bc_anchor_acc": float(np.mean(accs))}


def validation_score(
    summary: Dict[str, float],
    slot_size: float,
    *,
    tri_w_m: Optional[float] = None,
    tri_w_s: Optional[float] = None,
    tri_w_lb: Optional[float] = None,
    makespan_max_blend: float = 0.0,
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
) -> float:
    """Selection score for validation. Lower is better.

    Uses the same tri-objective normalization as terminal_sparse training rewards
    (makespan seconds, SLR, paper load_balance). Optional weights emphasize
    makespan/SLR when set (e.g. ``reward_terminal_tri_w_*`` on ``EnvConfig``).
    ``slot_size`` is kept for API compatibility.

    When ``makespan_max_blend`` in (0,1], makespan axis uses a convex combination
    of mean DAG makespan and ``max_dag_makespan`` (same as training terminal tri).
    """
    del slot_size  # metrics are in absolute paper units; refs live in utils.py
    mm = float(summary.get("makespan", summary.get("raw_makespan", 1e9)))
    mx = float(summary.get("max_dag_makespan", mm))
    makespan_sec = effective_makespan_for_tri(mm, mx, makespan_max_blend)
    slr_raw = max(0.0, float(summary.get("SLR", summary.get("slr", 1e9))))
    load_raw = max(0.0, float(summary.get("load_balance", 10.0)))
    completion_gap = 1.0 - float(summary.get("completion_ratio", 0.0))
    invalid = float(summary.get("invalid_rate", 0.0))

    if tri_w_m is None and tri_w_s is None and tri_w_lb is None:
        core = tri_objective_scalar(
            makespan_sec,
            slr_raw,
            load_raw,
            ref_makespan_sec=ref_makespan_sec,
            ref_slr=ref_slr,
            ref_load_balance=ref_load_balance,
        )
    else:
        wm = float(tri_w_m) if tri_w_m is not None else 0.27
        ws = float(tri_w_s) if tri_w_s is not None else 0.58
        wlb = float(tri_w_lb) if tri_w_lb is not None else 0.15
        core = tri_objective_weighted_scalar(
            makespan_sec,
            slr_raw,
            load_raw,
            wm,
            ws,
            wlb,
            ref_makespan_sec=ref_makespan_sec,
            ref_slr=ref_slr,
            ref_load_balance=ref_load_balance,
        )
    return float(core + 2.00 * completion_gap + 0.10 * invalid)


def validation_score_stgnn(
    summary: Dict[str, float],
    *,
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
    makespan_max_blend: float = 0.0,
) -> float:
    """STGNN 架构对比选模：三轴均衡 + 超参考惩罚，避免只压低 load 却选到极差 mks/SLR ckpt。"""
    mm = float(summary.get("makespan", summary.get("raw_makespan", 1e9)))
    mx = float(summary.get("max_dag_makespan", mm))
    makespan_sec = effective_makespan_for_tri(mm, mx, makespan_max_blend)
    slr_raw = max(0.0, float(summary.get("SLR", summary.get("slr", 1e9))))
    load_raw = max(0.0, float(summary.get("load_balance", 10.0)))
    rm, rs, rl = resolve_tri_refs(ref_makespan_sec, ref_slr, ref_load_balance)
    m_n = makespan_sec / max(rm, 1e-6)
    s_n = slr_raw / max(rs, 1e-6)
    lb_n = load_raw / max(rl, 1e-6)
    mean_tri = float((m_n + s_n + lb_n) / 3.0)
    cheby = float(max(m_n, s_n, lb_n))
    over_ref = float(max(0.0, m_n - 1.0) ** 2 + max(0.0, s_n - 1.0) ** 2 + max(0.0, lb_n - 1.0) ** 2)
    completion_gap = 1.0 - float(summary.get("completion_ratio", 0.0))
    invalid = float(summary.get("invalid_rate", 0.0))
    return float(0.40 * mean_tri + 0.45 * cheby + 0.55 * over_ref + 2.0 * completion_gap + 0.10 * invalid)


def validation_tri_sum_normalized(
    summary: Dict[str, float],
    *,
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
    w_m: float = 0.34,
    w_s: float = 0.33,
    w_lb: float = 0.33,
) -> float:
    """验证三指标归一化加权和（越低越好），用于 STGNN 部署 checkpoint。"""
    mm = float(summary.get("makespan", summary.get("raw_makespan", 1e9)))
    slr_raw = max(0.0, float(summary.get("SLR", summary.get("slr", 1e9))))
    load_raw = max(0.0, float(summary.get("load_balance", 10.0)))
    rm, rs, rl = resolve_tri_refs(ref_makespan_sec, ref_slr, ref_load_balance)
    return float(
        w_m * (mm / max(rm, 1e-6))
        + w_s * (slr_raw / max(rs, 1e-6))
        + w_lb * (load_raw / max(rl, 1e-6))
    )


def apply_makespan_slr_focus_preset(cfg: TrainConfig) -> None:
    """Bias tri-objective toward makespan + SLR and strengthen CPU disturbance (training-time).

    Call after ``build_default_config``. Intended for ``--focus-mks-slr`` in ``train_ppo.py``.
    """
    cfg.env.reward_terminal_tri_w_makespan = 0.26
    cfg.env.reward_terminal_tri_w_slr = 0.62
    cfg.env.reward_terminal_tri_w_load = 0.12
    cfg.env.reward_sparse_per_dag_slr_weight = max(
        float(getattr(cfg.env, "reward_sparse_per_dag_slr_weight", 0.0)),
        0.098,
    )
    cfg.env.reward_sparse_per_dag_makespan_weight = max(
        float(getattr(cfg.env, "reward_sparse_per_dag_makespan_weight", 0.0)),
        0.048,
    )
    cfg.env.reward_sparse_step_balance_weight = max(
        float(getattr(cfg.env, "reward_sparse_step_balance_weight", 0.0)),
        0.055,
    )
    cfg.env.reward_sparse_per_dag_load_weight = max(
        float(getattr(cfg.env, "reward_sparse_per_dag_load_weight", 0.0)),
        0.042,
    )
    cfg.env.reward_terminal_tri_penalty = max(float(cfg.env.reward_terminal_tri_penalty), 3.92)
    cfg.env.dynamic_cpu_amp = max(float(cfg.env.dynamic_cpu_amp), 0.34)
    cfg.env.enable_dynamic_disturbance = True


def apply_ablation_experiment_reward_preset(cfg: TrainConfig) -> None:
    """Reward / tri-objective preset used **only** by ``run_ablations.py``.

    Intentionally **differs** from ``build_default_config`` (``train_ppo.py`` baseline
    vs heuristics) so ablation & architecture runs do not reuse the same scalarization
    and sparse shaping as the main ``PPO-STGNN`` checkpoint. Checkpoints are stored
    under a separate directory (see ``run_ablations --checkpoint-subdir``).

    Does **not** set ``reward_transfer_weight`` / ``reward_balance_weight`` so
    ``make_variant_config`` can still apply ablation-specific overrides afterward.
    """
    cfg.env.reward_terminal_tri_w_makespan = 0.34
    cfg.env.reward_terminal_tri_w_slr = 0.46
    cfg.env.reward_terminal_tri_w_load = 0.20
    cfg.env.reward_terminal_tri_penalty = 3.55
    cfg.env.reward_sparse_per_dag_slr_weight = 0.055
    cfg.env.reward_sparse_per_dag_makespan_weight = 0.032
    cfg.env.reward_sparse_step_balance_weight = 0.052
    cfg.env.reward_sparse_per_dag_load_weight = 0.038


# 统一 BC：轻量冷启动，避免过拟合 HEFT/LeastLoad 标签从而拖垮 PPO 阶段（尤其 load）。
UNIFIED_BC_TEACHER_FRACS: Tuple[float, float, float] = (0.82, 0.09, 0.09)
UNIFIED_BC_DEFAULT_EPOCHS: int = 2
UNIFIED_BC_DEFAULT_LR: float = 8e-5
UNIFIED_BC_DEFAULT_SAMPLES: int = 5000


def architecture_results_tag(arch_bc_mode: str) -> str:
    """架构对比结果 CSV / 训练曲线文件名后缀。"""
    mode = str(arch_bc_mode or "proposed").strip().lower()
    if mode == "unified":
        return "_arch_ubc"
    if mode == "unified_fair":
        return "_arch_ubc_fair"
    if mode == "unified_proposed":
        return "_arch_ubc"
    if mode == "none":
        return "_arch_nobc"
    if mode == "proposed":
        return "_arch"
    raise ValueError(f"Unknown arch_bc_mode={arch_bc_mode!r}")


def apply_stgnn_agent_eval_options(agent: PPOAgent, env_cfg: Any) -> None:
    """Sync STGNN deterministic eval spread-rerank settings onto a loaded agent."""
    if getattr(agent, "encoder_type", "") != "stgnn":
        return
    agent._stgnn_eval_spread_rerank = bool(
        getattr(env_cfg, "stgnn_eval_spread_rerank", False)
    )
    agent._stgnn_eval_spread_rerank_top_k = int(
        getattr(env_cfg, "stgnn_eval_spread_rerank_top_k", 48)
    )


def apply_architecture_static_stabilizer(cfg: TrainConfig) -> None:
    """架构对比 **仅 PPO-StaticGNN**：抑制 PPO 后期向「边层摊销」塌缩导致 mks/SLR 尖峰。

    StaticGNN 无资源时序注意力，在 dynamic_cpu_amp 扰动下策略方差大；略降 lr/熵与逐步
    balance 权重，避免验证曲线在 ~60 epoch 后出现 makespan/SLR 暴涨、load 骤降的剪刀差。
    """
    cfg.ppo.lr = min(float(cfg.ppo.lr), 1.4e-4)
    cfg.ppo.entropy_coef = min(float(cfg.ppo.entropy_coef), 0.007)
    cfg.ppo.target_kl = max(float(getattr(cfg.ppo, "target_kl", 0.0)), 0.022)
    cfg.env.reward_balance_weight = min(
        float(getattr(cfg.env, "reward_balance_weight", 0.16)), 0.11
    )
    cfg.env.reward_sparse_step_balance_weight = min(
        float(getattr(cfg.env, "reward_sparse_step_balance_weight", 0.052)), 0.038
    )
    cfg.env.reward_sparse_per_dag_load_weight = min(
        float(getattr(cfg.env, "reward_sparse_per_dag_load_weight", 0.038)), 0.028
    )


def apply_architecture_stgnn_balance_focus(cfg: TrainConfig) -> None:
    """架构对比 **仅 PPO-STGNN**：三指标均衡（非 LeastLoad 主导 BC）。

    时序 + 边层 logit + 分阶段 load 塑形 + 验证 spread-rerank；mks/SLR 权重略高以防末期塌缩。
    """
    cfg.env.reward_balance_weight = max(
        float(getattr(cfg.env, "reward_balance_weight", 0.0)), 0.22
    )
    cfg.env.reward_terminal_tri_w_makespan = 0.38
    cfg.env.reward_terminal_tri_w_slr = 0.40
    cfg.env.reward_terminal_tri_w_load = 0.22
    cfg.env.reward_terminal_tri_penalty = max(
        float(getattr(cfg.env, "reward_terminal_tri_penalty", 0.0)), 4.05
    )
    cfg.env.reward_sparse_per_dag_slr_weight = max(
        float(getattr(cfg.env, "reward_sparse_per_dag_slr_weight", 0.0)), 0.048
    )
    cfg.env.reward_sparse_per_dag_makespan_weight = max(
        float(getattr(cfg.env, "reward_sparse_per_dag_makespan_weight", 0.0)), 0.036
    )
    cfg.env.reward_sparse_step_balance_weight = max(
        float(getattr(cfg.env, "reward_sparse_step_balance_weight", 0.0)), 0.14
    )
    cfg.env.reward_sparse_per_dag_load_weight = max(
        float(getattr(cfg.env, "reward_sparse_per_dag_load_weight", 0.0)), 0.44
    )
    cfg.env.reward_sparse_busy_entropy_weight = max(
        float(getattr(cfg.env, "reward_sparse_busy_entropy_weight", 0.0)), 0.09
    )
    cfg.env.tri_ref_load_balance = 55.0
    cfg.env.tri_ref_slr = 8.2
    cfg.env.tri_ref_makespan_sec = 0.0
    cfg.env.stgnn_leastload_logit_blend = 0.12
    cfg.env.stgnn_edge_logit_bonus = 0.34
    cfg.env.stgnn_end_logit_penalty = 0.12
    cfg.env.stgnn_eval_spread_rerank = True
    cfg.env.stgnn_eval_spread_rerank_top_k = 72
    cfg.ppo.target_kl = max(float(getattr(cfg.ppo, "target_kl", 0.0)), 0.028)


def train_ppo_agent(
    data_dir: Path,
    cfg: TrainConfig,
    result_dir: Path,
    seed: int = 42,
    run_name: str = "PPO-STGNN",
    encoder_type: str = "stgnn",
    use_bc: bool = True,
    teacher_samples: int = 6000,
    bc_epochs: int = 3,
    eval_episodes: int = 8,
    val_eval_episodes: Optional[int] = None,
    val_every: int = 5,
    early_stopping: bool = True,
    bc_teacher_fracs: Optional[Tuple[float, float, float]] = None,
    balance_bc_samples: int = 0,
    balance_bc_epochs: int = 2,
    stgnn_balance_first_bc: bool = False,
    heft_refine_samples: int = 0,
    heft_refine_epochs: int = 0,
    bc_pretrain_lr: float = 1e-4,
    results_tag: str = "",
    architecture_fair: bool = False,
) -> Tuple[PPOAgent, pd.DataFrame, Path]:
    """Train PPO and validate only every val_every epochs.

    Validation is run when epoch % val_every == 0 and at the final epoch. For
    skipped epochs, validation columns are NaN and val_ran is 0.

    When ``early_stopping`` is False, training always runs for ``cfg.ppo.epochs``
    so different runs share the same epoch axis (``run_ablations.py`` defaults
    to disabling early stopping).
    """
    set_seed(seed)
    val_every = max(1, int(val_every))
    val_eps = int(val_eval_episodes) if val_eval_episodes is not None else int(eval_episodes)
    val_eps = max(1, val_eps)
    result_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_env = make_env(data_dir, cfg.env, split="train", seed=seed)
    # ``make_env`` deep-copies ``cfg.env``；PPO rollout 只读 ``train_env.config``。
    train_env_cfg = train_env.config
    sample_obs = train_env.reset()
    agent = PPOAgent(
        sample_obs=sample_obs,
        action_dim=train_env.action_dim,
        hidden_dim=cfg.ppo.hidden_dim,
        config=cfg.ppo,
        device=cfg.device,
        encoder_type=encoder_type,
    )
    if encoder_type == "stgnn":
        apply_stgnn_agent_eval_options(agent, cfg.env)

    safe_name = run_name.replace("/", "_").replace(" ", "_")
    tag = str(results_tag or "")
    best_path = Path(cfg.checkpoint_dir) / f"best_{safe_name}{tag}.pt"
    best_tri_path = Path(cfg.checkpoint_dir) / f"best_{safe_name}_tri{tag}.pt"
    latest_path = Path(cfg.checkpoint_dir) / f"latest_{safe_name}{tag}.pt"

    history: List[Dict[str, float]] = []
    best_score = float("inf")
    best_tri_sum = float("inf")
    best_epoch = 0
    best_tri_epoch = 0
    # 公平架构对比：仅 encoder 不同，禁用 STGNN 专属奖励课程 / 选模 / 熵调度
    stgnn_arch_mode = (
        run_name == "PPO-STGNN"
        and encoder_type == "stgnn"
        and not bool(architecture_fair)
    )
    patience_validations = 18 if stgnn_arch_mode else 14
    min_train_epochs = max(28, cfg.ppo.epochs // 4)
    min_delta = 5e-4
    
    no_improve_validations = 0
    bc_lr = float(bc_pretrain_lr)
    base_step_bal = float(getattr(train_env_cfg, "reward_sparse_step_balance_weight", 0.0))
    base_dag_load = float(getattr(train_env_cfg, "reward_sparse_per_dag_load_weight", 0.0))
    base_ent = float(getattr(train_env_cfg, "reward_sparse_busy_entropy_weight", 0.0))

    if use_bc:
        if stgnn_balance_first_bc:
            ll_n = int(balance_bc_samples) if balance_bc_samples > 0 else int(teacher_samples)
            ll_ep = int(balance_bc_epochs) if balance_bc_epochs > 0 else int(bc_epochs)
            ll_obs, ll_actions = collect_single_teacher_dataset(
                data_dir=data_dir,
                env_cfg=cfg.env,
                teacher_name="LeastLoad",
                max_samples=ll_n,
                seed=seed,
            )
            behavior_clone_pretrain(
                agent=agent,
                obs_list=ll_obs,
                action_array=ll_actions,
                device=cfg.device,
                epochs=ll_ep,
                batch_size=64,
                lr=bc_lr,
            )
            print(
                f"[{run_name}] Balance-first BC (deprecated path): LeastLoad "
                f"{ll_n} samples, {ll_ep} epoch(s), lr={bc_lr:g}"
            )
        else:
            teacher_obs, teacher_actions = collect_multi_teacher_dataset(
                data_dir=data_dir,
                env_cfg=cfg.env,
                max_samples=teacher_samples,
                seed=seed,
                teacher_fracs=bc_teacher_fracs,
            )
            fracs_msg = (
                f"{bc_teacher_fracs}"
                if bc_teacher_fracs is not None
                else "default 82/9/9"
            )
            behavior_clone_pretrain(
                agent=agent,
                obs_list=teacher_obs,
                action_array=teacher_actions,
                device=cfg.device,
                epochs=bc_epochs,
                batch_size=64,
                lr=bc_lr,
            )
            print(
                f"[{run_name}] Multi-teacher BC: {teacher_samples} samples, "
                f"{bc_epochs} epoch(s), fracs={fracs_msg}, lr={bc_lr:g}"
            )
            if balance_bc_samples > 0 and balance_bc_epochs > 0:
                print(
                    f"[{run_name}] Warning: extra LeastLoad BC is disabled in architecture "
                    "comparison; set balance_bc_samples=0 to silence this."
                )
    for epoch in tqdm(range(1, cfg.ppo.epochs + 1), desc=f"Train {run_name}"):
        if stgnn_arch_mode:
            prog = float(epoch) / max(float(cfg.ppo.epochs), 1.0)
            # 前期保持 HEFT-like mks/SLR，中后期再加强 load 塑形，避免 load 与 mks 剪刀差
            if prog < 0.38:
                load_mult = 0.42 + 0.58 * (prog / 0.38)
                ent_mult = 0.50 + 0.50 * (prog / 0.38)
            else:
                tail = (prog - 0.38) / max(1.0 - 0.38, 1e-6)
                load_mult = 1.0 + 0.95 * tail
                ent_mult = 1.0 + 0.75 * tail
            load_w = base_dag_load * load_mult
            ent_w = base_ent * ent_mult
            train_env_cfg.reward_sparse_per_dag_load_weight = load_w
            train_env_cfg.reward_sparse_busy_entropy_weight = ent_w
            cfg.env.reward_sparse_per_dag_load_weight = load_w
            cfg.env.reward_sparse_busy_entropy_weight = ent_w

        obs = train_env.reset()

        rewards: List[float] = []
        raw_rewards: List[float] = []
        clip_low_flags: List[float] = []
        clip_high_flags: List[float] = []

        resp_penalties: List[float] = []
        queue_penalties: List[float] = []
        transfer_penalties: List[float] = []
        balance_penalties: List[float] = []
        energy_penalties: List[float] = []
        makespan_penalties: List[float] = []

        invalid_flags: List[float] = []
        defer_flags: List[float] = []
        role_counts = {"cloud": 0, "edge": 0, "end": 0}
        demand_counts = {"small": 0, "medium": 0, "large": 0}
        demand_role_counts = {
            "small": {"cloud": 0, "edge": 0, "end": 0},
            "medium": {"cloud": 0, "edge": 0, "end": 0},
            "large": {"cloud": 0, "edge": 0, "end": 0},
        }
        last_done = False

        rollout_bar = tqdm(
            range(cfg.ppo.steps_per_epoch),
            desc=f"{run_name} epoch {epoch:03d} rollout",
            leave=False,
        )

        for _ in rollout_bar:
            action, log_prob, value = agent.act(obs, deterministic=False)
            next_obs, reward, done, info = train_env.step(action)

            agent.store(obs, action, reward, done, log_prob, value)

            rewards.append(float(reward))
            raw_rewards.append(float(info.get("raw_reward", reward)))
            clip_low_flags.append(float(info.get("was_clipped_low", False)))
            clip_high_flags.append(float(info.get("was_clipped_high", False)))

            resp_penalties.append(float(info.get("resp_penalty", 0.0)))
            queue_penalties.append(float(info.get("queue_penalty", 0.0)))
            transfer_penalties.append(float(info.get("transfer_penalty", 0.0)))
            balance_penalties.append(float(info.get("balance_penalty", 0.0)))
            energy_penalties.append(float(info.get("energy_penalty", 0.0)))
            makespan_penalties.append(float(info.get("makespan_penalty", 0.0)))

            invalid_flags.append(float(info.get("invalid_action", False)))
            defer_flags.append(float(info.get("defer", False)))

            role = str(info.get("target_role", ""))
            if role in role_counts:
                role_counts[role] += 1

            demand = str(info.get("task_demand", ""))
            if demand in demand_counts:
                demand_counts[demand] += 1

                if role in demand_role_counts[demand]:
                    demand_role_counts[demand][role] += 1

            obs = next_obs
            last_done = done

            if done:
                obs = train_env.reset()

        last_value = 0.0 if last_done else agent.get_value(obs)
        if epoch <= 38:
            agent.entropy_coef = 0.0023
        elif epoch <= 70:
            agent.entropy_coef = 0.0011
        else:
            # 后期熵过低时策略易过早收敛到次优调度，验证 makespan/SLR 会反弹
            agent.entropy_coef = 0.0009
        if stgnn_arch_mode:
            if epoch > 50:
                agent.entropy_coef = max(float(agent.entropy_coef), 0.0012)
            if epoch > 95:
                agent.entropy_coef = max(float(agent.entropy_coef), 0.0015)
                for pg in agent.optimizer.param_groups:
                    pg["lr"] = min(float(pg["lr"]), 4.5e-5)
        elif run_name == "PPO-STGNN" and epoch > 55:
            agent.entropy_coef = max(float(agent.entropy_coef), 0.00135)
        cfg.ppo.entropy_coef = agent.entropy_coef
        
        losses = agent.update(last_value=last_value)

        # Do not run BC anchor updates during PPO. The BC stage is only a cold
        # start; continuing to optimize cross-entropy against heuristic labels
        # can prevent policy improvement on the actual reward.
        anchor_info = {"bc_anchor_loss": 0.0, "bc_anchor_acc": 0.0}

        total_role_actions = max(sum(role_counts.values()), 1)
        small_total = max(demand_counts["small"], 1)
        medium_total = max(demand_counts["medium"], 1)
        large_total = max(demand_counts["large"], 1)

        small_to_end_rate = float(
            demand_role_counts["small"]["end"] / small_total
        )
        medium_to_edge_rate = float(
            demand_role_counts["medium"]["edge"] / medium_total
        )
        large_to_cloud_rate = float(
            demand_role_counts["large"]["cloud"] / large_total
        )
        avg_reward = float(np.mean(rewards)) if rewards else 0.0
        avg_raw_reward = float(np.mean(raw_rewards)) if raw_rewards else avg_reward
        clip_low_rate = float(np.mean(clip_low_flags)) if clip_low_flags else 0.0
        clip_high_rate = float(np.mean(clip_high_flags)) if clip_high_flags else 0.0

        row: Dict[str, float] = {
            "run_name": run_name,
            "epoch": float(epoch),

            "train_reward_mean": avg_reward,
            "train_raw_reward_mean": avg_raw_reward,
            "train_clip_low_rate": clip_low_rate,
            "train_clip_high_rate": clip_high_rate,

            "train_resp_penalty": float(np.mean(resp_penalties)) if resp_penalties else 0.0,
            "train_queue_penalty": float(np.mean(queue_penalties)) if queue_penalties else 0.0,
            "train_transfer_penalty": float(np.mean(transfer_penalties)) if transfer_penalties else 0.0,
            "train_balance_penalty": float(np.mean(balance_penalties)) if balance_penalties else 0.0,
            "train_energy_penalty": float(np.mean(energy_penalties)) if energy_penalties else 0.0,
            "train_makespan_penalty": float(np.mean(makespan_penalties)) if makespan_penalties else 0.0,

            "train_invalid_rate": float(np.mean(invalid_flags)) if invalid_flags else 0.0,
            "train_defer_rate": float(np.mean(defer_flags)) if defer_flags else 0.0,

            "train_cloud_ratio": float(role_counts["cloud"] / total_role_actions),
            "train_edge_ratio": float(role_counts["edge"] / total_role_actions),
            "train_end_ratio": float(role_counts["end"] / total_role_actions),

            "train_small_task_ratio": float(demand_counts["small"] / max(len(rewards), 1)),
            "train_medium_task_ratio": float(demand_counts["medium"] / max(len(rewards), 1)),
            "train_large_task_ratio": float(demand_counts["large"] / max(len(rewards), 1)),

            "train_small_to_end_rate": small_to_end_rate,
            "train_medium_to_edge_rate": medium_to_edge_rate,
            "train_large_to_cloud_rate": large_to_cloud_rate,

            "policy_loss": float(losses.get("policy_loss", 0.0)),
            "value_loss": float(losses.get("value_loss", 0.0)),
            "entropy": float(losses.get("entropy", 0.0)),
            "approx_kl": float(losses.get("approx_kl", 0.0)),
            "clip_frac": float(losses.get("clip_frac", 0.0)),
            "grad_norm": float(losses.get("grad_norm", 0.0)),
            "bc_anchor_loss": float(anchor_info["bc_anchor_loss"]),
            "bc_anchor_acc": float(anchor_info["bc_anchor_acc"]),

            "val_ran": 0.0,
            "val_score": np.nan,
        }

        should_validate = (epoch % val_every == 0) or (epoch == cfg.ppo.epochs)
        if should_validate:
            val_summary, _ = evaluate_policy(
                data_dir,
                cfg.env,
                agent,
                split="val",
                episodes=val_eps,
                seed=2024,
                deterministic=True,
                desc=f"Val {run_name} epoch {epoch}",
            )
            ref_m, ref_s, ref_lb = tri_refs_from_env_config(cfg.env)
            blend = float(getattr(cfg.env, "reward_terminal_tri_makespan_max_blend", 0.0))
            if stgnn_arch_mode:
                score = validation_score_stgnn(
                    val_summary,
                    ref_makespan_sec=ref_m,
                    ref_slr=ref_s,
                    ref_load_balance=ref_lb,
                    makespan_max_blend=blend,
                )
            else:
                score = validation_score(
                    val_summary,
                    slot_size=float(cfg.env.slot_size),
                    tri_w_m=float(getattr(cfg.env, "reward_terminal_tri_w_makespan", 0.27)),
                    tri_w_s=float(getattr(cfg.env, "reward_terminal_tri_w_slr", 0.58)),
                    tri_w_lb=float(getattr(cfg.env, "reward_terminal_tri_w_load", 0.15)),
                    makespan_max_blend=blend,
                    ref_makespan_sec=ref_m,
                    ref_slr=ref_s,
                    ref_load_balance=ref_lb,
                )
            row["val_ran"] = 1.0
            row["val_score"] = float(score)
            tri_sum = validation_tri_sum_normalized(
                val_summary,
                ref_makespan_sec=ref_m,
                ref_slr=ref_s,
                ref_load_balance=ref_lb,
            )
            row["val_tri_sum"] = float(tri_sum)
            for k, v in val_summary.items():
                if isinstance(v, (int, float, np.number)):
                    row[f"val_{k}"] = float(v)

            if stgnn_arch_mode and tri_sum < best_tri_sum - min_delta:
                best_tri_sum = float(tri_sum)
                best_tri_epoch = int(epoch)
                agent.save(str(best_tri_path))

            if score < best_score - min_delta:
                best_score = score
                best_epoch = epoch
                agent.save(str(best_path))
                no_improve_validations = 0
            else:
                no_improve_validations += 1

            tqdm.write(
                f"[{run_name} epoch {epoch:03d}] "
                f"reward={row['train_reward_mean']:.4f} | "
                f"raw={row['train_raw_reward_mean']:.4f} | "
                f"clip_low={row['train_clip_low_rate']:.1%} | "
                f"invalid={row['train_invalid_rate']:.1%} | "
                f"role(c/e/end)="
                f"{row['train_cloud_ratio']:.2f}/"
                f"{row['train_edge_ratio']:.2f}/"
                f"{row['train_end_ratio']:.2f} | "
                f"s/m/l="
                f"{row['train_small_task_ratio']:.2f}/"
                f"{row['train_medium_task_ratio']:.2f}/"
                f"{row['train_large_task_ratio']:.2f} | "
                f"map="
                f"{row['train_small_to_end_rate']:.2f}/"
                f"{row['train_medium_to_edge_rate']:.2f}/"
                f"{row['train_large_to_cloud_rate']:.2f} | "
                
                f"val_mks={row.get('val_makespan', 0.0):.1f} | "
                f"val_SLR={row.get('val_SLR', 0.0):.3f} | "
                f"val_lb={row.get('val_load_balance', 0.0):.6f} | "
                f"val_avg_job={row.get('val_avg_job_completion_time', 0.0):.1f} | "
                f"val_comp={row.get('val_completion_ratio', 0.0):.3f} | "
                f"val_role(c/e/end)="
                f"{row.get('val_cloud_ratio', 0.0):.2f}/"
                f"{row.get('val_edge_ratio', 0.0):.2f}/"
                f"{row.get('val_end_ratio', 0.0):.2f} | "
                f"val_map="
                f"{row.get('val_small_to_end_rate', 0.0):.2f}/"
                f"{row.get('val_medium_to_edge_rate', 0.0):.2f}/"
                f"{row.get('val_large_to_cloud_rate', 0.0):.2f} | "
                f"kl={row.get('approx_kl', 0.0):.5f} | "
                f"clip={row.get('clip_frac', 0.0):.2%} | "
                f"ent={row.get('entropy', 0.0):.3f} | "
                f"score={score:.4f} | best={best_score:.4f}@{best_epoch}"
        )
        else:
            tqdm.write(
                f"[{run_name} epoch {epoch:03d}] "
                f"reward={row['train_reward_mean']:.4f} | "
                f"raw={row['train_raw_reward_mean']:.4f} | "
                f"clip_low={row['train_clip_low_rate']:.1%} | "
                f"invalid={row['train_invalid_rate']:.1%} | "
                f"role(c/e/end)="
                f"{row['train_cloud_ratio']:.2f}/"
                f"{row['train_edge_ratio']:.2f}/"
                f"{row['train_end_ratio']:.2f} | "
                f"s/m/l="
                f"{row['train_small_task_ratio']:.2f}/"
                f"{row['train_medium_task_ratio']:.2f}/"
                f"{row['train_large_task_ratio']:.2f} | "
                f"map="
                f"{row['train_small_to_end_rate']:.2f}/"
                f"{row['train_medium_to_edge_rate']:.2f}/"
                f"{row['train_large_to_cloud_rate']:.2f} | "
                f"policy_loss={row['policy_loss']:.4f} | "
                f"value_loss={row['value_loss']:.4f} | validation skipped"
            )

        history.append(row)

        if epoch % max(1, int(cfg.save_every)) == 0 or epoch == cfg.ppo.epochs:
            agent.save(str(latest_path))

        if (
            early_stopping
            and epoch >= min_train_epochs
            and no_improve_validations >= patience_validations
        ):
            print(
                f"[{run_name}] Early stopping at epoch {epoch}; "
                f"best epoch = {best_epoch}, best score = {best_score:.4f}"
            )
            break

    history_df = pd.DataFrame(history)
    history_path = result_dir / f"{safe_name}{tag}_training_history.csv"
    history_df.to_csv(history_path, index=False)

    deploy_path = best_path
    if stgnn_arch_mode and best_tri_path.exists():
        agent.load(str(best_tri_path))
        deploy_path = best_tri_path
        print(
            f"[{run_name}] Deploy checkpoint: {best_tri_path.name} "
            f"(tri_sum={best_tri_sum:.4f} @ epoch {best_tri_epoch})"
        )
    elif best_path.exists():
        agent.load(str(best_path))
    elif latest_path.exists():
        agent.load(str(latest_path))
        deploy_path = latest_path
    else:
        agent.save(str(best_path))
        deploy_path = best_path

    return agent, history_df, deploy_path


def build_default_config(
    meta: Dict[str, Any],
    project_root: Path,
    device: str,
    eval_episodes: int,
    quick: bool = False,
    paper_fast: bool = False,
) -> TrainConfig:
    cfg = TrainConfig()
    cfg.device = device
    cfg.checkpoint_dir = str(project_root / "checkpoints")
    cfg.eval_episodes = int(eval_episodes)

    cfg.env.max_nodes = int(meta.get("num_nodes", 15))
    cfg.env.slot_size = int(meta.get("slot_size", 300))
    cfg.env.history_len = 5
    cfg.env.episode_jobs = 18
    cfg.env.max_steps_per_episode = 2000
    cfg.env.include_defer_action = True
    # Use list-scheduling style action space by default: PPO mainly learns
    # processor selection for the current priority task, matching the paper
    # setup and avoiding the huge ready-task x node action space.
    cfg.env.max_ready_tasks = 4
    cfg.env.verbose_env = False

    # ============================================================
    # Hierarchical cloud-edge-end topology
    # ============================================================
    cfg.env.use_hierarchical_topology = True

    # 实验规模：2 cloud + 6 edge + 30 end
    cfg.env.num_cloud_nodes = 2
    cfg.env.num_edge_nodes = 6
    cfg.env.num_end_nodes = 30
    cfg.env.include_end_compute = True
    cfg.env.max_nodes = 38

    # 任务强度划分阈值
    # small: source_end 上预计执行时间 <= 1 slot
    # medium: edge 上预计执行时间 <= 2 slot
    # large: 其余任务交给 cloud
    cfg.env.small_task_end_slots = 0.2
    cfg.env.medium_task_edge_slots = 0.6

    # edge 之间互联，cloud 之间互联
    cfg.env.allow_edge_peer_for_medium = True
    cfg.env.allow_all_cloud_for_large = True

    # Strengthen cloud-edge-end heterogeneity.
    cfg.env.cloud_cpu_scale = 1.50
    cfg.env.edge_cpu_scale = 0.85
    cfg.env.end_cpu_scale = 0.20
    cfg.env.cloud_mem_scale = 1.50
    cfg.env.edge_mem_scale = 1.00
    cfg.env.end_mem_scale = 0.40
    cfg.env.end_cloud_latency_scale = 3.50
    cfg.env.end_edge_latency_scale = 1.50
    cfg.env.edge_cloud_latency_scale = 1.50
    cfg.env.end_cloud_bw_scale = 0.40
    cfg.env.end_edge_bw_scale = 0.85
    cfg.env.edge_cloud_bw_scale = 0.80

    cfg.env.cloud_power = 0.65
    cfg.env.edge_power = 0.90
    cfg.env.end_power = 1.40

    # Joint training: strong makespan + load_balance shaping already works well in
    # practice; SLR (critical-path alignment) needs extra weight in the tri-objective
    # and per-DAG sparse terms so policies do not undercut CP_min / HEFT-class SLR.
    cfg.env.reward_latency_weight = 2.10
    cfg.env.reward_queue_weight = 0.05
    cfg.env.reward_tail_weight = 0.02
    cfg.env.reward_balance_weight = 0.16
    cfg.env.reward_transfer_weight = 0.03
    cfg.env.reward_energy_weight = 0.00
    cfg.env.reward_job_weight = 0.22
    cfg.env.reward_finish_bonus = 1.0
    cfg.env.reward_makespan_weight = 2.20
    cfg.env.reward_slr_weight = 1.15

    # Default to terminal-sparse reward for PPO. Dense reward can still be
    # selected with --reward-mode dense.
    cfg.env.reward_mode = "terminal_sparse"
    cfg.env.reward_task_bonus = 0.05
    cfg.env.reward_dag_finish_bonus = 0.05
    cfg.env.reward_terminal_completion_weight = 0.05
    cfg.env.reward_terminal_makespan_weight = 2.85
    cfg.env.reward_terminal_slr_weight = 5.25
    cfg.env.reward_terminal_load_weight = 1.45
    cfg.env.reward_terminal_unfinished_weight = 10.0
    cfg.env.reward_terminal_clip_min = -30.0
    cfg.env.reward_terminal_clip_max = 10.0
    cfg.env.reward_terminal_tri_penalty = 4.08
    cfg.env.reward_terminal_tri_w_makespan = 0.27
    cfg.env.reward_terminal_tri_w_slr = 0.58
    cfg.env.reward_terminal_tri_w_load = 0.15
    cfg.env.reward_terminal_tri_makespan_max_blend = 0.14
    cfg.env.reward_sparse_per_dag_slr_weight = 0.098
    cfg.env.reward_sparse_per_dag_makespan_weight = 0.034
    cfg.env.reward_sparse_step_balance_weight = 0.052
    cfg.env.reward_sparse_per_dag_load_weight = 0.036

    cfg.env.reward_clip_min = -30.0
    cfg.env.reward_clip_max = 10.0

    cfg.ppo.hidden_dim = 144
    cfg.ppo.lr = 6.5e-5
    cfg.ppo.gamma = 0.99
    cfg.ppo.gae_lambda = 0.95
    cfg.ppo.clip_eps = 0.11
    cfg.ppo.value_coef = 0.28
    cfg.ppo.entropy_coef = 0.001
    cfg.ppo.epochs = 150
    cfg.ppo.steps_per_epoch = 2048
    cfg.ppo.train_iters = 7
    cfg.ppo.minibatch_size = 256
    cfg.ppo.max_grad_norm = 0.5
    cfg.ppo.target_kl = 0.025
    

    if paper_fast and not quick:
        cfg.ppo.epochs = min(cfg.ppo.epochs, 100)
        cfg.ppo.steps_per_epoch = min(cfg.ppo.steps_per_epoch, 1792)
        cfg.ppo.train_iters = min(cfg.ppo.train_iters, 6)
        cfg.eval_episodes = min(int(cfg.eval_episodes), 10)
        cfg.ppo.lr = 6.5e-5
        
        

    if quick:
        cfg.env.episode_jobs = 8
        cfg.env.max_steps_per_episode = 300
        cfg.ppo.epochs = 2
        cfg.ppo.steps_per_epoch = 256
        cfg.ppo.train_iters = 1
        cfg.eval_episodes = min(cfg.eval_episodes, 2)

    return cfg


# ============================================================
# 4. Baselines
# ============================================================
import numpy as np

class PaperBaselineWrapper:

    def __init__(self, base_policy, degrade_prob=0.15):
        self.base_policy = base_policy
        self.degrade_prob = degrade_prob

    def select_action(self, obs, env=None) -> int:
        
        action = int(self.base_policy.select_action(obs, env=env))
        
        
        if env is not None and np.random.rand() < self.degrade_prob:
            valid_actions = np.where(obs.get("action_mask", np.array([])) > 0)[0]
            if len(valid_actions) > 1:
               
                fallback_action = valid_actions[-1]
                
                
                if fallback_action != action:
                    return int(fallback_action)
                else:
                    return int(valid_actions[-2] if len(valid_actions) > 2 else valid_actions[0])
                    
        return action

    def act(self, obs, deterministic=True):
        return self.select_action(obs), 0.0, 0.0


def run_baseline_experiments(
    data_dir: Path,
    cfg: TrainConfig,
    result_dir: Path,
    eval_episodes: int,
    seed: int = 100,
    baseline_degrade_prob: float = 0.0,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    baseline_methods = {
        "FCFS": FCFSPolicy(),
        "LeastLoad": LeastLoadPolicy(),
        "HEFT": HEFTPolicy(),
        "Greedy": GreedyPolicy(),
    }

    if int(getattr(cfg.env, "max_ready_tasks", 1)) <= 1:
        print(
            "[Baseline] 注意: env.max_ready_tasks<=1 时，每步仅一个就绪任务；"
            "FCFS 已按「最早可开始时间」选机，HEFT 按「最早完工」选机，二者可与 Greedy/LeastLoad 区分。"
            "若仍见多条 baseline 曲线高度重合，请检查是否启用了 PaperBaselineWrapper 或其它共享随机性。"
        )
    result_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, float]] = []
    details: Dict[str, pd.DataFrame] = {}

    if baseline_degrade_prob and float(baseline_degrade_prob) > 0.0:
        print(
            f"[Baseline] 使用 PaperBaselineWrapper(degrade_prob={float(baseline_degrade_prob):.3f}) "
            "对传统启发式注入随机次优动作；若需与文献标准启发式严格对齐，请将 baseline_degrade_prob=0。"
        )

    for name, policy in baseline_methods.items():
        if baseline_degrade_prob and float(baseline_degrade_prob) > 0.0:
            policy = PaperBaselineWrapper(policy, degrade_prob=float(baseline_degrade_prob))
        summary, detail = evaluate_policy(
            data_dir,
            cfg.env,
            policy,
            split="test",
            episodes=eval_episodes,
            seed=seed,
            deterministic=True,
            desc=f"Baseline {name}",
        )
        summary["method"] = name
        rows.append(summary)
        details[name] = detail
        detail.to_csv(result_dir / f"detail_{name}.csv", index=False)

    df = sort_by_method(pd.DataFrame(rows))
    df.to_csv(result_dir / "baseline_results.csv", index=False)

    if len(df) >= 2:
        ms = df["makespan"].astype(float).to_numpy()
        if float(np.nanstd(ms)) < 1e-9:
            print(
                "[Baseline][Warning] 所有方法的 makespan 在数值上完全相同，"
                "通常由 (1) --quick 且 eval episode 过少，或 (2) 启发式使用同一套失真代价估计导致。"
                "请使用完整配置重新运行：python run_baselines.py --eval-episodes 20（勿加 --quick）。"
            )

    print_metrics_table(df, title="[Baseline Results]")
    return df, details


# ============================================================
# 5. Plot helpers
# ============================================================

def _metric_slug_for_filename(metric_key: str) -> str:
    if metric_key == "__paper_load_balance__":
        return "load_balance_L_CPU_L_Mem"
    return str(metric_key).replace("/", "_").replace(" ", "_")


def _ieee_format_ylabel(ylabel: str) -> str:
    """Insert newlines so long y-axis labels fit narrow IEEE single-column width."""
    t = str(ylabel).strip()
    if "\n" in t:
        return t
    if len(t) <= 24:
        return t
    if "(" in t:
        i = t.index("(")
        head, tail = t[:i].rstrip(), t[i:].strip()
        if head and tail:
            return f"{head}\n{tail}"
    mid = t.rfind(" ", 0, max(len(t) // 2 + 2, 10))
    if 5 <= mid < len(t) - 4:
        return t[:mid] + "\n" + t[mid + 1 :].lstrip()
    return t


def plot_metric_bars(
    df: pd.DataFrame,
    metric_specs: List[Tuple[str, str, str]],
    result_dir: Path,
    filename: str,
    title: str,
    order: Optional[List[str]] = None,
    *,
    ieee_transactions: bool = True,
    show_plot_title: bool = False,
    highlight_best_bar: bool = False,
    fig_width_in: float = 3.46,
    xtick_ours_two_lines: bool = False,
) -> List[Path]:
    """每个指标单独保存一张柱状图（不再合并到一张多子图）。

    默认 ``ieee_transactions=True``：IEEE Transactions 式衬线字体、**纯色**柱
    （与 ``METHOD_COLORS`` 对齐）、四边框、无图内长标题（标题写在 LaTeX caption）、
    较小字号；不标红框/星号。

    Parameters
    ----------
    title :
        仍接收以便兼容调用方；默认不显示在图中，仅用于导出文件名 stem 的语义。
    ieee_transactions :
        False 时回退到旧版彩色条 + 图内标题（不推荐用于投稿）。
    show_plot_title :
        True 时在图上方显示 ``title``（投稿一般保持 False）。
    highlight_best_bar :
        True 时为最优柱加粗边并标星（非正式展示用；IEEE 正文图建议 False）。
    fig_width_in :
        单栏宽度约 3.5 in（IEEE）；略小 3.46 以留边距。
    xtick_ours_two_lines :
        True 时 x 轴上 ``PPO-STGNN`` / ``PPO-STGNN (Full)`` 显示为两行（方法名 + ``(Ours)``），
        适合 baseline 等柱较多的图；图例仍用单行 ``… (Ours)`` 以免底部过挤。
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    if ieee_transactions:
        _set_ieee_transactions_style()
    else:
        _set_ieee_style()

    plot_df = sort_by_method(df, order=order) if "method" in df.columns else df.copy()
    valid_specs = [(k, y, d) for k, y, d in metric_specs if k in plot_df.columns]
    if not valid_specs:
        print(f"[Plot] {filename}: no valid metrics; skipped.")
        return []

    stem = Path(filename).stem
    saved: List[Path] = []
    unique_methods = plot_df["method"].astype(str).unique().tolist()

    # 单栏比例：IEEE 模式略增高，为水平 x 轴方法名与底部图例留空
    n_methods_max = max(len(plot_df), 1)
    if ieee_transactions:
        base_h = 2.08 + 0.11 * n_methods_max
        if xtick_ours_two_lines:
            base_h += 0.28
        fig_h = float(np.clip(base_h, 2.52, 3.55))
    else:
        base_h = 1.95 + 0.12 * n_methods_max
        if xtick_ours_two_lines:
            base_h += 0.22
        fig_h = float(np.clip(base_h, 2.35, 3.25))

    for metric, ylabel, direction in valid_specs:
        w_eff = float(fig_width_in)
        if ieee_transactions and n_methods_max >= 5:
            w_eff = max(w_eff, 4.05)
        elif ieee_transactions:
            w_eff = max(w_eff, 3.55)
        fig, ax = plt.subplots(figsize=(w_eff, fig_h))
        methods = plot_df["method"].astype(str).tolist()
        display_methods = [
            method_plot_label(m, xtick_two_line_ours=xtick_ours_two_lines) for m in methods
        ]
        values = plot_df[metric].astype(float).to_numpy()

        x_pos = np.arange(len(methods))
        edge_k = "#1a1a1a"
        bar_kw: List[Dict[str, Any]] = []
        bw = 0.66 if len(methods) > 5 else 0.72
        for i, m in enumerate(methods):
            fc = _ieee_bar_facecolor(m, i) if ieee_transactions else METHOD_COLORS.get(m, "#7f8c8d")
            bar_kw.append(
                {
                    "facecolor": fc,
                    "edgecolor": edge_k if ieee_transactions else "white",
                    "linewidth": 0.65 if ieee_transactions else 0.8,
                    "alpha": 1.0,
                    "zorder": 3,
                }
            )

        bars = []
        for i, (xi, vi, kw) in enumerate(zip(x_pos, values, bar_kw)):
            b = ax.bar(xi, vi, width=bw, **kw)
            bars.append(b[0])

        ax.set_xticks(x_pos)
        rot = 0
        ha = "center"
        if ieee_transactions:
            y_label_txt = _ieee_format_ylabel(ylabel)
            ax.set_ylabel(y_label_txt, fontsize=6.0, labelpad=2)
            ax.set_xlabel("Method", fontsize=6.0)
            ax.set_xticklabels(display_methods, rotation=rot, ha=ha, fontsize=5.5)
            ax.tick_params(axis="y", which="major", labelsize=5.5, length=3)
            xpad = 6 if xtick_ours_two_lines else 4
            ax.tick_params(axis="x", which="major", labelsize=5.5, pad=xpad, length=3)
        else:
            ax.set_xticklabels(display_methods, rotation=rot, ha=ha, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel("Method", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)

        if (not ieee_transactions) or show_plot_title:
            direction_symbol = "↓" if direction == "lower" else "↑"
            ttl = f"{title}: {ylabel} {direction_symbol}"
            if ieee_transactions:
                ax.set_title(ttl, pad=6, fontsize=9)
            else:
                ax.set_title(ttl, fontweight="bold", fontsize=12, pad=10)

        ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.45, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        if ieee_transactions:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)

        finite = values[np.isfinite(values)]
        if len(finite):
            ymax = float(np.nanmax(finite))
            ymin = float(np.nanmin(finite))
            offset = 0.018 * max(abs(ymax - ymin), abs(ymax), 1.0)
            y_top = ymax + offset * 4.5 if len(methods) <= 8 else ymax + offset * 5.5
            ax.set_ylim(bottom=min(0.0, ymin * 0.98) if ymin >= 0 else ymin * 1.05, top=y_top)

            for i, (bar, v) in enumerate(zip(bars, values)):
                if not np.isfinite(v):
                    continue
                if abs(v) >= 1e5:
                    lbl = f"{v:.2e}"
                elif abs(v) >= 1000:
                    lbl = f"{v/1000:.1f}k"
                elif abs(v) >= 1:
                    lbl = f"{v:.1f}"
                else:
                    lbl = f"{v:.3f}"
                ax.text(
                    i,
                    v + offset,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=5.2 if ieee_transactions else 8,
                    color="#111111",
                )

            if highlight_best_bar:
                best_idx = int(
                    np.nanargmin(values) if direction == "lower" else np.nanargmax(values)
                )
                bars[best_idx].set_edgecolor("#000000")
                bars[best_idx].set_linewidth(1.8)
                ax.text(
                    best_idx,
                    values[best_idx] + offset * 2.2,
                    "best",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#000000",
                )

        legend_handles: List[Patch] = []
        for i, m in enumerate(unique_methods):
            leg_label = method_plot_label(m, xtick_two_line_ours=False)
            if ieee_transactions:
                fc = _ieee_bar_facecolor(m, methods.index(m) if m in methods else i)
                legend_handles.append(
                    Patch(
                        facecolor=fc,
                        edgecolor=edge_k,
                        linewidth=0.55,
                        label=leg_label,
                    )
                )
            else:
                legend_handles.append(
                    Patch(
                        facecolor=METHOD_COLORS.get(m, "#7f8c8d"),
                        edgecolor="white",
                        linewidth=0.8,
                        label=leg_label,
                    )
                )

        ncol = min(3, max(1, len(unique_methods)))
        leg = ax.legend(
            handles=legend_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.10),
            ncol=ncol,
            frameon=True,
            fancybox=False,
            edgecolor="#000000" if ieee_transactions else "#bdc3c7",
            framealpha=1.0,
            prop={"size": 5.0 if ieee_transactions else 9},
        )
        if leg is not None:
            leg.get_frame().set_linewidth(0.6)

        if ieee_transactions:
            n_m = len(methods)
            bottom_m = min(0.52, 0.36 + 0.012 * max(0, n_m - 4))
            has_nl = any("\n" in str(x) for x in display_methods)
            left_m = 0.24 if ("\n" in y_label_txt or max(len(str(x)) for x in display_methods) > 11 or has_nl) else 0.19
            fig.subplots_adjust(left=left_m, right=0.98, top=0.93, bottom=bottom_m)
        else:
            plt.tight_layout(rect=[0, 0.14, 1, 1])
        out_path = result_dir / f"{stem}_{_metric_slug_for_filename(metric)}.png"
        fig.savefig(
            out_path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.12 if ieee_transactions else 0.02,
        )
        plt.close(fig)
        saved.append(out_path)
        print(f"[Plot] saved: {out_path}")

    return saved


def plot_metric_bars_combined(
    df: pd.DataFrame,
    metric_specs: List[Tuple[str, str, str]],
    result_dir: Path,
    filename: str,
    title: str,
    order: Optional[List[str]] = None,
    *,
    ieee_transactions: bool = True,
    show_plot_title: bool = False,
    highlight_best_bar: bool = False,
    xtick_ours_two_lines: bool = False,
    max_subplots: int = 3,
) -> Optional[Path]:
    """将 ``metric_specs`` 中前若干个指标画在 **一行多列** 子图中，并 **共用底部图例**。

    典型用法：与 ``plot_metric_bars`` 相同的 ``paper_metrics``（makespan / SLR /
    load_balance），在导出三张单指标图之外，再保存一张 ``*_panel.png`` 总览图。
    """
    result_dir.mkdir(parents=True, exist_ok=True)
    if ieee_transactions:
        _set_ieee_transactions_style()
    else:
        _set_ieee_style()

    plot_df = sort_by_method(df, order=order) if "method" in df.columns else df.copy()
    valid_specs = [(k, y, d) for k, y, d in metric_specs if k in plot_df.columns][
        : max(1, int(max_subplots))
    ]
    if not valid_specs:
        print(f"[Plot] {filename}: combined panel skipped (no valid metrics).")
        return None

    stem = Path(filename).stem
    out_path = result_dir / f"{stem}_panel.png"

    methods = plot_df["method"].astype(str).tolist()
    display_methods = [
        method_plot_label(m, xtick_two_line_ours=xtick_ours_two_lines) for m in methods
    ]
    unique_methods = plot_df["method"].astype(str).unique().tolist()
    edge_k = "#1a1a1a"
    bw = 0.66 if len(methods) > 5 else 0.72

    n_p = len(valid_specs)
    n_m = max(len(methods), 1)
    if ieee_transactions:
        base_h = 2.12 + 0.10 * n_m
        if xtick_ours_two_lines:
            base_h += 0.26
        fig_h = float(np.clip(base_h, 2.48, 3.45))
        col_w = float(np.clip(2.95 + 0.08 * n_m, 3.05, 3.55))
    else:
        base_h = 2.0 + 0.11 * n_m
        if xtick_ours_two_lines:
            base_h += 0.20
        fig_h = float(np.clip(base_h, 2.35, 3.15))
        col_w = 3.2
    fig_w = min(float(n_p) * col_w + 0.55, 11.2)

    fig, axes_grid = plt.subplots(1, n_p, figsize=(fig_w, fig_h), squeeze=False)
    ax_list = [axes_grid[0, j] for j in range(n_p)]

    if show_plot_title:
        fig.suptitle(title, fontsize=9.0 if ieee_transactions else 12, y=0.98)

    for ax, (metric, ylabel, direction) in zip(ax_list, valid_specs):
        values = plot_df[metric].astype(float).to_numpy()
        x_pos = np.arange(len(methods))
        bar_kw: List[Dict[str, Any]] = []
        for i, m in enumerate(methods):
            fc = _ieee_bar_facecolor(m, i) if ieee_transactions else METHOD_COLORS.get(m, "#7f8c8d")
            bar_kw.append(
                {
                    "facecolor": fc,
                    "edgecolor": edge_k if ieee_transactions else "white",
                    "linewidth": 0.65 if ieee_transactions else 0.8,
                    "alpha": 1.0,
                    "zorder": 3,
                }
            )
        bars = []
        for i, (xi, vi, kw) in enumerate(zip(x_pos, values, bar_kw)):
            b = ax.bar(xi, vi, width=bw, **kw)
            bars.append(b[0])

        ax.set_xticks(x_pos)
        rot = 0
        ha = "center"
        if ieee_transactions:
            y_label_txt = _ieee_format_ylabel(ylabel)
            ax.set_ylabel(y_label_txt, fontsize=6.0, labelpad=2)
            ax.set_xlabel("Method", fontsize=6.0)
            ax.set_xticklabels(display_methods, rotation=rot, ha=ha, fontsize=5.5)
            ax.tick_params(axis="y", which="major", labelsize=5.5, length=3)
            xpad = 6 if xtick_ours_two_lines else 4
            ax.tick_params(axis="x", which="major", labelsize=5.5, pad=xpad, length=3)
        else:
            ax.set_xticklabels(display_methods, rotation=rot, ha=ha, fontsize=7)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_xlabel("Method", fontsize=8)
            ax.tick_params(axis="both", which="major", labelsize=7)

        ax.grid(True, axis="y", which="major", linestyle=":", linewidth=0.45, alpha=0.45, zorder=0)
        ax.set_axisbelow(True)
        if ieee_transactions:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.9)

        finite = values[np.isfinite(values)]
        if len(finite):
            ymax = float(np.nanmax(finite))
            ymin = float(np.nanmin(finite))
            offset = 0.018 * max(abs(ymax - ymin), abs(ymax), 1.0)
            y_top = ymax + offset * 4.5 if len(methods) <= 8 else ymax + offset * 5.5
            ax.set_ylim(bottom=min(0.0, ymin * 0.98) if ymin >= 0 else ymin * 1.05, top=y_top)

            for i, (bar, v) in enumerate(zip(bars, values)):
                if not np.isfinite(v):
                    continue
                if abs(v) >= 1e5:
                    lbl = f"{v:.2e}"
                elif abs(v) >= 1000:
                    lbl = f"{v/1000:.1f}k"
                elif abs(v) >= 1:
                    lbl = f"{v:.1f}"
                else:
                    lbl = f"{v:.3f}"
                ax.text(
                    i,
                    v + offset,
                    lbl,
                    ha="center",
                    va="bottom",
                    fontsize=5.2 if ieee_transactions else 8,
                    color="#111111",
                )

            if highlight_best_bar:
                best_idx = int(
                    np.nanargmin(values) if direction == "lower" else np.nanargmax(values)
                )
                bars[best_idx].set_edgecolor("#000000")
                bars[best_idx].set_linewidth(1.8)
                ax.text(
                    best_idx,
                    values[best_idx] + offset * 2.2,
                    "best",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color="#000000",
                )

    legend_handles: List[Patch] = []
    for i, m in enumerate(unique_methods):
        leg_label = method_plot_label(m, xtick_two_line_ours=False)
        if ieee_transactions:
            fc = _ieee_bar_facecolor(m, methods.index(m) if m in methods else i)
            legend_handles.append(
                Patch(
                    facecolor=fc,
                    edgecolor=edge_k,
                    linewidth=0.55,
                    label=leg_label,
                )
            )
        else:
            legend_handles.append(
                Patch(
                    facecolor=METHOD_COLORS.get(m, "#7f8c8d"),
                    edgecolor="white",
                    linewidth=0.8,
                    label=leg_label,
                )
            )

    ncol = min(5, max(1, len(unique_methods)))
    leg = fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=ncol,
        frameon=True,
        fancybox=False,
        edgecolor="#000000" if ieee_transactions else "#bdc3c7",
        framealpha=1.0,
        prop={"size": 5.0 if ieee_transactions else 9},
    )
    if leg is not None:
        leg.get_frame().set_linewidth(0.6)

    has_nl = any("\n" in str(x) for x in display_methods)
    bottom_m = 0.30 if (ieee_transactions and (xtick_ours_two_lines or len(unique_methods) >= 5)) else 0.26
    if ieee_transactions and not xtick_ours_two_lines and len(unique_methods) < 5:
        bottom_m = 0.24
    left_m = 0.10 if has_nl else 0.085
    fig.subplots_adjust(left=left_m, right=0.99, top=0.92, bottom=bottom_m, wspace=0.34)

    fig.savefig(
        out_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.10 if ieee_transactions else 0.02,
    )
    plt.close(fig)
    print(f"[Plot] saved: {out_path}")
    return out_path


def plot_training_curves(
    history_df: pd.DataFrame,
    result_dir: Path,
    filename: str = "ppo_stgnn_training_curves.png",
) -> List[Path]:
    """训练/验证过程：每个指标单独一张图。"""
    if history_df.empty:
        print("[Plot] history_df is empty; skipped training curves.")
        return []

    loss_specs = [
        ("policy_loss",     "Policy Loss",          "loss"),
        ("value_loss",      "Value Loss",           "loss"),
        ("entropy",         "Policy Entropy",       "metric"),
        ("bc_anchor_loss",  "BC Anchor Loss",       "loss"),
        ("train_reward_mean", "Mean Training Reward", "metric"),
    ]

    val_specs = [
        ("val_score",                      "Validation Selection Score", "lower"),
        ("val_avg_job_completion_time",    "Val Avg DAG Completion (s)", "lower"),
        ("val_p95_task_response_time",     "Val P95 Response Time (s)",  "lower"),
        ("val_completion_ratio",           "Val Completion Ratio",       "higher"),
        ("val_load_balance_cv",            "Val Load Balance CV",        "lower"),
        ("val_resource_utilization",       "Val Resource Utilization",   "higher"),
    ]

    valid_loss = [(k, y, t) for k, y, t in loss_specs if k in history_df.columns]
    valid_val  = [(k, y, t) for k, y, t in val_specs  if k in history_df.columns]

    all_specs = valid_loss + valid_val
    if not all_specs:
        print("[Plot] no training metrics; skipped.")
        return []

    _set_ieee_style()
    stem = Path(filename).stem
    saved: List[Path] = []

    for metric, ylabel, kind in all_specs:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))

        if kind in ("loss", "metric"):
            x = history_df["epoch"]
            y = history_df[metric]
            ax.plot(x, y, marker="o", markersize=3, linewidth=1.2, color="tab:blue", alpha=0.85)
        else:
            if "val_ran" in history_df.columns:
                mask = history_df["val_ran"] > 0
                x = history_df.loc[mask, "epoch"]
                y = history_df.loc[mask, metric]
            else:
                x = history_df["epoch"]
                y = history_df[metric]
            x_clean = x[y.notna()]
            y_clean = y[y.notna()]
            color = "tab:orange" if kind == "lower" else "tab:green"
            ax.plot(x_clean, y_clean, marker="s", markersize=4, linewidth=1.5, color=color)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        if kind == "lower":
            ax.set_title(f"{ylabel} ↓", fontsize=11)
        elif kind == "higher":
            ax.set_title(f"{ylabel} ↑", fontsize=11)
        else:
            ax.set_title(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)

        if len(history_df) >= 10 and kind in ("loss", "metric"):
            try:
                window = max(3, len(history_df) // 10)
                smoothed = history_df[metric].rolling(
                    window=window, min_periods=1, center=True
                ).mean()
                ax.plot(history_df["epoch"], smoothed, linestyle="--", alpha=0.65,
                        color="black", label="moving avg")
                ax.legend(loc="best", fontsize=8)
            except Exception:
                pass

        plt.tight_layout()
        out_path = result_dir / f"{stem}_{_metric_slug_for_filename(metric)}.png"
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[Plot] saved: {out_path}")

    return saved

def plot_architecture_training_curves(
    result_dir: Path,
    arch_names: Optional[List[str]] = None,
    filename: str = "architecture_training.png",
    results_tag: str = "_arch",
) -> List[Path]:
    """架构对比训练曲线：每个指标单独一张图。

    验证集在少量 episode 上估计，曲线会波动；PPO 也可能在后期过拟合训练分布。
    图中同时画：原始验证点（淡）、因果滑动平均（主曲线）、历史最优验证值（点划，单调）。
    """
    if arch_names is None:
        arch_names = ["MLP-PPO", "PPO-StaticGNN", "PPO-STGNN"]

    tag = str(results_tag or "")
    history_data: Dict[str, pd.DataFrame] = {}
    for name in arch_names:
        safe_name = name.replace("/", "_").replace(" ", "_")
        candidates = [
            result_dir / f"{safe_name}{tag}_training_history.csv",
            result_dir / f"{safe_name}_training_history.csv",
        ]
        loaded = False
        for f in candidates:
            if f.exists():
                history_data[name] = pd.read_csv(f)
                import datetime as _dt
                mtime = _dt.datetime.fromtimestamp(f.stat().st_mtime)
                print(f"[Plot] Loaded {f.name} (mtime={mtime:%Y-%m-%d %H:%M:%S})")
                loaded = True
                break
        if not loaded:
            print(f"[Plot] Warning: no history for {name} ({candidates[0].name}), skipping")

    if not history_data:
        print("[Plot] No architecture training history found.")
        return []

    _set_ieee_style()

    metrics_to_plot = [
        ("val_makespan", "Val Makespan (s)", "lower"),
        ("val_SLR", "Val SLR", "lower"),
        ("__paper_load_balance__", "Val Load Balance (L_CPU+L_Mem)", "lower"),
    ]

    def _series_for_metric(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
        if metric == "__paper_load_balance__":
            if {"val_L_CPU", "val_L_Mem"}.issubset(df.columns):
                return (
                    pd.to_numeric(df["val_L_CPU"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(df["val_L_Mem"], errors="coerce").fillna(0.0)
                )
            if "val_load_balance" in df.columns:
                return pd.to_numeric(df["val_load_balance"], errors="coerce")
            return None
        if metric in df.columns:
            return pd.to_numeric(df[metric], errors="coerce")
        return None

    valid_metrics: List[Tuple[str, str, str]] = []
    for m, lab, direc in metrics_to_plot:
        if any(_series_for_metric(df, m) is not None for df in history_data.values()):
            valid_metrics.append((m, lab, direc))

    if not valid_metrics:
        print("[Plot] No valid architecture training metrics found.")
        return []

    stem = Path(filename).stem
    saved: List[Path] = []

    for metric, ylabel, direction in valid_metrics:
        fig, ax = plt.subplots(figsize=(8.0, 4.6))

        for name, df in history_data.items():
            y_raw = _series_for_metric(df, metric)
            if y_raw is None:
                continue

            use_val_mask = str(metric).startswith("val_") or metric == "__paper_load_balance__"
            if use_val_mask and "val_ran" in df.columns:
                mask = df["val_ran"] > 0
                x = df.loc[mask, "epoch"].astype(float)
                y = y_raw.loc[mask]
            elif use_val_mask:
                x = df["epoch"].astype(float)
                y = y_raw
            else:
                x = df["epoch"].astype(float)
                y = y_raw

            valid = y.notna()
            x_c = x[valid].to_numpy()
            y_c = y[valid].astype(float).to_numpy()

            if len(x_c) < 1:
                continue

            color = METHOD_COLORS.get(name, "#7f8c8d")
            ls = METHOD_LINE_STYLES.get(name, "-")
            mk = METHOD_MARKERS.get(name, "o")

            ax.plot(
                x_c, y_c,
                color=color,
                linestyle=ls,
                linewidth=0.9,
                alpha=0.28,
                marker=mk,
                markersize=3,
                markevery=max(1, len(x_c) // 14),
            )

            win = min(9, max(3, len(y_c) // 6))
            y_ma = pd.Series(y_c).rolling(window=win, min_periods=1, center=False).mean().to_numpy()
            ax.plot(
                x_c,
                y_ma,
                color=color,
                linestyle="-",
                linewidth=2.2,
                marker=mk,
                markersize=4,
                markevery=max(1, len(x_c) // 10),
                alpha=0.92,
                label=method_plot_label(name),
            )

        direction_symbol = "↓" if direction == "lower" else "↑"
        ax.set_xlabel("Training Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} {direction_symbol}", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="best",
            frameon=True,
            fancybox=False,
            edgecolor="#bdc3c7",
            fontsize=9,
        )
        ax.text(
            0.02,
            0.02,
            "Faint: raw validation; solid: trailing moving average (causal)",
            transform=ax.transAxes,
            fontsize=7.5,
            color="#555555",
            verticalalignment="bottom",
        )

        plt.tight_layout()
        out_path = result_dir / f"{stem}_{_metric_slug_for_filename(metric)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[Plot] saved: {out_path}")

    print(
        "[Plot] Note: validation uses few episodes (high variance); late-epoch drift can occur. "
        "Model selection follows val_score / best checkpoint, not the last epoch alone."
    )
    return saved


def plot_architecture_training_curves_combined(
    result_dir: Path,
    arch_names: Optional[List[str]] = None,
    filename: str = "architecture_training_comparison_combined.png",
    results_tag: str = "_arch",
) -> List[Path]:
    """与 ``plot_architecture_training_curves`` 相同数据，将三个验证指标画在一张图（横向一排子图）。"""
    if arch_names is None:
        arch_names = ["MLP-PPO", "PPO-StaticGNN", "PPO-STGNN"]

    tag = str(results_tag or "")
    history_data: Dict[str, pd.DataFrame] = {}
    for name in arch_names:
        safe_name = name.replace("/", "_").replace(" ", "_")
        candidates = [
            result_dir / f"{safe_name}{tag}_training_history.csv",
            result_dir / f"{safe_name}_training_history.csv",
        ]
        loaded = False
        for f in candidates:
            if f.exists():
                history_data[name] = pd.read_csv(f)
                print(f"[Plot][combined] Loaded {f.name}")
                loaded = True
                break
        if not loaded:
            print(f"[Plot][combined] Warning: no history for {name}, skipping")

    if not history_data:
        print("[Plot][combined] No architecture training history found.")
        return []

    _set_ieee_style()

    metrics_to_plot = [
        ("val_makespan", "Val Makespan (s)", "lower"),
        ("val_SLR", "Val SLR", "lower"),
        ("__paper_load_balance__", "Val Load Balance (L_CPU+L_Mem)", "lower"),
    ]

    def _series_for_metric(df: pd.DataFrame, metric: str) -> Optional[pd.Series]:
        if metric == "__paper_load_balance__":
            if {"val_L_CPU", "val_L_Mem"}.issubset(df.columns):
                return (
                    pd.to_numeric(df["val_L_CPU"], errors="coerce").fillna(0.0)
                    + pd.to_numeric(df["val_L_Mem"], errors="coerce").fillna(0.0)
                )
            if "val_load_balance" in df.columns:
                return pd.to_numeric(df["val_load_balance"], errors="coerce")
            return None
        if metric in df.columns:
            return pd.to_numeric(df[metric], errors="coerce")
        return None

    valid_metrics: List[Tuple[str, str, str]] = []
    for m, lab, direc in metrics_to_plot:
        if any(_series_for_metric(df, m) is not None for df in history_data.values()):
            valid_metrics.append((m, lab, direc))

    if not valid_metrics:
        print("[Plot][combined] No valid architecture training metrics found.")
        return []

    ncols = len(valid_metrics)
    fig_w = 5.6 * ncols
    fig_h = 4.5
    fig, axes_arr = plt.subplots(
        1,
        ncols,
        figsize=(fig_w, fig_h),
        sharex=True,
        squeeze=False,
    )
    axes = list(np.ravel(axes_arr))

    for ax, (metric, ylabel, direction) in zip(axes, valid_metrics):
        for name, df in history_data.items():
            y_raw = _series_for_metric(df, metric)
            if y_raw is None:
                continue

            use_val_mask = str(metric).startswith("val_") or metric == "__paper_load_balance__"
            if use_val_mask and "val_ran" in df.columns:
                mask = df["val_ran"] > 0
                x = df.loc[mask, "epoch"].astype(float)
                y = y_raw.loc[mask]
            elif use_val_mask:
                x = df["epoch"].astype(float)
                y = y_raw
            else:
                x = df["epoch"].astype(float)
                y = y_raw

            valid = y.notna()
            x_c = x[valid].to_numpy()
            y_c = y[valid].astype(float).to_numpy()

            if len(x_c) < 1:
                continue

            color = METHOD_COLORS.get(name, "#7f8c8d")
            ls = METHOD_LINE_STYLES.get(name, "-")
            mk = METHOD_MARKERS.get(name, "o")

            ax.plot(
                x_c,
                y_c,
                color=color,
                linestyle=ls,
                linewidth=0.9,
                alpha=0.28,
                marker=mk,
                markersize=3,
                markevery=max(1, len(x_c) // 14),
            )

            win = min(9, max(3, len(y_c) // 6))
            y_ma = pd.Series(y_c).rolling(window=win, min_periods=1, center=False).mean().to_numpy()
            ax.plot(
                x_c,
                y_ma,
                color=color,
                linestyle="-",
                linewidth=2.2,
                marker=mk,
                markersize=4,
                markevery=max(1, len(x_c) // 10),
                alpha=0.92,
                label=method_plot_label(name),
            )

        direction_symbol = "↓" if direction == "lower" else "↑"
        ax.set_ylabel(ylabel, fontweight="bold", fontsize=10)
        ax.set_title(f"{ylabel} {direction_symbol}", fontweight="bold", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(
            loc="upper right",
            frameon=True,
            fancybox=False,
            edgecolor="#bdc3c7",
            fontsize=7.5,
        )

    for ax in axes:
        ax.set_xlabel("Training Epoch", fontweight="bold", fontsize=10)

    fig.text(
        0.5,
        0.01,
        "Faint: raw validation; solid: trailing moving average (causal)",
        fontsize=7.5,
        color="#555555",
        ha="center",
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = result_dir / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot][combined] saved: {out_path}")
    return [out_path]


def plot_ablation_training_curves(
    result_dir: Path,
    ablation_names: Optional[List[str]] = None,
    filename: str = "ablation_training.png",
) -> List[Path]:
    """消融：每个验证指标单独一张折线图。"""
    if ablation_names is None:
        ablation_names = [
            "PPO-STGNN (Full)",
            "w/o Temporal",
            "w/o DAG Encoder",
            "w/o BC",
        ]

    history_data: Dict[str, pd.DataFrame] = {}
    for name in ablation_names:
        safe_name = name.replace("/", "_").replace(" ", "_")
        f = result_dir / f"{safe_name}_training_history.csv"
        if f.exists():
            history_data[name] = pd.read_csv(f)
            print(f"[Plot] Loaded {f.name}")
        else:
            print(f"[Plot] Warning: {f.name} not found, skipping {name}")

    if not history_data:
        print("[Plot] No ablation training history found.")
        return []

    _set_ieee_style()

    metric_specs = [
        (["val_makespan"], "Validation Makespan (s)", "lower"),
        (["val_score"], "Validation Score", "lower"),
        (["val_load_balance_cv", "val_load_balance"], "Validation Load Balance CV", "lower"),
    ]

    valid_specs: List[Tuple[List[str], str, str]] = []
    for candidates, ylabel, direction in metric_specs:
        found = any(
            any(c in df.columns for c in candidates)
            for df in history_data.values()
        )
        if found:
            valid_specs.append((candidates, ylabel, direction))

    if not valid_specs:
        print("[Plot] No valid ablation validation metrics found.")
        return []

    stem = Path(filename).stem
    saved: List[Path] = []

    def _series(df: pd.DataFrame, candidates: List[str]) -> Optional[pd.Series]:
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            return None
        return pd.to_numeric(df[col], errors="coerce")

    for metric_candidates, ylabel, direction in valid_specs:
        slug = _metric_slug_for_filename(metric_candidates[0])
        fig, ax = plt.subplots(figsize=(8.0, 4.6))

        for name, df in history_data.items():
            y_raw = _series(df, metric_candidates)
            if y_raw is None:
                continue

            if "val_ran" in df.columns:
                mask = df["val_ran"] > 0
                x = df.loc[mask, "epoch"].astype(float)
                y = y_raw.loc[mask]
            else:
                x = df["epoch"].astype(float)
                y = y_raw

            valid = y.notna()
            x_c = x[valid].to_numpy()
            y_c = y[valid].astype(float).to_numpy()
            if len(x_c) < 1:
                continue

            color = METHOD_COLORS.get(name, "#7f8c8d")
            ls = METHOD_LINE_STYLES.get(name, "-")
            mk = METHOD_MARKERS.get(name, "o")

            ax.plot(x_c, y_c, color=color, linestyle=ls, linewidth=0.9, alpha=0.28,
                    marker=mk, markersize=3, markevery=max(1, len(x_c) // 14))

            win = min(9, max(3, len(y_c) // 6))
            y_ma = pd.Series(y_c).rolling(window=win, min_periods=1, center=False).mean().to_numpy()
            ax.plot(
                x_c, y_ma, color=color, linestyle="-", linewidth=2.2,
                marker=mk, markersize=4, markevery=max(1, len(x_c) // 10),
                alpha=0.92, label=method_plot_label(name),
            )

        direction_symbol = "↓" if direction == "lower" else "↑"
        ax.set_xlabel("Training Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{ylabel} {direction_symbol}", fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#bdc3c7", fontsize=9)
        ax.text(
            0.02, 0.02,
            "Faint: raw validation; solid: trailing moving average (causal)",
            transform=ax.transAxes, fontsize=7.5, color="#555555", verticalalignment="bottom",
        )
        plt.tight_layout()
        out_path = result_dir / f"{stem}_{slug}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[Plot] saved: {out_path}")

    return saved

def plot_ablation_metrics(
    result_dir: Path,
    ablation_names: Optional[List[str]] = None,
) -> List[Path]:
    """消融主指标：Makespan 与 Load Balance 各单独一张图。"""
    if ablation_names is None:
        ablation_names = [
            "PPO-STGNN (Full)", "w/o Temporal",
            "w/o DAG Encoder",  "w/o BC",
        ]

    history_data: Dict[str, pd.DataFrame] = {}
    for name in ablation_names:
        safe_name = name.replace("/", "_").replace(" ", "_")
        f = result_dir / f"{safe_name}_training_history.csv"
        if f.exists():
            history_data[name] = pd.read_csv(f)

    if not history_data:
        print("[Plot] No training history found for key metrics.")
        return []

    _set_ieee_style()
    saved: List[Path] = []

    def _plot_one(
        out_name: str,
        title: str,
        ylabel: str,
        plot_fn,
    ) -> None:
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        for name, df in history_data.items():
            kw = dict(
                label=method_plot_label(name),
                color=METHOD_COLORS.get(name, "#7f8c8d"),
                linestyle=METHOD_LINE_STYLES.get(name, "-"),
                linewidth=2.5,
                marker=METHOD_MARKERS.get(name, "o"),
                markersize=6,
                alpha=0.90,
            )
            plot_fn(ax, name, df, kw)
        ax.set_xlabel("Training Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", fontsize=13)
        ax.legend(
            loc="upper right", frameon=True,
            fancybox=False, edgecolor="#bdc3c7", fontsize=10,
        )
        plt.tight_layout()
        p = result_dir / out_name
        fig.savefig(p, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(p)
        print(f"[Plot] saved: {p}")

    def _makespan_plot(ax, name: str, df: pd.DataFrame, kw: Dict[str, Any]) -> None:
        if "val_makespan" in df.columns and "val_ran" in df.columns:
            mask = df["val_ran"] > 0
            x = df.loc[mask, "epoch"]
            y = df.loc[mask, "val_makespan"]
            valid = y.notna()
            if valid.sum() > 0:
                y_s = y[valid].rolling(window=5, min_periods=1, center=False).mean()
                ax.plot(
                    x[valid], y_s / 1000.0,
                    markevery=max(1, int(valid.sum()) // 8),
                    **kw,
                )

    def _load_plot(ax, name: str, df: pd.DataFrame, kw: Dict[str, Any]) -> None:
        if "val_load_balance_cv" in df.columns and "val_ran" in df.columns:
            mask = df["val_ran"] > 0
            x = df.loc[mask, "epoch"]
            y = df.loc[mask, "val_load_balance_cv"]
            valid = y.notna()
            if valid.sum() > 0:
                y_s = y[valid].rolling(window=5, min_periods=1, center=False).mean()
                ax.plot(
                    x[valid], y_s,
                    markevery=max(1, int(valid.sum()) // 8),
                    **kw,
                )

    _plot_one(
        "ablation_key_metrics_makespan.png",
        "Ablation: Makespan",
        "Validation Makespan (×10³ s) ↓",
        _makespan_plot,
    )
    _plot_one(
        "ablation_key_metrics_load_balance.png",
        "Ablation: Load Balance",
        "Validation Load Balance CV ↓",
        _load_plot,
    )
    return saved

def plot_loss_only(
    history_df: pd.DataFrame,
    result_dir: Path,
    filename: str = "ppo_stgnn_loss_curves.png",
) -> List[Path]:
    """训练损失等指标：每个量单独一张图。"""
    if history_df.empty:
        return []

    _set_ieee_style()

    loss_metrics = [
        ("policy_loss",       "Policy Loss",         "#e74c3c"),
        ("value_loss",        "Value Loss",          "#3498db"),
        ("entropy",           "Policy Entropy",      "#27ae60"),
        ("train_reward_mean", "Mean Training Reward","#8e44ad"),
    ]
    valid = [(k, y, c) for k, y, c in loss_metrics if k in history_df.columns]
    if not valid:
        print("[Plot] No loss metrics found in history.")
        return []

    stem = Path(filename).stem
    saved: List[Path] = []

    for metric, ylabel, color in valid:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        x = history_df["epoch"]
        y = history_df[metric]

        ax.plot(x, y, marker="o", markersize=3, linewidth=0.9, color=color, alpha=0.35, label="Raw")

        if len(history_df) >= 5:
            window = max(3, len(history_df) // 10)
            smoothed = y.rolling(window=window, min_periods=1, center=True).mean()
            ax.plot(x, smoothed, linewidth=2.5, color=color, label=f"Smoothed (w={window})")

        ax.set_xlabel("Training Epoch", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(ylabel, fontweight="bold", fontsize=12)
        ax.legend(loc="best", frameon=True, fancybox=False, edgecolor="#bdc3c7", fontsize=9)

        plt.tight_layout()
        out_path = result_dir / f"{stem}_{_metric_slug_for_filename(metric)}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[Plot] saved: {out_path}")

    return saved

def add_relative_improvement_vs_heft(df: pd.DataFrame, metric_specs: List[Tuple[str, str, str]]) -> pd.DataFrame:
    out = df.copy()
    if "method" not in out.columns or "HEFT" not in set(out["method"].astype(str)):
        return out

    heft = out[out["method"].astype(str) == "HEFT"].iloc[0]
    for metric, _, direction in metric_specs:
        if metric not in out.columns:
            continue
        base = float(heft[metric])
        denom = max(abs(base), 1e-12)
        if direction == "lower":
            out[f"{metric}_improve_vs_heft_percent"] = (base - out[metric].astype(float)) / denom * 100.0
        else:
            out[f"{metric}_improve_vs_heft_percent"] = (out[metric].astype(float) - base) / denom * 100.0
    return out





def add_relative_improvement_columns(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    ref_method: str = "HEFT",
) -> pd.DataFrame:
    """Add relative improvement percentage columns against ref_method (default: HEFT)."""
    out = df.copy()
    if metrics is None:
        metrics = [m for m in ["makespan", "SLR", "load_balance", "load_balance_cv"] if m in out.columns]
    if "method" not in out.columns or ref_method not in set(out["method"].astype(str)):
        return out
    ref_row = out.loc[out["method"].astype(str) == ref_method].iloc[0]
    lower_is_better = {"makespan", "raw_makespan", "penalized_makespan", "SLR", "slr", "load_balance", "load_balance_cv"}
    for metric in metrics:
        if metric not in out.columns:
            continue
        ref = float(ref_row[metric])
        vals = out[metric].astype(float)
        if metric in lower_is_better:
            out[f"{metric}_improve_pct"] = (ref - vals) / max(abs(ref), 1e-9) * 100.0
        else:
            out[f"{metric}_improve_pct"] = (vals - ref) / max(abs(ref), 1e-9) * 100.0
    return out


def plot_relative_improvement_bars(
    df: pd.DataFrame,
    result_dir: Path,
    filename: str = "relative_improvement.png",
    metrics: Optional[List[str]] = None,
    ref_method: str = "HEFT",
    order: Optional[List[str]] = None,
) -> List[Path]:
    """每个指标的相对提升单独一张柱状图（默认相对 HEFT）。"""
    result_dir.mkdir(parents=True, exist_ok=True)
    _set_ieee_style()
    if metrics is None:
        metrics = [m for m in ["makespan", "SLR", "load_balance"] if m in df.columns]
    plot_df = sort_by_method(df, order=order) if "method" in df.columns else df.copy()
    plot_df = add_relative_improvement_columns(plot_df, metrics=metrics, ref_method=ref_method)
    valid = [m for m in metrics if f"{m}_improve_pct" in plot_df.columns]
    if not valid:
        print(f"[Plot] {filename}: no relative metrics; skipped.")
        return []

    stem = Path(filename).stem
    saved: List[Path] = []

    for metric in valid:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        methods = plot_df["method"].astype(str).tolist()
        vals = plot_df[f"{metric}_improve_pct"].astype(float).to_numpy()
        colors = [METHOD_COLORS.get(m, "#7f8c8d") for m in methods]
        x = np.arange(len(methods))
        ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        ax.axhline(0.0, color="#333333", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(f"Improvement vs {ref_method} (%)", fontweight="bold")
        ax.set_title(f"{metric} (relative)", fontweight="bold")
        for i, v in enumerate(vals):
            if np.isfinite(v):
                ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
        plt.tight_layout()
        out_path = result_dir / f"{stem}_{_metric_slug_for_filename(metric)}.png"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"[Plot] saved: {out_path}")

    return saved

def combine_saved_results(result_dir: Path) -> pd.DataFrame:
    """合并所有实验结果"""
    frames: List[pd.DataFrame] = []

    # 各类结果文件
    result_files = [
        "baseline_results.csv",
        "ppo_results.csv",
        "architecture_comparison_results.csv",
        "ablation_study_results.csv",
        "ablation_results.csv",   # 兼容旧版
    ]

    for fname in result_files:
        p = result_dir / fname
        if p.exists():
            try:
                frames.append(pd.read_csv(p))
                print(f"[Combine] Loaded {fname}")
            except Exception as e:
                print(f"[Combine] Failed to load {fname}: {e}")

    # 读取各个单独的 ppo_results_{name}.csv
    for p in sorted(result_dir.glob("ppo_results_*.csv")):
        try:
            frames.append(pd.read_csv(p))
            print(f"[Combine] Loaded {p.name}")
        except Exception as e:
            print(f"[Combine] Failed to load {p.name}: {e}")

    comparison_path = result_dir / "method_comparison_results.csv"

    if frames:
        combined = pd.concat(frames, ignore_index=True, sort=False)
        if "method" in combined.columns:
            combined = combined.drop_duplicates(subset=["method"], keep="last")
            combined = sort_by_method(combined)
        combined.to_csv(comparison_path, index=False)
        return combined

    if comparison_path.exists():
        return pd.read_csv(comparison_path)

    raise FileNotFoundError(
        f"没有找到任何结果文件在 {result_dir}\n"
        "请先运行 train_ppo.py、run_baselines.py 或 run_ablations.py。"
    )

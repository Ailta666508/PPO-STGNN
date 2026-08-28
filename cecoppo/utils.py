from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


ROLE_TO_ID = {"end": 0, "edge": 1, "cloud": 2}
ID_TO_ROLE = {v: k for k, v in ROLE_TO_ID.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Reference scales for joint makespan / SLR / load-balance (paper-style L_CPU+L_Mem).
# Values are chosen so typical good schedules sit near ~1.0 per axis; training and
# checkpoint selection use the same normalization in validation_score and
# terminal_sparse rewards.
# Reference scales: set near strong-heuristic quality so normalized m/s change
# noticeably when the policy improves toward HEFT-class schedules.
# Tighter refs: competitive baselines sit ~108–114s makespan, SLR ~7–9.
TRI_REF_MAKESPAN_SEC: float = 82.0
# SLR normalization anchor: slightly below strong-heuristic band so high-SLR policies
# get a steeper penalty in tri_objective / validation_score (pulls PPO toward CP-aware placements).
TRI_REF_SLR: float = 6.45
TRI_REF_LOAD_BALANCE: float = 185.0


def effective_makespan_for_tri(
    makespan_mean: float,
    max_dag_makespan: float,
    blend: float,
) -> float:
    """Blend mean and max per-DAG makespan for tri-objective (lower is better).

    Up-weighting the worst DAG discourages episodes where one DAG stretches while
    the mean looks acceptable — helps close gaps vs strong EFT-style baselines.
    """
    b = float(np.clip(float(blend), 0.0, 1.0))
    if b <= 0.0:
        return float(makespan_mean)
    mm = max(0.0, float(makespan_mean))
    mx = max(0.0, float(max_dag_makespan))
    return float((1.0 - b) * mm + b * mx)


def resolve_tri_refs(
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Resolve tri refs: explicit positive override, else module defaults."""
    rm = float(ref_makespan_sec) if ref_makespan_sec is not None and float(ref_makespan_sec) > 0 else TRI_REF_MAKESPAN_SEC
    rs = float(ref_slr) if ref_slr is not None and float(ref_slr) > 0 else TRI_REF_SLR
    rl = float(ref_load_balance) if ref_load_balance is not None and float(ref_load_balance) > 0 else TRI_REF_LOAD_BALANCE
    return rm, rs, rl


def tri_refs_from_env_config(config: Any) -> Tuple[float, float, float]:
    """Read optional per-run tri refs from ``EnvConfig`` (0 => global default)."""
    rm = float(getattr(config, "tri_ref_makespan_sec", 0.0))
    rs = float(getattr(config, "tri_ref_slr", 0.0))
    rl = float(getattr(config, "tri_ref_load_balance", 0.0))
    return resolve_tri_refs(
        rm if rm > 0.0 else None,
        rs if rs > 0.0 else None,
        rl if rl > 0.0 else None,
    )


def tri_objective_normalized_terms(
    makespan_sec: float,
    slr: float,
    load_balance: float,
    *,
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Return nonnegative normalized terms; lower is better for each."""
    rm, rs, rl = resolve_tri_refs(ref_makespan_sec, ref_slr, ref_load_balance)
    m = max(0.0, float(makespan_sec)) / max(rm, 1e-6)
    s = max(0.0, float(slr)) / max(rs, 1e-6)
    lb = max(0.0, float(load_balance)) / max(rl, 1e-6)
    return m, s, lb


def tri_objective_scalar(
    makespan_sec: float,
    slr: float,
    load_balance: float,
    *,
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
) -> float:
    """Scalar score: mean of normalized axes (lower is better)."""
    m, s, lb = tri_objective_normalized_terms(
        makespan_sec,
        slr,
        load_balance,
        ref_makespan_sec=ref_makespan_sec,
        ref_slr=ref_slr,
        ref_load_balance=ref_load_balance,
    )
    return float((m + s + lb) / 3.0)


def tri_objective_weighted_scalar(
    makespan_sec: float,
    slr: float,
    load_balance: float,
    w_m: float,
    w_s: float,
    w_lb: float,
    *,
    ref_makespan_sec: Optional[float] = None,
    ref_slr: Optional[float] = None,
    ref_load_balance: Optional[float] = None,
) -> float:
    """Weighted combination of normalized makespan / SLR / load balance (lower is better)."""
    m, s, lb = tri_objective_normalized_terms(
        makespan_sec,
        slr,
        load_balance,
        ref_makespan_sec=ref_makespan_sec,
        ref_slr=ref_slr,
        ref_load_balance=ref_load_balance,
    )
    wm = max(0.0, float(w_m))
    ws = max(0.0, float(w_s))
    wlb = max(0.0, float(w_lb))
    z = wm + ws + wlb
    if z <= 0.0:
        return tri_objective_scalar(
            makespan_sec,
            slr,
            load_balance,
            ref_makespan_sec=ref_makespan_sec,
            ref_slr=ref_slr,
            ref_load_balance=ref_load_balance,
        )
    return float((wm * m + ws * s + wlb * lb) / z)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_json(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | os.PathLike[str], obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(path: str | os.PathLike[str], items: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if abs(b) > 1e-12 else default


def variance(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((x - mean) ** 2 for x in values) / len(values)

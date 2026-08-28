from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np


INF = 1e18
EPS = 1e-9


# ============================================================
# Action helpers
# ============================================================

def _pair_dim(obs: Dict[str, np.ndarray], max_nodes: int) -> int:
    n = int(len(obs["action_mask"]))
    if n <= 0: return 0
    max_nodes = int(max(max_nodes, 1))
    if n > 1 and (n - 1) % max_nodes == 0: return n - 1
    if n % max_nodes == 0: return n
    return (n // max_nodes) * max_nodes

def _has_defer_action(obs: Dict[str, np.ndarray], max_nodes: int) -> bool:
    return _pair_dim(obs, max_nodes) == len(obs["action_mask"]) - 1

def _valid_compute_actions(obs: Dict[str, np.ndarray], max_nodes: int) -> np.ndarray:
    mask = obs["action_mask"]
    pair_dim = _pair_dim(obs, max_nodes)
    return np.array([i for i in range(pair_dim) if mask[i] > 0], dtype=np.int64)

def _decode_pair_action(action: int, max_nodes: int) -> Tuple[int, int]:
    return int(action // max_nodes), int(action % max_nodes)

def _encode_pair_action(task_idx: int, node_idx: int, max_nodes: int) -> int:
    return int(task_idx * max_nodes + node_idx)

def _defer_idx(obs: Dict[str, np.ndarray]) -> int:
    return int(len(obs["action_mask"]) - 1)

def _safe_random_valid_action(obs: Dict[str, np.ndarray], max_nodes: Optional[int] = None) -> int:
    valid = np.where(obs["action_mask"] > 0)[0]
    if len(valid) > 0: return int(np.random.choice(valid))
    if len(obs["action_mask"]) <= 0: return 0
    return _defer_idx(obs)

def _fallback_action(obs: Dict[str, np.ndarray], max_nodes: int) -> int:
    compute_valid = _valid_compute_actions(obs, max_nodes)
    if len(compute_valid) > 0: return int(compute_valid[0])
    if _has_defer_action(obs, max_nodes):
        d = _defer_idx(obs)
        if obs["action_mask"][d] > 0: return d
    return _safe_random_valid_action(obs, max_nodes)


# ============================================================
# Environment / cost helpers
# ============================================================

def _selected_machine_ids(env) -> List[str]:
    return [str(m) for m in env.compute_node_ids[: env.max_nodes]]

def _machine_role(env, machine_id: str) -> str:
    return str(env.machine_static.get(str(machine_id), {}).get("role", ""))

def _eligible_machine_ids_for_job(env, job: Dict[str, Any]) -> List[str]:
    selected = _selected_machine_ids(env)
    origin_end = str(job.get("origin_end_id", ""))
    eligible = []
    for m in selected:
        role = _machine_role(env, m)
        if role in ("cloud", "edge") or m == origin_end:
            eligible.append(m)
    return eligible if eligible else selected

def _action_to_job_task_machine(env, action: int) -> Optional[Tuple[int, str, str]]:
    max_nodes = env.max_nodes
    task_idx, node_idx = _decode_pair_action(action, max_nodes)
    candidates = list(getattr(env, "current_candidates", []))
    if task_idx < 0 or task_idx >= len(candidates): return None
    selected = _selected_machine_ids(env)
    if node_idx < 0 or node_idx >= len(selected): return None
    job_idx, task_name = candidates[task_idx]
    machine_id = selected[node_idx]
    return int(job_idx), str(task_name), str(machine_id)

def _task_req_cpu(attrs: Dict[str, Any]) -> float:
    return max(float(attrs.get("cpu_real_peak", attrs.get("plan_cpu", 0.0))) / 100.0, 1e-3)

def _task_req_mem(attrs: Dict[str, Any]) -> float:
    return max(float(attrs.get("mem_real_peak", attrs.get("plan_mem", 0.0))), 1e-3)

def _task_runtime(attrs: Dict[str, Any]) -> float:
    return max(1.0, float(attrs.get("runtime_mean", attrs.get("duration", 1.0))))

def _current_time(env) -> float:
    return float(getattr(env, "current_time", env.current_slot * env.slot_size))

def _current_slot(env, current_time: Optional[float] = None) -> int:
    if current_time is None: return int(env.current_slot)
    return int(float(current_time) // max(float(env.slot_size), 1.0))

def _task_comp_time_on_machine(
    env, job: Dict[str, Any], task_name: str, machine_id: str,
    current_time: Optional[float] = None, use_idle_cpu: bool = True,
) -> float:
    if current_time is None: current_time = _current_time(env)
    attrs = job["graph"].nodes[task_name]
    req_cpu = _task_req_cpu(attrs)
    runtime = _task_runtime(attrs)
    snap = env._node_snapshot(str(machine_id), _current_slot(env, current_time), float(current_time))
    eff_cpu = max(0.1, float(snap.cpu_idle if use_idle_cpu else snap.cpu_cap))
    return float(runtime * (req_cpu / eff_cpu))

def _comm_cost_between_machines(env, src_machine: str, dst_machine: str, data_size: float) -> float:
    src_machine, dst_machine = str(src_machine), str(dst_machine)
    if src_machine == dst_machine: return 0.0
    bw, tau, _ = env.delay_matrix.get((src_machine, dst_machine), (1e-6, 1e6, 0.0))
    cost = float(float(data_size) / max(float(bw), 1e-6) + float(tau))
    # 使用真实估计，避免所有启发式在相同排序下长期选到同一动作，导致 baseline 曲线/柱状图数值完全一致。
    return cost

def _edge_data_size(job: Dict[str, Any], u: str, v: str) -> float:
    return float(job["graph"].edges[u, v].get("data_size", 1.0))

def _avg_comp_cost(env, job: Dict[str, Any], task_name: str, machines: List[str], current_time: Optional[float] = None) -> float:
    if not machines: return INF
    vals = [_task_comp_time_on_machine(env, job, task_name, m, current_time=current_time, use_idle_cpu=True) for m in machines]
    return float(np.mean(vals)) if vals else INF

def _avg_comm_cost(env, job: Dict[str, Any], u: str, v: str, machines: List[str]) -> float:
    if not machines: return 0.0
    data_size = _edge_data_size(job, u, v)
    vals = [_comm_cost_between_machines(env, src, dst, data_size) for src in machines for dst in machines]
    return float(np.mean(vals)) if vals else 0.0


def _estimate_action_components(obs: Dict[str, np.ndarray], env, action: int) -> Optional[Dict[str, float]]:
    decoded = _action_to_job_task_machine(env, int(action))
    if decoded is None: return None
    job_idx, task_name, target_machine = decoded
    job = env.active_jobs[job_idx]
    attrs = job["graph"].nodes[task_name]
    current_time = _current_time(env)
    ready_time = float(env._task_ready_time(job, task_name))
    snap = env._node_snapshot(target_machine, _current_slot(env, current_time), current_time)

    req_cpu = _task_req_cpu(attrs)
    runtime = _task_runtime(attrs)
    eff_cpu = max(0.1, float(snap.cpu_idle))
    exec_time = float(runtime * (req_cpu / eff_cpu))
    
    true_transfer_cost = float(env._task_transfer_cost(job, task_name, target_machine))
    role = str(getattr(snap, "role", _machine_role(env, target_machine)))

    # 与真实环境一致；若人为压低云端传输估计，多种启发式的 argmin 排序会高度重合。
    estimated_transfer = true_transfer_cost

    queue_delay = max(0.0, float(env.machine_available_time.get(str(target_machine), current_time)) - max(current_time, ready_time))
    estimated_queue = queue_delay * 0.85
    
    # Heuristic 认为的完成时间
    estimated_start = max(current_time, ready_time) + estimated_queue
    finish_time = estimated_start + exec_time + estimated_transfer
    response_time = finish_time - ready_time

    task_idx, node_idx = _decode_pair_action(int(action), env.max_nodes)
    match = float(obs["pair_interaction_x"][task_idx, node_idx, 2]) if "pair_interaction_x" in obs and task_idx < obs["pair_interaction_x"].shape[0] and node_idx < obs["pair_interaction_x"].shape[1] else 0.0

    compute_power = float(getattr(env, "role_power", {}).get(role, 1.0))
    tx_power = float(getattr(env, "role_tx_power", {}).get(role, 0.2))
    energy_proxy = float(compute_power * exec_time * min(req_cpu, eff_cpu) + tx_power * true_transfer_cost)

    slot = max(float(env.slot_size), 1.0)
    return {
        "job_idx": float(job_idx),
        "machine_idx": float(node_idx),
        "start": float(estimated_start),
        "finish": float(finish_time),
        "response": float(response_time),
        "ready": float(ready_time),
        "exec_time": float(exec_time),
        "transfer": float(estimated_transfer),
        "queue": float(estimated_queue),
        "energy_proxy": float(energy_proxy),
        "match": float(match),
        "queue_norm": float(estimated_queue / slot),
        "response_norm": float(response_time / slot),
        "transfer_norm": float(estimated_transfer / slot),
    }

def _estimate_action_finish_time(obs: Dict[str, np.ndarray], env, action: int) -> float:
    comp = _estimate_action_components(obs, env, int(action))
    return float(comp["finish"]) if comp else INF


def _action_start_finish_tuple(obs: Dict[str, np.ndarray], env, action: int) -> Tuple[float, float, int]:
    """(earliest_start, estimated_finish, node_idx).

    When ``env.max_ready_tasks == 1`` there is only one ready task per step. If
    FCFS and HEFT both break ties purely by minimum *finish* time, they collapse
    to the same mapping and baseline CSV metrics become identical. FCFS here
    prefers machines that become available first (FIFO start discipline), then
    finish time, then stable node index; HEFT prefers minimum finish (EFT),
    then start time, then node index.
    """
    decoded = _action_to_job_task_machine(env, int(action))
    if decoded is None:
        return (INF, INF, 10**9)
    job_idx, task_name, machine_id = decoded
    job = env.active_jobs[job_idx]
    max_nodes = int(env.max_nodes)
    _, node_idx = _decode_pair_action(int(action), max_nodes)
    current_time = _current_time(env)
    ready_time = float(env._task_ready_time(job, task_name))
    avail = float(env.machine_available_time.get(str(machine_id), current_time))
    start = float(max(current_time, ready_time, avail))
    finish = _estimate_action_finish_time(obs, env, int(action))
    return (start, finish, int(node_idx))


# ============================================================
# HEFT helpers
# ============================================================

def _topological_nodes(job: Dict[str, Any]) -> List[str]:
    g = job["graph"]
    try: return [str(n) for n in nx.topological_sort(g)]
    except Exception: return [str(n) for n in g.nodes]

def _compute_heft_upward_ranks(env, job: Dict[str, Any], machines: List[str]) -> Dict[str, float]:
    g = job["graph"]
    topo = _topological_nodes(job)
    current_time = _current_time(env)
    avg_comp = {n: _avg_comp_cost(env, job, n, machines, current_time=current_time) for n in topo}
    rank_u: Dict[str, float] = {}
    for n in reversed(topo):
        succs = [str(s) for s in g.successors(n)]
        if not succs:
            rank_u[n] = float(avg_comp.get(n, 0.0))
            continue
        max_succ = max(_avg_comm_cost(env, job, n, s, machines) + rank_u.get(s, 0.0) for s in succs)
        rank_u[n] = float(avg_comp.get(n, 0.0) + max_succ)
    return rank_u


# ============================================================
# Random
# ============================================================

class RandomPolicy:
    def __init__(self, task_top_k: int = 1, node_top_k: int = 3):
        self.task_top_k = max(1, int(task_top_k))
        self.node_top_k = max(1, int(node_top_k))

    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if "action_mask" not in obs or len(obs["action_mask"]) <= 0: return 0
        if env is None or not hasattr(env, "max_nodes"): return _safe_random_valid_action(obs)
        max_nodes = int(env.max_nodes)
        valid_compute = _valid_compute_actions(obs, max_nodes)
        if len(valid_compute) <= 0: return _fallback_action(obs, max_nodes)

        task_to_actions: Dict[int, List[int]] = {}
        for a in valid_compute:
            task_idx, _ = _decode_pair_action(int(a), max_nodes)
            task_to_actions.setdefault(int(task_idx), []).append(int(a))
        if not task_to_actions: return _fallback_action(obs, max_nodes)

        task_choices = sorted(task_to_actions.keys())[: self.task_top_k]
        chosen_task = int(np.random.choice(task_choices))
        node_actions = task_to_actions[chosen_task]

        scored = []
        for a in node_actions:
            comp = _estimate_action_components(obs, env, int(a))
            if comp is None:
                scored.append((float("inf"), int(a)))
                continue
            score = np.log1p(max(0.0, comp.get("response_norm", 0.0))) + 0.30 * np.log1p(max(0.0, comp.get("queue_norm", 0.0))) + 0.10 * np.log1p(max(0.0, comp.get("transfer_norm", 0.0))) - 0.05 * comp.get("match", 0.0)
            scored.append((float(score), int(a)))

        scored.sort(key=lambda x: x[0])
        top_actions = [a for _, a in scored[: self.node_top_k]]
        return int(np.random.choice(top_actions)) if top_actions else _fallback_action(obs, max_nodes)


# ============================================================
# FCFS (Restored to original logic, influenced by optimism bias)
# ============================================================

class FCFSPolicy:
    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if env is None: return _safe_random_valid_action(obs)
        max_nodes = env.max_nodes
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0: return _fallback_action(obs, max_nodes)
        candidates = list(getattr(env, "current_candidates", []))
        if not candidates: return _fallback_action(obs, max_nodes)

        best_task_idx, best_task_key = None, None
        for task_idx, (job_idx, task_name) in enumerate(candidates):
            job = env.active_jobs[job_idx]
            attrs = job["graph"].nodes[task_name]
            key = (float(job.get("arrival_time", 0.0)), float(env._task_ready_time(job, task_name)), float(attrs.get("depth", 0.0)), str(task_name))
            if best_task_key is None or key < best_task_key:
                best_task_key, best_task_idx = key, task_idx

        task_actions = [int(a) for a in valid if _decode_pair_action(int(a), max_nodes)[0] == best_task_idx]
        if not task_actions: return int(valid[0])
        return int(
            min(
                task_actions,
                key=lambda a: _action_start_finish_tuple(obs, env, int(a)),
            )
        )


# ============================================================
# LeastLoad (Restored to original multi-objective logic)
# ============================================================

class LeastLoadPolicy:
    def __init__(
        self, response_weight: float = 1.00, queue_weight: float = 0.45,
        exec_weight: float = 0.40, transfer_weight: float = 0.35,
        load_weight: float = 0.25, match_weight: float = 0.15,
    ):
        self.response_weight, self.queue_weight = response_weight, queue_weight
        self.exec_weight, self.transfer_weight = exec_weight, transfer_weight
        self.load_weight, self.match_weight = load_weight, match_weight

    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if env is None: return _safe_random_valid_action(obs)
        max_nodes = env.max_nodes
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0: return _fallback_action(obs, max_nodes)

        resource_x = obs.get("resource_x", None)
        best_action, best_score = None, INF

        for action in valid:
            task_idx, node_idx = _decode_pair_action(int(action), max_nodes)
            comp = _estimate_action_components(obs, env, int(action))
            if comp is None: continue

            load_penalty = 0.0
            if resource_x is not None and node_idx < resource_x.shape[1]:
                current_resource = resource_x[-1]
                cpu_used_ratio = max(0.0, 1.0 - float(current_resource[node_idx][5]))
                mem_used_ratio = max(0.0, 1.0 - float(current_resource[node_idx][6]))
                load_penalty = (0.35 * cpu_used_ratio + 0.25 * mem_used_ratio + 0.25 * max(0.0, float(current_resource[node_idx][7])) + 0.15 * max(0.0, float(current_resource[node_idx][12])))

            score = (self.response_weight * np.log1p(max(0.0, comp["response_norm"])) + self.queue_weight * np.log1p(max(0.0, comp["queue_norm"])) + self.exec_weight * np.log1p(max(0.0, comp["exec_time"] / max(float(getattr(env, "slot_size", 1.0)), 1.0))) + self.transfer_weight * np.log1p(max(0.0, comp["transfer_norm"])) + self.load_weight * load_penalty - self.match_weight * comp["match"])
            if score < best_score:
                best_score, best_action = float(score), int(action)
        return int(best_action) if best_action is not None else _fallback_action(obs, max_nodes)


# ============================================================
# Greedy (Restored to original multi-objective logic)
# ============================================================

class GreedyPolicy:
    def __init__(
        self, response_weight: float = 1.00, queue_weight: float = 0.35,
        transfer_weight: float = 0.30, energy_weight: float = 0.05, match_weight: float = 0.20,
    ):
        self.response_weight, self.queue_weight = response_weight, queue_weight
        self.transfer_weight, self.energy_weight, self.match_weight = transfer_weight, energy_weight, match_weight

    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if env is None: return _safe_random_valid_action(obs)
        max_nodes = env.max_nodes
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0: return _fallback_action(obs, max_nodes)

        best_action, best_score = None, INF
        slot = max(float(env.slot_size), 1.0)
        for action in valid:
            comp = _estimate_action_components(obs, env, int(action))
            if comp is None: continue
            score = (self.response_weight * np.log1p(max(0.0, comp["response_norm"])) + self.queue_weight * np.log1p(max(0.0, comp["queue_norm"])) + self.transfer_weight * np.log1p(max(0.0, comp["transfer_norm"])) + self.energy_weight * np.log1p(max(0.0, comp["energy_proxy"] / slot)) - self.match_weight * comp["match"])
            if score < best_score:
                best_score, best_action = float(score), int(action)
        return int(best_action) if best_action is not None else _fallback_action(obs, max_nodes)


# ============================================================
# Min-Min (optional heuristic; not included in default baseline comparison).
# When max_ready_tasks==1, per-step decisions often coincide with EFT; use FCFS/HEFT
# differentiation in FCFSPolicy/HEFTPolicy or raise max_ready_tasks for richer baselines.
class MinMinPolicy:
    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if env is None: return _safe_random_valid_action(obs)
        max_nodes = env.max_nodes
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0: return _fallback_action(obs, max_nodes)

        task_best: Dict[int, Tuple[float, int]] = {}
        for action in valid:
            task_idx, _ = _decode_pair_action(int(action), max_nodes)
            finish_time = _estimate_action_finish_time(obs, env, int(action))
            if task_idx not in task_best or finish_time < task_best[task_idx][0]:
                task_best[task_idx] = (finish_time, int(action))

        if not task_best: return _fallback_action(obs, max_nodes)
        _, best_action = min(task_best.values(), key=lambda x: x[0])
        return int(best_action)


# ============================================================
# HEFT
# ============================================================

class HEFTPolicy:
    def select_action(self, obs: Dict[str, np.ndarray], env=None) -> int:
        if env is None: return _safe_random_valid_action(obs)
        max_nodes = env.max_nodes
        valid = _valid_compute_actions(obs, max_nodes)
        if len(valid) == 0: return _fallback_action(obs, max_nodes)
        candidates = list(getattr(env, "current_candidates", []))
        if not candidates: return _fallback_action(obs, max_nodes)

        valid_by_task: Dict[int, List[int]] = {}
        for action in valid:
            task_idx, _ = _decode_pair_action(int(action), max_nodes)
            valid_by_task.setdefault(task_idx, []).append(int(action))

        rank_cache: Dict[int, Dict[str, float]] = {}
        machines_cache: Dict[int, List[str]] = {}
        best_task_idx, best_task_key = None, None

        for task_idx, (job_idx, task_name) in enumerate(candidates):
            if task_idx not in valid_by_task: continue
            job = env.active_jobs[job_idx]
            if job_idx not in machines_cache: machines_cache[job_idx] = _eligible_machine_ids_for_job(env, job)
            if job_idx not in rank_cache: rank_cache[job_idx] = _compute_heft_upward_ranks(env, job, machines_cache[job_idx])
            
            task_key = (-float(rank_cache[job_idx].get(str(task_name), 0.0)), float(env._task_ready_time(job, task_name)), str(task_name))
            if best_task_key is None or task_key < best_task_key:
                best_task_key, best_task_idx = task_key, task_idx

        if best_task_idx is None: return _fallback_action(obs, max_nodes)
        task_actions = valid_by_task.get(best_task_idx, [])
        if not task_actions: return _fallback_action(obs, max_nodes)

        def _heft_machine_key(a: int) -> Tuple[float, float, int]:
            s, f, n = _action_start_finish_tuple(obs, env, int(a))
            return (f, s, n)

        return int(min(task_actions, key=_heft_machine_key))
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Set, Any

import networkx as nx


TASK_RANDOM_PREFIX = "task_"


@dataclass
class ParsedTaskName:
    task_id: str
    numeric_id: int | None
    parents: List[int]
    is_random_name: bool


def parse_task_name(task_name: str) -> ParsedTaskName:
    task_name = str(task_name)
    if task_name.startswith(TASK_RANDOM_PREFIX):
        return ParsedTaskName(task_id=task_name, numeric_id=None, parents=[], is_random_name=True)
    nums = [int(x) for x in re.findall(r"\d+", task_name)]
    if not nums:
        return ParsedTaskName(task_id=task_name, numeric_id=None, parents=[], is_random_name=True)
    return ParsedTaskName(task_id=task_name, numeric_id=nums[0], parents=nums[1:], is_random_name=False)


def compute_node_depth(g: nx.DiGraph) -> Dict[str, int]:
    depth: Dict[str, int] = {}
    if g.number_of_nodes() == 0:
        return depth
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        depth[node] = 0 if not preds else 1 + max(depth[p] for p in preds)
    return depth


def edge_comm_cost(g: nx.DiGraph, u: str, v: str) -> float:
    data_size = float(g.edges[u, v].get("data_size", 1.0))
    return 1.0 + 0.1 * data_size


def compute_rank_up(g: nx.DiGraph) -> Dict[str, float]:
    rank: Dict[str, float] = {}
    for node in reversed(list(nx.topological_sort(g))):
        succs = list(g.successors(node))
        w = float(g.nodes[node].get("runtime_mean", g.nodes[node].get("duration", 1.0)))
        if not succs:
            rank[node] = w
        else:
            rank[node] = w + max(edge_comm_cost(g, node, s) + rank[s] for s in succs)
    return rank


def compute_rank_down(g: nx.DiGraph) -> Dict[str, float]:
    rank: Dict[str, float] = {}
    for node in nx.topological_sort(g):
        preds = list(g.predecessors(node))
        if not preds:
            rank[node] = 0.0
        else:
            rank[node] = max(
                rank[p] + float(g.nodes[p].get("runtime_mean", g.nodes[p].get("duration", 1.0))) + edge_comm_cost(g, p, node)
                for p in preds
            )
    return rank


def build_job_dag(task_rows: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    numeric_to_name: Dict[int, str] = {}
    for row in task_rows:
        parsed = parse_task_name(str(row["task_name"]))
        if parsed.numeric_id is not None:
            numeric_to_name[parsed.numeric_id] = str(row["task_name"])

    g = nx.DiGraph()
    for row in task_rows:
        task_name = str(row["task_name"])
        duration = max(1.0, float(row.get("runtime_mean", row.get("duration", 1.0)) or 1.0))
        g.add_node(
            task_name,
            task_name=task_name,
            instance_num=int(row.get("instance_num", 1) or 1),
            plan_cpu=float(row.get("plan_cpu", 0.0) or 0.0),
            plan_mem=float(row.get("plan_mem", 0.0) or 0.0),
            start_time=int(row.get("start_time", 0) or 0),
            end_time=int(row.get("end_time", 0) or 0),
            task_type=str(row.get("task_type", "batch")),
            duration=duration,
            runtime_mean=duration,
            cpu_real_peak=float(row.get("cpu_real_peak", row.get("plan_cpu", 0.0)) or 0.0),
            mem_real_peak=float(row.get("mem_real_peak", row.get("plan_mem", 0.0)) or 0.0),
        )

    for row in task_rows:
        task_name = str(row["task_name"])
        parsed = parse_task_name(task_name)
        for p in parsed.parents:
            if p in numeric_to_name:
                parent_name = numeric_to_name[p]
                if parent_name != task_name:
                    g.add_edge(parent_name, task_name, data_size=max(1.0, float(row.get("instance_num", 1) or 1)))

    if g.number_of_nodes() == 0:
        return None
    if not nx.is_directed_acyclic_graph(g):
        return None

    depth = compute_node_depth(g)
    rank_up = compute_rank_up(g)
    rank_down = compute_rank_down(g)

    nodes = []
    topo_order = list(nx.topological_sort(g))
    for node in topo_order:
        attrs = dict(g.nodes[node])
        attrs["depth"] = depth.get(node, 0)
        attrs["in_degree"] = int(g.in_degree(node))
        attrs["out_degree"] = int(g.out_degree(node))
        attrs["rank_up"] = float(rank_up.get(node, 0.0))
        attrs["rank_down"] = float(rank_down.get(node, 0.0))
        attrs["critical_score"] = float(attrs["rank_up"] + attrs["rank_down"])
        nodes.append(attrs)

    return {
        "job_name": str(task_rows[0].get("job_name", "unknown_job")),
        "nodes": nodes,
        "edges": [[u, v, float(g.edges[u, v].get("data_size", 1.0))] for u, v in g.edges()],
        "topo_order": topo_order,
        "num_nodes": len(nodes),
        "num_edges": g.number_of_edges(),
        "max_depth": max([attrs["depth"] for attrs in nodes] or [0]),
    }


def ready_nodes_static(g: nx.DiGraph, assigned: Set[str]) -> List[str]:
    out = []
    for node in g.nodes:
        if node in assigned:
            continue
        preds = list(g.predecessors(node))
        if all(p in assigned for p in preds):
            out.append(node)
    return out

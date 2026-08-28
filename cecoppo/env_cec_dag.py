from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import hashlib
from bisect import bisect_right

from cecoppo.dag_utils import ready_nodes_static
from cecoppo.utils import (
    ROLE_TO_ID,
    load_json,
    load_jsonl,
    variance,
    TRI_REF_MAKESPAN_SEC,
    TRI_REF_LOAD_BALANCE,
    tri_objective_normalized_terms,
    tri_objective_weighted_scalar,
    tri_refs_from_env_config,
    effective_makespan_for_tri,
)

# Process-level caches: avoid re-reading CSV/JSONL and rebuilding lookups for every eval episode.
_DATAFRAME_CACHE = {}
_JOB_CACHE = {}
_LOOKUP_CACHE = {}


@dataclass
class NodeSnapshot:
    machine_id: str
    role: str
    logical_domain: str
    cpu_cap: float
    mem_cap: float
    cpu_idle: float
    mem_idle: float
    pressure_score: float
    bg_cpu: float
    bg_mem: float
    net_in: float
    net_out: float
    queue_len: float


class CloudEdgeDagEnv:
    def __init__(self, data_dir: str | Path, config: Any):
        self.data_dir = Path(data_dir)
        self.config = config
        self.max_nodes = config.max_nodes
        self.max_dag_nodes = config.max_dag_nodes
        self.max_ready_tasks = getattr(config, "max_ready_tasks", 4)
        self.current_candidates = []
        
        self.include_defer_action = config.include_defer_action
        self.slot_size = config.slot_size
        self.history_len = getattr(config, "history_len", 5)
        self.episode_jobs = config.episode_jobs
        self.max_steps_per_episode = config.max_steps_per_episode
        self.reward_latency_weight = float(getattr(config, "reward_latency_weight", 2.0))
        self.reward_queue_weight = float(getattr(config, "reward_queue_weight", 0.30))
        self.reward_tail_weight = float(getattr(config, "reward_tail_weight", 0.05))
        self.reward_balance_weight = float(getattr(config, "reward_balance_weight", 0.25))
        self.reward_transfer_weight = float(getattr(config, "reward_transfer_weight", 0.05))
        self.reward_energy_weight = float(getattr(config, "reward_energy_weight", 0.01))
        self.reward_job_weight = float(getattr(config, "reward_job_weight", 1.0))
        self.reward_finish_bonus = float(getattr(config, "reward_finish_bonus", 3.0))
        self.split = getattr(config, "split", "train")
        self.rng = random.Random(config.seed)
        self.reward_makespan_weight = float(getattr(config, "reward_makespan_weight", 1.0))
        self.reward_slr_weight = float(getattr(config, "reward_slr_weight", 0.0))
        # ============================================================
        # 云-边-端异构差异增强参数
        # ============================================================
        # 计算能力缩放：云强、边中、端弱
        self.role_cpu_scale = {
            "cloud": getattr(config, "cloud_cpu_scale", 1.60),
            "edge": getattr(config, "edge_cpu_scale", 1.00),
            "end": getattr(config, "end_cpu_scale", 0.35),
        }

        self.role_mem_scale = {
            "cloud": getattr(config, "cloud_mem_scale", 1.40),
            "edge": getattr(config, "edge_mem_scale", 1.00),
            "end": getattr(config, "end_mem_scale", 0.45),
        }

        # 计算能耗系数：端设备能耗敏感，边缘适中，云端单位计算更高效
        self.role_power = {
             "cloud": getattr(config, "cloud_power", 0.65),
            "edge": getattr(config, "edge_power", 0.90),
            "end": getattr(config, "end_power", 1.35),
        }

        # 传输能耗系数：跨层传输越远，能耗越高
        self.role_tx_power = {
            "cloud": getattr(config, "cloud_tx_power", 0.30),
            "edge": getattr(config, "edge_tx_power", 0.22),
            "end": getattr(config, "end_tx_power", 0.18),
        }

        # 链路差异增强倍率
        self.end_cloud_latency_scale = getattr(config, "end_cloud_latency_scale", 2.50)
        self.end_edge_latency_scale = getattr(config, "end_edge_latency_scale", 1.30)
        self.edge_cloud_latency_scale = getattr(config, "edge_cloud_latency_scale", 1.60)

        self.end_cloud_bw_scale = getattr(config, "end_cloud_bw_scale", 0.45)
        self.end_edge_bw_scale = getattr(config, "end_edge_bw_scale", 0.80)
        self.edge_cloud_bw_scale = getattr(config, "edge_cloud_bw_scale", 0.75)

        
        data_key = str(self.data_dir.resolve())
        base_cache = _DATAFRAME_CACHE.get(data_key)
        if base_cache is None:
            base_cache = {
                "nodes_df": pd.read_csv(self.data_dir / "nodes.csv", low_memory=False),
                "node_state_df": pd.read_csv(self.data_dir / "node_state.csv", low_memory=False),
                "service_bg_df": (
                    pd.read_csv(self.data_dir / "service_bg.csv", low_memory=False)
                    if (self.data_dir / "service_bg.csv").exists()
                    else pd.DataFrame()
                ),
                "meta": load_json(self.data_dir / "meta.json"),
            }
            _DATAFRAME_CACHE[data_key] = base_cache

        self.nodes_df = base_cache["nodes_df"]
        self.node_state_df = base_cache["node_state_df"]
        self.service_bg_df = base_cache["service_bg_df"]
        self.meta = base_cache["meta"]

        job_key = (data_key, self.split)
        if job_key not in _JOB_CACHE:
            split_file = self.data_dir / f"jobs_{self.split}.jsonl"
            jobs = load_jsonl(
            split_file if split_file.exists() else self.data_dir / "jobs_all.jsonl"
            )

            def _job_arrival(j: Dict[str, Any]) -> float:
                return min(
                    [float(n.get("start_time", 0.0)) for n in j.get("nodes", [])] or [0.0]
                )

            jobs = sorted(jobs, key=_job_arrival)
            _JOB_CACHE[job_key] = jobs
        self.jobs = _JOB_CACHE[job_key]
        '''
        # 云边端全部作为候选计算节点；端节点具备弱计算能力
        self.compute_nodes_df = self.nodes_df[self.nodes_df["role"].isin(["end", "edge", "cloud"])].copy().reset_index(drop=True)
        self.compute_nodes_df["machine_id"] = self.compute_nodes_df["machine_id"].astype(str)

        if "attached_edge" in self.compute_nodes_df.columns:
            self.compute_nodes_df["attached_edge"] = (
                self.compute_nodes_df["attached_edge"]
                .fillna("")
                .astype(str)
            )

        if "logical_domain" in self.compute_nodes_df.columns:
            self.compute_nodes_df["logical_domain"] = (
                self.compute_nodes_df["logical_domain"]
                .fillna("unknown")
                .astype(str)
            )
        
        role_order = {"cloud": 0, "edge": 1, "end": 2}
        self.compute_nodes_df["_role_order"] = self.compute_nodes_df["role"].map(role_order).fillna(9).astype(int)
        self.compute_nodes_df = self.compute_nodes_df.sort_values(["_role_order", "machine_id"]).drop(columns=["_role_order"]).reset_index(drop=True)
        '''
        # ============================================================
        # 按配置严格选择云、边、端节点数量
        # 数据集原始规模是 2 cloud + 8 edge + 80 end。
        # 如果 config 设置 2+6+30，这里会从原始数据中采样子拓扑。
        # ============================================================

        all_compute_df = self.nodes_df[
            self.nodes_df["role"].isin(["cloud", "edge", "end"])
        ].copy().reset_index(drop=True)

        all_compute_df["machine_id"] = all_compute_df["machine_id"].astype(str)
        all_compute_df["role"] = all_compute_df["role"].astype(str)

        if "attached_edge" in all_compute_df.columns:
            all_compute_df["attached_edge"] = (
                all_compute_df["attached_edge"]
                .fillna("")
                .astype(str)
            )
        else:
            all_compute_df["attached_edge"] = ""

        if "logical_domain" in all_compute_df.columns:
            all_compute_df["logical_domain"] = (
                all_compute_df["logical_domain"]
                .fillna("unknown")
                .astype(str)
            )
        else:
            all_compute_df["logical_domain"] = "unknown"

        num_cloud = int(getattr(config, "num_cloud_nodes", 2))
        num_edge = int(getattr(config, "num_edge_nodes", 8))
        num_end = int(getattr(config, "num_end_nodes", 80))
        include_end_compute = bool(getattr(config, "include_end_compute", True))

        cloud_df = (
            all_compute_df[all_compute_df["role"] == "cloud"]
            .sort_values("machine_id")
            .head(num_cloud)
        )

        edge_df = (
            all_compute_df[all_compute_df["role"] == "edge"]
            .sort_values("machine_id")
            .head(num_edge)
        )

        selected_edge_ids = set(edge_df["machine_id"].astype(str).tolist())

        if include_end_compute:
            end_all = (
                all_compute_df[all_compute_df["role"] == "end"]
                .sort_values("machine_id")
                .copy()
            )

            # 优先选择 attached_edge 属于已选 edge 的终端，保证端-边拓扑一致
            end_df = end_all[end_all["attached_edge"].isin(selected_edge_ids)].head(num_end)

            # 如果不够，再从剩余 end 中补足
            if len(end_df) < num_end:
                need = num_end - len(end_df)
                used_end_ids = set(end_df["machine_id"].astype(str).tolist())
                extra_end_df = end_all[
                    ~end_all["machine_id"].astype(str).isin(used_end_ids)
                ].head(need)
                end_df = pd.concat([end_df, extra_end_df], axis=0, ignore_index=True)
        else:
            end_df = all_compute_df.iloc[0:0].copy()

        self.compute_nodes_df = pd.concat(
            [cloud_df, edge_df, end_df],
            axis=0,
            ignore_index=True,
        )

        # 重新按 cloud -> edge -> end 排序
        role_order = {"cloud": 0, "edge": 1, "end": 2}
        self.compute_nodes_df["_role_order"] = (
            self.compute_nodes_df["role"].map(role_order).fillna(9).astype(int)
        )
        self.compute_nodes_df = (
            self.compute_nodes_df
            .sort_values(["_role_order", "machine_id"])
            .drop(columns=["_role_order"])
            .reset_index(drop=True)
        )

        # 强制同步 max_nodes，避免 config.max_nodes 写错
        self.max_nodes = int(len(self.compute_nodes_df))
        self.config.max_nodes = self.max_nodes
        if bool(getattr(self.config, "verbose_env", False)):
            print(
                "[Env Nodes] "
                f"cloud={int((self.compute_nodes_df['role'] == 'cloud').sum())}, "
                f"edge={int((self.compute_nodes_df['role'] == 'edge').sum())}, "
                f"end={int((self.compute_nodes_df['role'] == 'end').sum())}, "
                f"total={self.max_nodes}"
            )

        
        self.compute_node_ids = self.compute_nodes_df["machine_id"].astype(str).tolist()
        self.cloud_node_ids = self.compute_nodes_df[self.compute_nodes_df["role"] == "cloud"]["machine_id"].astype(str).tolist()
        self.edge_node_ids = self.compute_nodes_df[self.compute_nodes_df["role"] == "edge"]["machine_id"].astype(str).tolist()
        self.end_node_ids = self.compute_nodes_df[self.compute_nodes_df["role"] == "end"]["machine_id"].astype(str).tolist()
        self.machine_static = self.compute_nodes_df.set_index("machine_id").to_dict("index")

        # ============================================================
        # Hierarchical cloud-edge-end topology
        # ============================================================

        self.use_hierarchical_topology = bool(
            getattr(config, "use_hierarchical_topology", True)
        )

        # end -> attached edge
        self.end_to_edge: Dict[str, str] = {}

        if "attached_edge" in self.compute_nodes_df.columns:
            for _, row in self.compute_nodes_df.iterrows():
                if str(row.get("role", "")) == "end":
                    end_id = str(row["machine_id"])
                    edge_id = str(row.get("attached_edge", ""))

                    # 只保留当前已选择的 edge
                    if edge_id in set(self.edge_node_ids):
                        self.end_to_edge[end_id] = edge_id

        # edge -> end list
        self.edge_to_ends: Dict[str, List[str]] = {e: [] for e in self.edge_node_ids}
        for end_id, edge_id in self.end_to_edge.items():
            if edge_id in self.edge_to_ends:
                self.edge_to_ends[edge_id].append(end_id)

        # edge-edge 全互联
        self.edge_neighbors: Dict[str, List[str]] = {}
        for e in self.edge_node_ids:
            self.edge_neighbors[e] = [x for x in self.edge_node_ids if x != e]

        # cloud-cloud 全互联
        self.cloud_neighbors: Dict[str, List[str]] = {}
        for c in self.cloud_node_ids:
            self.cloud_neighbors[c] = [x for x in self.cloud_node_ids if x != c]

        # edge -> cloud
        # 如果数据集没有明确的 edge-cloud 连接关系，默认每个 edge 可连接所有 cloud
        self.edge_to_clouds: Dict[str, List[str]] = {
            e: list(self.cloud_node_ids) for e in self.edge_node_ids
        }

        # machine -> role
        self.machine_role: Dict[str, str] = {}
        for _, row in self.compute_nodes_df.iterrows():
            self.machine_role[str(row["machine_id"])] = str(row["role"])
        
        scaled_cpu_caps = []
        scaled_mem_caps = []

        for _, row in self.compute_nodes_df.iterrows():
            role = str(row.get("role", "edge"))
            scaled_cpu_caps.append(
                float(row.get("cpu_num", 0.0)) * self.role_cpu_scale.get(role, 1.0)
            )
            scaled_mem_caps.append(
                float(row.get("mem_size", 0.0)) * self.role_mem_scale.get(role, 1.0)
            )

        self.max_cpu_cap = max(max(scaled_cpu_caps) if scaled_cpu_caps else 1.0, 1.0)
        self.max_mem_cap = max(max(scaled_mem_caps) if scaled_mem_caps else 1.0, 1.0)
        
        # self.max_cpu_cap = max(float(self.compute_nodes_df["cpu_num"].max()), 1.0)
        # self.max_mem_cap = max(float(self.compute_nodes_df["mem_size"].max()), 1.0)
        self.domain_to_idx = {d: i for i, d in enumerate(sorted(self.compute_nodes_df["logical_domain"].astype(str).unique().tolist()))}
        self.max_domain_idx = max(len(self.domain_to_idx), 1)

        lookup_cache = _LOOKUP_CACHE.get(data_key)
        if lookup_cache is None:
            lookup_cache = {
                "node_state_lookup": self._build_node_state_lookup(),
                "machine_slots": self._build_machine_slots(),
            }
            _LOOKUP_CACHE[data_key] = lookup_cache
        self.node_state_lookup = lookup_cache["node_state_lookup"]
        self.machine_slots = lookup_cache["machine_slots"]
        self.delay_matrix = self._build_delay_matrix()
        self._selected_nodes_cache = [str(m) for m in self.compute_node_ids[: self.max_nodes]]
        self._resource_adj_template, self._resource_edge_attr_template = self._build_resource_graph_templates()
        self._cached_obs = None
        self._cached_candidates = []

        self.current_slot = int(self.meta.get("min_slot", 0))
        self.current_time = float(self.current_slot * self.slot_size)
        self.active_jobs: List[Dict[str, Any]] = []
        self.finished_jobs = 0
        self.total_jobs = 0
        self.steps = 0
        self.done = False
        self.machine_available_time: Dict[str, float] = {}
        self.task_response_times: List[float] = []
        self.job_completion_times: List[float] = []
        self.task_energy: List[float] = []
        self.transfer_costs: List[float] = []
        self.load_balance_history: List[float] = []
        self.machine_busy_time: Dict[str, float] = {}
        # Paper-style memory load accumulator: sum over tasks of
        # (task memory demand / processor memory capacity) * execution time.
        self.machine_mem_usage_time: Dict[str, float] = {}
        self.episode_start_time: float = self.current_time

    def _build_node_state_lookup(self) -> Dict[Tuple[str, int], Dict[str, float]]:
        #把节点时序状态转成 dict 查询表。

        #原 notebook 使用 DataFrame.iterrows() 遍历 18 万行以上数据，初始化环境会非常慢。
        #这里改成 records/zip 方式，速度更稳定。
        
        lookup: Dict[Tuple[str, int], Dict[str, float]] = {}
        bg_lookup: Dict[Tuple[str, int], Tuple[float, float]] = {}

        if not self.service_bg_df.empty:
            bg_df = self.service_bg_df.fillna(0.0)
            for mid, slot, cpu, mem in zip(
                bg_df["machine_id"].astype(str),
                bg_df["slot"].astype(int),
                bg_df.get("container_cpu_used", pd.Series(0.0, index=bg_df.index)),
                bg_df.get("container_mem_used", pd.Series(0.0, index=bg_df.index)),
            ):
                bg_lookup[(str(mid), int(slot))] = (float(cpu), float(mem))

        state_df = self.node_state_df.fillna(0.0)
        for rec in state_df.to_dict("records"):
            key = (str(rec["machine_id"]), int(rec["slot"]))
            cpu_bg, mem_bg = bg_lookup.get(key, (0.0, 0.0))
            rec["container_cpu_used"] = float(cpu_bg)
            rec["container_mem_used"] = float(mem_bg)
            lookup[key] = rec
        return lookup

    def _build_machine_slots(self) -> Dict[str, List[int]]:
        slots: Dict[str, List[int]] = {}
        for machine_id, g in self.node_state_df.groupby("machine_id"):
            slots[str(machine_id)] = sorted(int(s) for s in g["slot"].tolist())
        return slots

    def _lookup_state(self, machine_id: str, slot: int) -> Dict[str, float]:
        machine_id = str(machine_id)
        slot = int(slot)
        key = (machine_id, slot)
        if key in self.node_state_lookup:
            return self.node_state_lookup[key]
        slots = self.machine_slots.get(machine_id, [])
        if not slots:
            return {}
        idx = bisect_right(slots, slot) - 1
        if idx >= 0:
            return self.node_state_lookup.get((machine_id, slots[idx]), {})
        return self.node_state_lookup.get((machine_id, slots[0]), {})

    def _build_delay_matrix(self) -> Dict[Tuple[str, str], Tuple[float, float, float]]:
        #构造云-边-端三层链路，返回 (bandwidth_MBps, latency_s, jitter_s)。

        #增强差异：
        #- 端-端：带宽低、时延高；
        #- 端-边：低时延，但带宽有限；
        #- 边-云：带宽较高，但跨层时延明显；
        #- 端-云：最差链路，需要经过边缘转发，时延和传输成本最高；
        #- 云-云：最高带宽、最低时延；
        #- 边-边：同域较优，跨域较差。
        
        delays: Dict[Tuple[str, str], Tuple[float, float, float]] = {}
        records = self.compute_nodes_df.to_dict("records")

        for src in records:
            for dst in records:
                src_id = str(src["machine_id"])
                dst_id = str(dst["machine_id"])
                src_role, dst_role = str(src["role"]), str(dst["role"])

                if src_id == dst_id:
                    delays[(src_id, dst_id)] = (1e9, 0.0, 0.0)
                    continue

                same_domain = str(src.get("logical_domain", "")) == str(dst.get("logical_domain", ""))

                # 端-边
                if {src_role, dst_role} == {"end", "edge"}:
                    attached = (
                        src_role == "end" and str(src.get("attached_edge", "")) == dst_id
                    ) or (
                        dst_role == "end" and str(dst.get("attached_edge", "")) == src_id
                    )

                    if attached:
                        bw, tau, jitter = 90.0, 0.014, 0.003
                    else:
                        bw, tau, jitter = 45.0, 0.030, 0.007

                    bw *= self.end_edge_bw_scale
                    tau *= self.end_edge_latency_scale
                    jitter *= self.end_edge_latency_scale

                # 边-云
                elif {src_role, dst_role} == {"edge", "cloud"}:
                    bw, tau, jitter = 500.0, 0.055, 0.010

                    bw *= self.edge_cloud_bw_scale
                    tau *= self.edge_cloud_latency_scale
                    jitter *= self.edge_cloud_latency_scale

                # 端-云：显式增强为最远跨层链路
                elif {src_role, dst_role} == {"end", "cloud"}:
                    bw, tau, jitter = 160.0, 0.090, 0.018

                    bw *= self.end_cloud_bw_scale
                    tau *= self.end_cloud_latency_scale
                    jitter *= self.end_cloud_latency_scale

                # 边-边
                elif src_role == dst_role == "edge":
                    if same_domain:
                        bw, tau, jitter = 420.0, 0.010, 0.003
                    else:
                        bw, tau, jitter = 180.0, 0.030, 0.008

                # 云-云
                elif src_role == dst_role == "cloud":
                    bw, tau, jitter = 1200.0, 0.004, 0.001

                 # 端-端
                elif src_role == dst_role == "end":
                    if same_domain:
                        bw, tau, jitter = 25.0, 0.045, 0.012
                    else:
                        bw, tau, jitter = 12.0, 0.080, 0.020

                else:
                    bw, tau, jitter = 80.0, 0.050, 0.012

                delays[(src_id, dst_id)] = (
                    max(1e-6, float(bw)),
                    max(0.0, float(tau)),
                    max(0.0, float(jitter)),
                )

        return delays
    def _estimate_task_exec_time_for_machine(
        self,
        job: Dict[str, Any],
        task_name: str,
        machine_id: str,
    ) -> float:
        """
        估计 task 在指定 machine 上的执行时间。
        这个估计应当和 _execute_task() 里的 exec_time 公式保持一致。
        """
        graph = job.get("graph", None)
        attrs = {}

        if graph is not None and task_name in graph.nodes:
            attrs = dict(graph.nodes[task_name])

        req_cpu = max(
            float(attrs.get("cpu_real_peak", attrs.get("plan_cpu", 0.0))) / 100.0,
            1e-3,
        )

        runtime = max(
            1.0,
            float(attrs.get("runtime_mean", attrs.get("duration", 1.0))),
        )

        current_time = float(
            getattr(self, "current_time", self.current_slot * self.slot_size)
        )
        slot = int(current_time // max(self.slot_size, 1.0))

        snap = self._node_snapshot(str(machine_id), slot, current_time)
        eff_cpu = max(0.1, float(snap.cpu_idle))

        return float(runtime * (req_cpu / eff_cpu))
    '''
    def _classify_task_demand(
        self,
        job: Dict[str, Any],
        task_name: str,
    ) -> str:
        """
        根据任务在 end / edge 上的预计执行时间划分任务强度。

        small:
            来源 end 自己能较快完成

        medium:
            end 较慢，但 edge 能较快完成

        large:
            edge 也较慢，应交给 cloud
        """

        slot = float(getattr(self.config, "slot_size", 300.0))

        source_end = str(job.get("source_end", ""))
        source_edge = str(job.get("source_edge", ""))

        small_end_slots = float(getattr(self.config, "small_task_end_slots", 1.0))
        medium_edge_slots = float(getattr(self.config, "medium_task_edge_slots", 2.0))

        # 如果没有 source_end，保守地当作 medium
        if source_end not in self.end_node_ids:
            return "medium"

        end_time = self._estimate_task_exec_time_for_machine(
            job, task_name, source_end
        )

        # 1. end 自己能处理的小任务
        if end_time <= small_end_slots * slot:
            return "small"

        # 2. 看 edge 是否能处理
        candidate_edges = []

        if source_edge in self.edge_node_ids:
            candidate_edges.append(source_edge)

            if bool(getattr(self.config, "allow_edge_peer_for_medium", True)):
                candidate_edges.extend(self.edge_neighbors.get(source_edge, []))

        candidate_edges = list(dict.fromkeys(candidate_edges))

        if not candidate_edges:
            return "large"

        edge_times = [
            self._estimate_task_exec_time_for_machine(job, task_name, e)
            for e in candidate_edges
        ]
        best_edge_time = min(edge_times) if edge_times else float("inf")

        if best_edge_time <= medium_edge_slots * slot:
            return "medium"

        # 3. edge 也不适合，则交给 cloud
        return "large"
    '''
    def _classify_task_demand(
        self,
        job: Dict[str, Any],
        task_name: str,
    ) -> str:
        """
        将 task 分类为 small / medium / large。

        small:
            轻量任务，适合 source_end 本地执行。

        medium:
            中等任务，适合 source_edge 或其它 edge 执行。

        large:
            重任务，交给 cloud 执行。
        """

        slot = float(getattr(self.config, "slot_size", 300.0))

        source_end = str(job.get("source_end", job.get("origin_end_id", "")))
        source_edge = str(job.get("source_edge", job.get("origin_edge_id", "")))

        attrs = dict(job["graph"].nodes[task_name])

        req_cpu = max(
            float(attrs.get("cpu_real_peak", attrs.get("plan_cpu", 0.0))) / 100.0,
            1e-3,
        )

        req_mem = max(
            float(attrs.get("mem_real_peak", attrs.get("plan_mem", 0.0))),
            1e-3,
        )

        runtime = max(
            1.0,
            float(attrs.get("runtime_mean", attrs.get("duration", 1.0))),
        )

        # 任务自身强度，避免完全依赖当前 cpu_idle 导致所有任务被低估
        work_slots = runtime * req_cpu / max(slot, 1.0)

        small_work_slots = float(getattr(self.config, "small_task_work_slots", 0.15))
        large_work_slots = float(getattr(self.config, "large_task_work_slots", 0.80))

        # 1. 明显很重的任务，直接归为 large
        if work_slots >= large_work_slots or req_cpu >= float(getattr(self.config, "large_task_cpu", 1.0)):
            return "large"

        # 2. 明显很轻的任务，再检查 end 执行时间
        if source_end in self.end_node_ids:
            end_time = self._estimate_task_exec_time_for_machine(
                job,
                task_name,
                source_end,
            )

            small_end_slots = float(getattr(self.config, "small_task_end_slots", 0.20))

            if work_slots <= small_work_slots and end_time <= small_end_slots * slot:
                return "small"

        # 3. 中等任务：看 edge 是否能较快处理
        candidate_edges = []

        if source_edge in self.edge_node_ids:
            candidate_edges.append(source_edge)

            if bool(getattr(self.config, "allow_edge_peer_for_medium", True)):
                candidate_edges.extend(self.edge_neighbors.get(source_edge, []))

        candidate_edges = list(dict.fromkeys(candidate_edges))

        if candidate_edges:
            edge_times = [
                self._estimate_task_exec_time_for_machine(job, task_name, e)
                for e in candidate_edges
            ]

            best_edge_time = min(edge_times) if edge_times else float("inf")

            medium_edge_slots = float(getattr(self.config, "medium_task_edge_slots", 0.80))

            if best_edge_time <= medium_edge_slots * slot:
                return "medium"

        # 4. edge 也处理不快，则归为 large
        return "large"
    
    def _allowed_machines_for_task(
        self,
        job: Dict[str, Any],
        task_name: str,
    ) -> List[str]:
        """
        【修改版】解除严格的任务大小绑定，让调度算法自己决定去哪层！
        保留的基础物理约束：一个 DAG 只能在它的提交端(source_end)、边缘节点和云节点上执行，
        不能跑到别人的手机(其他 end)上越权执行。
        """

        # 如果在 config 里关闭了层级拓扑，则允许所有节点
        if not bool(getattr(self.config, "use_hierarchical_topology", True)):
            return list(self.compute_node_ids)

        source_end = str(job.get("source_end", job.get("origin_end_id", "")))
        allowed: List[str] = []

        # 1. 允许所有的云节点 (算力最强，但传输代价最大，让算法自己去权衡)
        if bool(getattr(self.config, "allow_all_cloud_for_large", True)):
            allowed.extend(self.cloud_node_ids)

        # 2. 允许所有的边缘节点 (算力适中，传输较近)
        allowed.extend(self.edge_node_ids)

        # 3. 端设备侧：只允许产生该任务的终端节点自己执行 (本地计算，无传输，但算力弱)
        if source_end in self.end_node_ids:
            allowed.append(source_end)

        # 去重，并且只保留当前 compute node 中存在的机器
        allowed_set = set(allowed)
        allowed = [
            m for m in self.compute_node_ids
            if m in allowed_set
        ]

        # 极端情况兜底
        if not allowed:
            if self.cloud_node_ids:
                allowed = [self.cloud_node_ids[0]]

        return allowed

    def _is_machine_allowed_for_task(
        self,
        job: Dict[str, Any],
        task_name: str,
        machine_id: str,
    ) -> bool:
        return str(machine_id) in set(
            self._allowed_machines_for_task(job, task_name)
        )

    def _build_resource_graph_templates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Precompute resource adjacency and static link attributes once per environment.

        These values depend only on the selected machines and delay_matrix, not on the
        current step, so rebuilding them inside every observation is unnecessary.
        """
        selected_nodes = self.compute_node_ids[: self.max_nodes]
        adj = np.ones((self.max_nodes, self.max_nodes), dtype=np.float32)
        edge_attr = np.zeros((self.max_nodes, self.max_nodes, 4), dtype=np.float32)

        for i, src in enumerate(selected_nodes):
            for j, dst in enumerate(selected_nodes):
                bw, tau, jitter = self.delay_matrix[(src, dst)]
                same_domain = float(
                    self.machine_static[src]["logical_domain"]
                    == self.machine_static[dst]["logical_domain"]
                )
                edge_attr[i, j] = np.array(
                    [
                        float(np.clip(bw / 1000.0, 0.0, 5.0)),
                        float(np.clip(tau / 0.05, 0.0, 10.0)),
                        float(np.clip(jitter / 0.01, 0.0, 10.0)),
                        same_domain,
                    ],
                    dtype=np.float32,
                )
        return adj, edge_attr

    def _stable_hash01(self, x: str) -> float:
        h = hashlib.md5(str(x).encode("utf-8")).hexdigest()
        return int(h[:8], 16) / float(0xFFFFFFFF)


    def _dynamic_cpu_factor(self, machine_id: str, current_time: float) -> float:
        if not getattr(self.config, "enable_dynamic_disturbance", False):
            return 1.0

        amp = float(getattr(self.config, "dynamic_cpu_amp", 0.35))
        slot_t = current_time / max(self.slot_size, 1.0)

        phase = 2.0 * np.pi * self._stable_hash01(machine_id)
        trend = 0.5 + 0.5 * np.sin(0.15 * slot_t + phase)

        factor = 1.0 - amp * trend
        return float(np.clip(factor, 0.15, 1.0))

    
    def _sample_jobs(self) -> List[Dict[str, Any]]:
        if len(self.jobs) <= self.episode_jobs:
            sampled = [copy.deepcopy(j) for j in self.jobs]
        else:
        # 前提：jobs 已按 start_time / arrival_time 排序；如果没有，建议初始化时排序一次
            start = self.rng.randint(0, len(self.jobs) - self.episode_jobs)
            sampled = [copy.deepcopy(j) for j in self.jobs[start:start + self.episode_jobs]]

        return sampled

    def _assign_job_sources(self) -> None:
        """
        给每个 DAG/job 分配一个来源 end 节点。

        source_end:
            产生该 DAG 的终端设备。

        source_edge:
            source_end 连接的边缘节点。
    
        为了兼容你旧代码里的 origin_end_id / origin_edge_id，
        这里也同步写入这两个字段。
        """
        valid_source_ends = [
            e for e in self.end_node_ids
            if e in self.end_to_edge
        ]

        if not valid_source_ends:
            valid_source_ends = list(self.end_node_ids)

        if not valid_source_ends:
            return

        for i, job in enumerate(self.active_jobs):
            old_source_end = str(job.get("source_end", ""))
    
            if old_source_end in valid_source_ends:
                source_end = old_source_end
            else:
                # validation 更稳定：用轮询，而不是随机
                source_end = valid_source_ends[i % len(valid_source_ends)]

            source_edge = self.end_to_edge.get(source_end, "")

            job["source_end"] = source_end
            job["source_edge"] = source_edge

                # 兼容旧字段
            job["origin_end_id"] = source_end
            job["origin_edge_id"] = source_edge

    def reset(self) -> Dict[str, np.ndarray]:
        self.current_slot = int(self.meta.get("min_slot", 0))
        self.current_time = float(self.current_slot * self.slot_size)
        self.active_jobs = self._sample_jobs()
        self._assign_job_sources()
        
        self.total_jobs = len(self.active_jobs)
        self.finished_jobs = 0
        self.steps = 0
        self.done = False
        self.machine_available_time = {str(mid): self.current_time for mid in self.compute_node_ids}
        self.task_response_times = []
        self.job_completion_times = []
        self.task_energy = []
        self.transfer_costs = []
        self.load_balance_history = []
        self.machine_busy_time = {str(mid): 0.0 for mid in self.compute_node_ids[: self.max_nodes]}
        self.machine_mem_usage_time = {str(mid): 0.0 for mid in self.compute_node_ids[: self.max_nodes]}

        for job in self.active_jobs:
            g = nx.DiGraph()
            for node in job["nodes"]:
                g.add_node(node["task_name"], **node)
            for e in job["edges"]:
                if len(e) == 3:
                    u, v, data_size = e
                else:
                    u, v = e
                    data_size = 1.0
                g.add_edge(u, v, data_size=float(data_size))
            job["graph"] = g
            job["assigned"] = set()
            job["task_start_time"] = {}
            job["task_finish_time"] = {}
            job["task_machine"] = {}
            # Per-DAG resource accounting used by the paper-style load balance metric.
            # The paper computes L_CPU and L_Mem for a single DAG schedule; in this
            # multi-workflow simulator we therefore compute those quantities per DAG
            # and then average over completed DAGs, instead of dividing episode-level
            # accumulated busy time by an average DAG makespan.
            job["task_exec_time"] = {}
            job["task_mem_usage_time"] = {}
            job["arrival_time"] = min([n.get("start_time", 0) for n in job["nodes"]] or [0])
            # 每个 DAG 由一个终端提交；该终端也可以作为弱计算节点执行轻量任务
            # source_end / source_edge 已经在 self._assign_job_sources() 中分配好
            job["origin_end_id"] = str(job.get("source_end", job.get("origin_end_id", "")))
            job["origin_edge_id"] = str(job.get("source_edge", job.get("origin_edge_id", "")))
            job["done"] = False
        # ------------------------------------------------------------
        # Paper-style static DAG mode
        # ------------------------------------------------------------
        # The reference paper evaluates static DAG scheduling: a DAG is available
        # at the beginning of its scheduling process, and makespan/SLR are not
        # polluted by trace-level inter-arrival gaps.  Keeping Alibaba raw arrival
        # offsets can make job_completion_time and SLR look almost constant and
        # unrealistically large.  Therefore the default is to treat all sampled
        # DAGs as available at t=0.  Set config.paper_static_arrivals=False if you
        # explicitly want dynamic multi-workflow arrival experiments.
        paper_static_arrivals = bool(getattr(self.config, "paper_static_arrivals", True))
        if paper_static_arrivals:
            for job in self.active_jobs:
                job["raw_arrival_time"] = float(
                    min([float(n.get("start_time", 0.0)) for n in job["nodes"]] or [0.0])
                )
                job["arrival_time"] = 0.0
        else:
            episode_base_arrival = min(
                [min([float(n.get("start_time", 0.0)) for n in job["nodes"]] or [0.0])
                 for job in self.active_jobs] or [0.0]
            )
            for job in self.active_jobs:
                raw_arrival = min([float(n.get("start_time", 0.0)) for n in job["nodes"]] or [0.0])
                job["raw_arrival_time"] = raw_arrival
                job["arrival_time"] = raw_arrival - episode_base_arrival

        self.current_time = 0.0
        self.current_slot = 0
        self.episode_start_time = 0.0
        self.episode_start_time = min(
            [float(job.get("arrival_time", self.current_time)) for job in self.active_jobs] or [self.current_time]
        )
        self.episode_last_finish_time = float(self.episode_start_time)
        self._cached_obs = None
        self._cached_candidates = []
        return self._get_obs()

    def _selected_nodes(self) -> List[str]:
        #当前参与 observation/action 的节点集合。
        return list(getattr(self, "_selected_nodes_cache", [str(m) for m in self.compute_node_ids[: self.max_nodes]]))


    def _episode_eligible_nodes(self) -> List[str]:
        
        #本 episode 中实际有资格参与调度的节点：
        #- 所有 cloud
        #- 所有 edge
        #- 每个 DAG 的 origin_end
        
        selected = set(self._selected_nodes())

        eligible = set()
        eligible.update(str(m) for m in self.cloud_node_ids)
        eligible.update(str(m) for m in self.edge_node_ids)
    
        for job in getattr(self, "active_jobs", []):
            source_end = str(job.get("source_end", job.get("origin_end_id", "")))
            if source_end:
                eligible.add(source_end)

        return [m for m in self._selected_nodes() if m in selected and m in eligible]


    def _safe_utilization(self, nodes: List[str], makespan: float) -> float:
        #给定节点集合上的资源利用率。
        if not nodes or makespan <= 1e-9:
            return 0.0

        busy = float(sum(self.machine_busy_time.get(str(m), 0.0) for m in nodes))
        util = busy / max(len(nodes) * makespan, 1e-9)
        return float(np.clip(util, 0.0, 1.0))
        
    def _machine_queue_len(self, machine_id: str, current_time: float) -> float:
        available_t = float(self.machine_available_time.get(str(machine_id), 0.0))
        if available_t <= current_time:
            return 0.0
        return max(0.0, (available_t - current_time) / max(self.slot_size, 1.0))

    def _node_snapshot(self, machine_id: str, slot: int, current_time: float) -> NodeSnapshot:
        static = self.machine_static[str(machine_id)]
        row = self._lookup_state(machine_id, slot)

        role = str(static.get("role", "edge"))
    
        raw_cpu_cap = float(static.get("cpu_num", 0.0))
        raw_mem_cap = float(static.get("mem_size", 0.0))

        cpu_scale = self.role_cpu_scale.get(role, 1.0)
        mem_scale = self.role_mem_scale.get(role, 1.0)

        # 云强、边中、端弱
        cpu_cap = raw_cpu_cap * cpu_scale
        mem_cap = raw_mem_cap * mem_scale

        raw_cpu_idle = float(row.get("cpu_idle", raw_cpu_cap))
        raw_mem_idle = float(row.get("mem_idle", raw_mem_cap))

        cpu_idle = raw_cpu_idle * cpu_scale
        mem_idle = raw_mem_idle * mem_scale
        cpu_idle *= self._dynamic_cpu_factor(machine_id, current_time)

        return NodeSnapshot(
            machine_id=str(machine_id),
            role=role,
            logical_domain=str(static.get("logical_domain", "unknown")),
            cpu_cap=cpu_cap,
            mem_cap=mem_cap,
            cpu_idle=max(0.0, cpu_idle),
            mem_idle=max(0.0, mem_idle),
            pressure_score=float(row.get("pressure_score", 0.0)),
            bg_cpu=float(row.get("container_cpu_used", 0.0)),
            bg_mem=float(row.get("container_mem_used", 0.0)),
            net_in=float(row.get("net_in_mean", 0.0)),
            net_out=float(row.get("net_out_mean", 0.0)),
            queue_len=self._machine_queue_len(machine_id, current_time),
        )
    def _task_ready_time(self, job: Dict[str, Any], task_name: str) -> float:
        preds = list(job["graph"].predecessors(task_name))
        if not preds:
            return float(job.get("arrival_time", 0.0))
        return max(float(job["task_finish_time"].get(p, job.get("arrival_time", 0.0))) for p in preds)

    def _select_current_task(self) -> Optional[Tuple[int, str]]:
        candidates = self._get_ready_candidates()
        if not candidates:
            return None
        return candidates[0]

    def _has_ready_task_now(self, current_time: float) -> bool:
        for job in self.active_jobs:
            if job.get("done", False):
                continue

            ready = ready_nodes_static(job["graph"], job["assigned"])
    
            for task_name in ready:
                ready_time = self._task_ready_time(job, task_name)
                if ready_time <= current_time + 1e-9:
                    return True

        return False


    def _advance_time_after_schedule(self, old_time: float, finish_time: float) -> None:
        # 如果当前时刻还有 ready task，不推进时间，允许继续在同一时刻调度并行任务
        if self._has_ready_task_now(old_time):
            self.current_time = old_time
            self.current_slot = int(self.current_time // max(self.slot_size, 1))
            return
    
        future_ready_times = []
        for job in self.active_jobs:
            if job.get("done", False):
                continue

            for task_name in ready_nodes_static(job["graph"], job["assigned"]):
                rt = self._task_ready_time(job, task_name)
                if rt > old_time + 1e-9:
                    future_ready_times.append(rt)

        future_machine_times = [
            float(t)
            for t in self.machine_available_time.values()
            if float(t) > old_time + 1e-9
        ]

        future_events = future_ready_times + future_machine_times

        self.current_time = min(future_events) if future_events else finish_time
        self.current_slot = int(self.current_time // max(self.slot_size, 1))
    
    def _get_ready_candidates(self) -> List[Tuple[int, str]]:
    
        #获取当前时刻真正可调度的 ready task。

        #修正点：
        #1. ready_nodes_static 只表示依赖关系上 ready，不代表时间上已经 ready；
        #2. 必须先过滤 ready_time <= current_time 的任务；
        #3. 如果当前没有任务 ready，才推进到最早 ready_time；
        #4. 不能因为高 critical_score 的未来任务而跳过当前可执行任务。
    
        all_candidates: List[Tuple[int, str, float, float, float]] = []

        for j_idx, job in enumerate(self.active_jobs):
            if job.get("done", False):
                continue

            ready = ready_nodes_static(job["graph"], job["assigned"])
    
            for task_name in ready:
                attrs = job["graph"].nodes[task_name]
                ready_time = float(self._task_ready_time(job, task_name))

                all_candidates.append(
                    (
                        j_idx,
                        task_name,
                        -float(attrs.get("critical_score", 0.0)),
                        ready_time,
                        float(attrs.get("depth", 0.0)),
                    )
                )

        if not all_candidates:
            return []

        now = float(getattr(self, "current_time", self.current_slot * self.slot_size))

        # 先找当前时刻已经 ready 的任务
        ready_now = [
            c for c in all_candidates
            if float(c[3]) <= now + 1e-9
        ]

        # 如果当前没有 ready task，才推进到最早 ready_time
        if not ready_now:
            earliest_ready = min(float(c[3]) for c in all_candidates)
            self.current_time = max(now, earliest_ready)
            now = self.current_time

            ready_now = [
                c for c in all_candidates
                if float(c[3]) <= now + 1e-9
            ]
        else:
            self.current_time = now

        self.current_slot = int(self.current_time // max(self.slot_size, 1))

        # 只在当前真正 ready 的任务中排序
        ready_now.sort(key=lambda x: (x[2], x[3], x[4]))

        ready_now = ready_now[: self.max_ready_tasks]

        return [(j_idx, task_name) for j_idx, task_name, _, _, _ in ready_now]
    
    @property
    def action_dim(self) -> int:
        base_dim = self.max_ready_tasks * self.max_nodes
        return base_dim + (1 if self.include_defer_action else 0)

    def _resource_temporal_features(self, current_time: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        selected_nodes = self._selected_nodes()
        L = self.history_len
        resource_x = np.zeros((L, self.max_nodes, 14), dtype=np.float32)
        resource_adj = np.broadcast_to(
            self._resource_adj_template,
            (L, self.max_nodes, self.max_nodes),
        ).copy()
        resource_edge_attr = np.broadcast_to(
            self._resource_edge_attr_template,
            (L, self.max_nodes, self.max_nodes, 4),
        ).copy()
        resource_time_attr = np.zeros((L, self.max_nodes, 3), dtype=np.float32)
        prev_cpu, prev_mem, prev_queue = {}, {}, {}
        for t_idx, slot in enumerate(range(self.current_slot - L + 1, self.current_slot + 1)):
            hist_slot = int(slot)
            hist_time = float(hist_slot * self.slot_size)
            
            for i, machine_id in enumerate(selected_nodes):
                
                snap = self._node_snapshot(machine_id, hist_slot, hist_time)
                role_onehot = np.zeros(3, dtype=np.float32)
                role_onehot[ROLE_TO_ID.get(snap.role, 1)] = 1.0
                
                cpu_cap_norm = snap.cpu_cap / max(self.max_cpu_cap, 1.0)
                mem_cap_norm = snap.mem_cap / max(self.max_mem_cap, 1.0)
                
                cpu_idle_ratio = float(np.clip(snap.cpu_idle / max(snap.cpu_cap, 1.0), 0.0, 1.5))
                mem_idle_ratio = float(np.clip(snap.mem_idle / max(snap.mem_cap, 1.0), 0.0, 1.5))

                pressure_norm = float(np.clip(snap.pressure_score / 100.0, 0.0, 2.0))
                bg_cpu_norm = float(np.clip(snap.bg_cpu / 100.0, 0.0, 2.0))
                bg_mem_norm = float(np.clip(snap.bg_mem / 100.0, 0.0, 2.0))
                net_in_norm = float(np.clip(snap.net_in / 100.0, 0.0, 2.0))
                net_out_norm = float(np.clip(snap.net_out / 100.0, 0.0, 2.0))
                queue_norm = float(np.clip(snap.queue_len / 10.0, 0.0, 2.0))
                
                domain_norm = float(self.domain_to_idx.get(snap.logical_domain, 0)) / max(self.max_domain_idx, 1)
                resource_x[t_idx, i] = np.array([role_onehot[0], role_onehot[1], role_onehot[2], cpu_cap_norm, mem_cap_norm, cpu_idle_ratio, mem_idle_ratio, pressure_norm, bg_cpu_norm, bg_mem_norm, net_in_norm, net_out_norm, queue_norm, domain_norm], dtype=np.float32)
                
                resource_time_attr[t_idx, i, 0] = cpu_idle_ratio - prev_cpu.get(machine_id, cpu_idle_ratio)
                resource_time_attr[t_idx, i, 1] = mem_idle_ratio - prev_mem.get(machine_id, mem_idle_ratio)
                resource_time_attr[t_idx, i, 2] = queue_norm - prev_queue.get(machine_id, queue_norm)
                
                prev_cpu[machine_id], prev_mem[machine_id], prev_queue[machine_id] = cpu_idle_ratio, mem_idle_ratio, queue_norm
            # resource_adj and resource_edge_attr are static templates copied above.
        return resource_x, resource_adj, resource_edge_attr, resource_time_attr

    def _remaining_task_graph(self, job: Dict[str, Any]) -> nx.DiGraph:
        remaining = [n for n in job["graph"].nodes if n not in job["assigned"]]
        return job["graph"].subgraph(remaining).copy()

    def _task_feature_vec(self, attrs: Dict[str, Any], ready_flag: float = 0.0) -> np.ndarray:
        return np.array([min(1.0, float(attrs.get("plan_cpu", 0.0)) / max(self.max_cpu_cap * 100.0, 1.0)), min(1.0, float(attrs.get("plan_mem", 0.0)) / max(self.max_mem_cap, 1.0)), min(2.0, float(attrs.get("runtime_mean", attrs.get("duration", 1.0))) / max(self.slot_size, 1.0)), min(1.0, float(attrs.get("in_degree", 0.0)) / max(self.max_dag_nodes, 1.0)), min(1.0, float(attrs.get("out_degree", 0.0)) / max(self.max_dag_nodes, 1.0)), min(1.0, float(attrs.get("depth", 0.0)) / max(self.max_dag_nodes, 1.0)), min(2.0, float(attrs.get("rank_up", 0.0)) / max(self.slot_size, 1.0)), min(2.0, float(attrs.get("rank_down", 0.0)) / max(self.slot_size, 1.0)), min(4.0, float(attrs.get("critical_score", 0.0)) / max(self.slot_size, 1.0)), ready_flag], dtype=np.float32)

    def _interaction_features(
        self,
        job: Dict[str, Any],
        task_attrs: Dict[str, Any],
        task_name: str,
        current_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        
        #针对一个 ready task，计算它与所有候选计算节点的交互特征。

        #返回：
        #- feats: shape = [max_nodes, 3]
         #   feats[i] 表示当前 task 放到第 i 个节点上的估计执行/传输/匹配特征
        #- node_mask: shape = [max_nodes]
          #  node_mask[i] = 1 表示当前 task 可以放到第 i 个节点执行

        #注意：
        #如果你使用联合动作：
        #action = ready_task_idx * max_nodes + node_idx
        #那么这里不要返回 self.action_dim 长度的 mask。
        #完整 action_mask 应该在 _interaction_features() 中构造。
        

        selected_nodes = self.compute_node_ids[: self.max_nodes]

        feats = np.zeros((self.max_nodes, 3), dtype=np.float32)
        node_mask = np.zeros((self.max_nodes,), dtype=np.float32)

        # Alibaba plan_cpu 中通常 100 表示 1 core
        req_cpu = max(
            float(task_attrs.get("cpu_real_peak", task_attrs.get("plan_cpu", 0.0))) / 100.0,
            1e-3,
        )

        req_mem = max(
            float(task_attrs.get("mem_real_peak", task_attrs.get("plan_mem", 0.0))),
            1e-3,
        )

        runtime = max(
            1.0,
            float(task_attrs.get("runtime_mean", task_attrs.get("duration", 1.0))),
        )

        source_end = str(job.get("source_end", job.get("origin_end_id", "")))
        source_edge = str(job.get("source_edge", job.get("origin_edge_id", "")))

        task_demand = self._classify_task_demand(job, task_name)
        allowed_machines = set(self._allowed_machines_for_task(job, task_name))

        for i, machine_id in enumerate(selected_nodes):
            machine_id = str(machine_id)
            snap = self._node_snapshot(machine_id, self.current_slot, current_time)

            # ============================================================
            # 1. 候选节点集合约束
            #    当前 DAG 只允许：
            #    - 提交终端 origin_end
            #    - 所有边缘节点
            #    - 所有云节点
            #    不允许其他终端参与这个 DAG 的计算
            # ============================================================
            # ============================================================
            # 层级拓扑约束：
            # small  -> source_end
            # medium -> source_edge / edge peers
            # large  -> cloud
            # ============================================================
            if bool(getattr(self.config, "use_hierarchical_topology", True)):
                if machine_id not in allowed_machines:
                    continue
            else:
                # 非层级模式下，仍然禁止其它 end 处理当前 DAG
                if snap.role == "end" and machine_id != source_end:
                    continue

            # ============================================================
            # 2. 计算排队时间
            # ============================================================
            queue_delay = max(
                0.0,
                float(self.machine_available_time.get(machine_id, current_time)) - current_time,
            )

            # ============================================================
            # 3. 节点可行性判断
            #    这里不要设置太严格，否则 valid action 太少，PPO 很难学。
            # ============================================================
            if snap.role == "cloud":
            # 云端算力强，允许更高负载
                cpu_ok = snap.cpu_idle >= req_cpu * 0.20
                mem_ok = snap.mem_idle >= req_mem * 0.15
                queue_ok = queue_delay <= 6.0 * self.slot_size
    
            elif snap.role == "edge":
            # 边缘节点是主要执行层
                cpu_ok = snap.cpu_idle >= req_cpu * 0.25
                mem_ok = snap.mem_idle >= req_mem * 0.20
                queue_ok = queue_delay <= 5.0 * self.slot_size

            else:
            # 终端节点较弱，只适合轻量任务
                cpu_ok = snap.cpu_idle >= req_cpu * 0.35
                mem_ok = snap.mem_idle >= req_mem * 0.25
                queue_ok = queue_delay <= 3.0 * self.slot_size

            # 可选：限制终端只执行较轻任务
            # 如果你发现终端执行过多重任务，可以打开这个限制
            # light_task = runtime <= 2.0 * self.slot_size and req_cpu <= 1.0
            # cpu_ok = cpu_ok and light_task

            if cpu_ok and mem_ok and queue_ok:
                node_mask[i] = 1.0

            # ============================================================
            # 4. 估计执行时间
            # ============================================================
            eff_cpu = max(0.1, float(snap.cpu_idle))

            est_exec = runtime * (req_cpu / eff_cpu) + queue_delay

            # ============================================================
            # 5. 估计传输时间
            # ============================================================
            est_trans = self._task_transfer_cost(job, task_name, machine_id)

            # ============================================================
            # 6. 资源匹配度
            # ============================================================
            cpu_match = min(1.0, float(snap.cpu_idle) / max(req_cpu, 1e-6))
            mem_match = min(1.0, float(snap.mem_idle) / max(req_mem, 1e-6))

            # 轻量任务在提交终端上执行，给一点本地性加成
            local_bonus = 0.0
            if machine_id == source_end:
                local_bonus += 0.15

            # 如果是提交终端绑定的边缘节点，也给一点 locality 加成
            if source_edge and machine_id == source_edge:
                local_bonus += 0.10

            # 云端适合重计算任务，重任务给一点加成
            heavy_task = runtime > 2.0 * self.slot_size or req_cpu > 1.0
            cloud_bonus = 0.10 if snap.role == "cloud" and heavy_task else 0.0

            match_score = (
                  0.45 * cpu_match
                + 0.35 * mem_match
                + local_bonus
                + cloud_bonus
            )

            # ============================================================
            # 7. 交互特征
            #    维度保持 3，不改 graph_encoder 就不会出错。
            # ============================================================
            feats[i] = np.array(
                [
                    min(6.0, np.log1p(est_exec / max(self.slot_size, 1.0))),
                    min(6.0, np.log1p(est_trans / max(self.slot_size, 1.0))),
                    min(2.0, match_score),
                ],
                dtype=np.float32,
            )

        # ============================================================
        # 8. 兜底机制
        #    如果所有节点都不可选，至少允许一个最优候选节点。
        #    否则 action_mask 全 0 会导致 PPO / baseline 出问题。
        # ============================================================
        if np.sum(node_mask) <= 0:
            best_idx = None
            best_score = 1e18

            for i, machine_id in enumerate(selected_nodes):
                machine_id = str(machine_id)
                snap = self._node_snapshot(machine_id, self.current_slot, current_time)

                if bool(getattr(self.config, "use_hierarchical_topology", True)):
                    if machine_id not in allowed_machines:
                        continue
                else:
                    if snap.role == "end" and machine_id != source_end:
                        continue

                est_exec_norm = float(feats[i, 0])
                est_trans_norm = float(feats[i, 1])
                match_score = float(feats[i, 2])

                score = est_exec_norm + est_trans_norm - 0.2 * match_score

                if score < best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                node_mask[best_idx] = 1.0

        return feats, node_mask

    def _dag_features(
       self,
       current: Optional[Tuple[int, str]],
       current_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
       dag_x = np.zeros((self.max_dag_nodes, 10), dtype=np.float32)
       dag_adj = np.zeros((self.max_dag_nodes, self.max_dag_nodes), dtype=np.float32)
       current_task_x = np.zeros((10,), dtype=np.float32)
       interaction_x = np.zeros((self.max_nodes, 3), dtype=np.float32) 
       node_mask = np.zeros((self.max_nodes,), dtype=np.float32)

       if current is None:
            return dag_x, dag_adj, current_task_x, interaction_x, node_mask

       job_idx, task_name = current
       job = self.active_jobs[job_idx]

       g_remain = self._remaining_task_graph(job)

       ordered_nodes = sorted(
            list(g_remain.nodes),
            key=lambda n: (
                -float(g_remain.nodes[n].get("critical_score", 0.0)),
                float(g_remain.nodes[n].get("depth", 0.0)),
            ),
        )[: self.max_dag_nodes]

       node_to_idx = {n: i for i, n in enumerate(ordered_nodes)}
       ready_set = set(ready_nodes_static(job["graph"], job["assigned"]))
    
       for n in ordered_nodes:
            attrs = dict(job["graph"].nodes[n])
            dag_x[node_to_idx[n]] = self._task_feature_vec(
                attrs,
                ready_flag=float(n in ready_set)
            )

       for u, v in g_remain.edges:
            if u in node_to_idx and v in node_to_idx:
                dag_adj[node_to_idx[u], node_to_idx[v]] = 1.0
                dag_adj[node_to_idx[v], node_to_idx[u]] = 1.0

       np.fill_diagonal(dag_adj, 1.0)

       attrs = dict(job["graph"].nodes[task_name])
       current_task_x = self._task_feature_vec(attrs, ready_flag=1.0)

       interaction_x, node_mask = self._interaction_features(
            job,
            attrs,
            task_name,
            current_time,
        )

       return dag_x, dag_adj, current_task_x, interaction_x, node_mask

    def _ready_task_features(
        self,
        candidates: List[Tuple[int, str]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        ready_task_x = np.zeros((self.max_ready_tasks, 10), dtype=np.float32)
        ready_task_mask = np.zeros((self.max_ready_tasks,), dtype=np.float32)

        for i, (job_idx, task_name) in enumerate(candidates[: self.max_ready_tasks]):
            job = self.active_jobs[job_idx]
            attrs = dict(job["graph"].nodes[task_name])
            ready_task_x[i] = self._task_feature_vec(attrs, ready_flag=1.0)
            ready_task_mask[i] = 1.0

        return ready_task_x, ready_task_mask

    def _pair_interaction_features(
        self,
        candidates: List[Tuple[int, str]],
        current_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    #为联合动作构造 pair_interaction_x 和完整 action_mask。

    #联合动作定义：
     #   action = ready_task_idx * max_nodes + node_idx

    #返回：
    #- pair_interaction_x: shape = [max_ready_tasks, max_nodes, 3]
    #- action_mask: shape = [max_ready_tasks * max_nodes + defer]
        pair_interaction_x = np.zeros(
            (self.max_ready_tasks, self.max_nodes, 3),
            dtype=np.float32,
        )
    
        action_mask = np.zeros((self.action_dim,), dtype=np.float32)

        for t_idx, (job_idx, task_name) in enumerate(candidates[: self.max_ready_tasks]):
            job = self.active_jobs[job_idx]
            attrs = job["graph"].nodes[task_name]

            interaction_x, node_mask = self._interaction_features(
                job,
                attrs,
                task_name,
                current_time,
            )

            pair_interaction_x[t_idx] = interaction_x

            for node_idx in range(self.max_nodes):
                if node_mask[node_idx] > 0:
                    action_idx = t_idx * self.max_nodes + node_idx
                    action_mask[action_idx] = 1.0

        if self.include_defer_action:
            pair_dim = self.max_ready_tasks * self.max_nodes
            has_valid_compute_action = np.any(action_mask[:pair_dim] > 0)

            if has_valid_compute_action:
                action_mask[-1] = 0.0
            else:
                action_mask[-1] = 1.0
 
        return pair_interaction_x, action_mask

    def _direct_transfer_time(
        self,
        src: str,
        dst: str,
        data_size: float,
    ) -> float:
        src = str(src)
        dst = str(dst)

        if src == dst:
            return 0.0

        bw, tau, _ = self.delay_matrix[(src, dst)]
        return float(data_size / max(bw, 1e-6) + tau)

    #层级路由函数
    def _hierarchical_transfer_time(
        self,
        src: str,
        dst: str,
        data_size: float,
    ) -> float:
        """
        按 cloud-edge-end 层级拓扑计算传输时间。

        end-end:
            不允许直接互传，返回大惩罚。

        end-edge:
            如果是 attached edge，直接传；
            如果是其它 edge，则 end -> attached edge -> peer edge。
    
        end-cloud:
            end -> attached edge -> cloud。

        edge-edge:
            允许。

        cloud-cloud:
            允许。

        cloud-end:
            cloud -> attached edge -> end。
        """
        src = str(src)
        dst = str(dst)

        if src == dst:
            return 0.0

        src_role = self.machine_role.get(src, "")
        dst_role = self.machine_role.get(dst, "")

        big = 1e9

        # end-end 不允许互相处理 / 互相直连
        if src_role == "end" and dst_role == "end":
            return big

        # edge-edge 允许
        if src_role == "edge" and dst_role == "edge":
            return self._direct_transfer_time(src, dst, data_size)

        # cloud-cloud 允许
        if src_role == "cloud" and dst_role == "cloud":
            return self._direct_transfer_time(src, dst, data_size)

        # edge-cloud / cloud-edge 允许
        if src_role == "edge" and dst_role == "cloud":
            return self._direct_transfer_time(src, dst, data_size)

        if src_role == "cloud" and dst_role == "edge":
            return self._direct_transfer_time(src, dst, data_size)

        # end -> edge
        if src_role == "end" and dst_role == "edge":
            src_edge = self.end_to_edge.get(src, "")

            if not src_edge:
                return big

            if dst == src_edge:
                return self._direct_transfer_time(src, dst, data_size)

            # end -> attached edge -> peer edge
            if dst in self.edge_node_ids:
                return (
                    self._direct_transfer_time(src, src_edge, data_size)
                    + self._direct_transfer_time(src_edge, dst, data_size)
                )

            return big

        # edge -> end
        if src_role == "edge" and dst_role == "end":
            dst_edge = self.end_to_edge.get(dst, "")

            if not dst_edge:
                return big

            if src == dst_edge:
                return self._direct_transfer_time(src, dst, data_size)

            if src in self.edge_node_ids:
                return (
                    self._direct_transfer_time(src, dst_edge, data_size)
                    + self._direct_transfer_time(dst_edge, dst, data_size)
                )

            return big

        # end -> cloud: end -> attached edge -> cloud
        if src_role == "end" and dst_role == "cloud":
            src_edge = self.end_to_edge.get(src, "")

            if not src_edge:
                return big

            return (
                self._direct_transfer_time(src, src_edge, data_size)
                + self._direct_transfer_time(src_edge, dst, data_size)
            )

        # cloud -> end: cloud -> attached edge -> end
        if src_role == "cloud" and dst_role == "end":
            dst_edge = self.end_to_edge.get(dst, "")

            if not dst_edge:
                return big

            return (
                self._direct_transfer_time(src, dst_edge, data_size)
                + self._direct_transfer_time(dst_edge, dst, data_size)
            )

        return big


    def _task_transfer_cost(
        self,
        job: Dict[str, Any],
        task_name: str,
        target_machine: str,
    ) -> float:
        preds = list(job["graph"].predecessors(task_name))
        target_machine = str(target_machine)
        total = 0.0

        # 无前驱任务：数据从 source_end 产生
        if not preds:
            origin = str(job.get("source_end", job.get("origin_end_id", "")))

            if origin and origin != target_machine:
                data_size = max(
                    1.0,
                    float(job["graph"].nodes[task_name].get("plan_mem", 1.0)),
            )

                if bool(getattr(self.config, "use_hierarchical_topology", True)):
                    return self._hierarchical_transfer_time(
                        origin,
                        target_machine,
                        data_size,
                    )

                return self._direct_transfer_time(origin, target_machine, data_size)

            return 0.0

        for p in preds:
            parent_machine = job["task_machine"].get(p)

            if parent_machine is None:
                continue

            parent_machine = str(parent_machine)

            if parent_machine == target_machine:
                continue

            data_size = float(
                job["graph"].edges[p, task_name].get("data_size", 1.0)
            )

            if bool(getattr(self.config, "use_hierarchical_topology", True)):
                total += self._hierarchical_transfer_time(
                    parent_machine,
                    target_machine,
                    data_size,
                )
            else:
                total += self._direct_transfer_time(
                    parent_machine,
                    target_machine,
                    data_size,
                )

        return float(total)

    def _paper_resource_load_vectors(
        self,
        nodes: Optional[List[str]] = None,
        makespan: Optional[float] = None,
        percent: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return paper-style CPU and memory load vectors.

        Following the paper objective f2(x)=L_CPU+L_Mem, we first build one
        resource-load vector per processor and then compute the variation across
        processors.

        CPU_k = accumulated CPU busy time on processor k / makespan
        MEM_k = accumulated normalized memory occupation time on k / makespan

        When percent=True, the vectors are expressed on a 0--100 resource-load
        scale before the variance is calculated.  This follows the paper-style
        normalized resource-load comparison and avoids misleadingly tiny values
        such as 0.0009 from variance on raw [0, 1] fractions.
        """
        if nodes is None:
            nodes = self._episode_eligible_nodes() or self._selected_nodes()
        if makespan is None:
            elapsed = float(max(getattr(self, "current_time", 0.0) - getattr(self, "episode_start_time", 0.0), 0.0))
            if elapsed <= 1e-9:
                elapsed = float(max(self.slot_size, 1.0))
            makespan = elapsed
        denom = max(float(makespan), 1e-9)
        scale = 100.0 if percent else 1.0
        cpu_load = np.array(
            [scale * self.machine_busy_time.get(str(m), 0.0) / denom for m in nodes],
            dtype=np.float64,
        )
        mem_load = np.array(
            [scale * self.machine_mem_usage_time.get(str(m), 0.0) / denom for m in nodes],
            dtype=np.float64,
        )
        return cpu_load, mem_load

    @staticmethod
    def _resource_variation(load_vec: np.ndarray) -> float:
        """Variation term used by L_CPU/L_Mem; smaller means better balance."""
        if load_vec is None or len(load_vec) <= 1:
            return 0.0
        return float(np.var(load_vec))

    def _compute_load_balance(self, current_time: float) -> float:
        """Paper-consistent load balance objective: L_CPU + L_Mem.

        L_CPU and L_Mem are variances of normalized CPU and memory load across
        processors.  The reported load vectors use a 0--100 resource-load scale;
        this affects only load_balance, not makespan or SLR.  Smaller is better.
        """
        nodes = self._episode_eligible_nodes() or self._selected_nodes()
        elapsed = float(max(current_time - getattr(self, "episode_start_time", 0.0), 0.0))
        if elapsed <= 1e-9:
            elapsed = float(max(self.slot_size, 1.0))
        cpu_load, mem_load = self._paper_resource_load_vectors(nodes, elapsed, percent=True)
        cpu_var = self._resource_variation(cpu_load)
        mem_var = self._resource_variation(mem_load)
        return float(cpu_var + mem_var)

    def _paper_load_balance_completed_job(self, job: Dict[str, Any]) -> float:
        """L_CPU + L_Mem for one completed DAG (aligned with get_episode_metrics)."""
        task_finish = job.get("task_finish_time", {})
        if not (job.get("done", False) and task_finish):
            return 0.0
        load_nodes = self._selected_nodes()
        if not load_nodes or len(load_nodes) <= 1:
            return 0.0
        finish = max(float(t) for t in task_finish.values())
        task_start = job.get("task_start_time", {})
        if task_start:
            local_start = min(float(t) for t in task_start.values())
        else:
            local_start = float(job.get("arrival_time", self.episode_start_time))
        job_makespan = max(1e-9, finish - local_start)
        cpu_time_by_node = {str(m): 0.0 for m in load_nodes}
        mem_time_by_node = {str(m): 0.0 for m in load_nodes}
        task_machine = job.get("task_machine", {})
        task_exec_time = job.get("task_exec_time", {})
        task_mem_usage_time = job.get("task_mem_usage_time", {})
        for task_name, machine_id in task_machine.items():
            m = str(machine_id)
            if m not in cpu_time_by_node:
                continue
            cpu_time_by_node[m] += float(task_exec_time.get(task_name, 0.0))
            mem_time_by_node[m] += float(task_mem_usage_time.get(task_name, 0.0))
        cpu_vec = np.array(
            [100.0 * cpu_time_by_node[str(m)] / job_makespan for m in load_nodes],
            dtype=np.float64,
        )
        mem_vec = np.array(
            [100.0 * mem_time_by_node[str(m)] / job_makespan for m in load_nodes],
            dtype=np.float64,
        )
        if len(cpu_vec) <= 1:
            return 0.0
        return float(self._resource_variation(cpu_vec) + self._resource_variation(mem_vec))

    def _global_features(self, current_time: float) -> np.ndarray:
        selected_nodes = self._episode_eligible_nodes()

        if not selected_nodes:
            selected_nodes = self._selected_nodes()

        cpu_utils, link_utils = [], []

        for machine_id in selected_nodes:
            snap = self._node_snapshot(machine_id, self.current_slot, current_time)
            cpu_utils.append(
                1.0 - min(1.0, snap.cpu_idle / max(snap.cpu_cap, 1.0))
            )

        for src in selected_nodes:
            for dst in selected_nodes:
                _, _, jitter = self.delay_matrix[(src, dst)]
                link_utils.append(float(np.clip(jitter / 0.01, 0.0, 10.0)))

        ready_count = sum(
             len(ready_nodes_static(job["graph"], job["assigned"]))
            for job in self.active_jobs
            if not job.get("done", False)
        )

        avg_resp = float(np.mean(self.task_response_times)) if self.task_response_times else 0.0

        ready_norm = min(
            2.0,
            ready_count / max(self.max_ready_tasks, 1),
        )

        return np.array(
            [
                self.current_slot / max(self.meta.get("max_slot", self.current_slot + 1), 1),
                self.steps / max(self.max_steps_per_episode, 1),
                self.finished_jobs / max(self.total_jobs, 1),
                float(np.mean(cpu_utils)) if cpu_utils else 0.0,
                self._compute_load_balance(current_time),
                float(np.mean(link_utils)) if link_utils else 0.0,
                ready_norm,
                min(2.0, avg_resp / max(self.slot_size, 1.0)),
            ],
            dtype=np.float32,
        )
        
    def _get_obs(self) -> Dict[str, np.ndarray]:
        candidates = self._get_ready_candidates()
        self.current_candidates = candidates

        current_time = float(
            getattr(self, "current_time", self.current_slot * self.slot_size)
        )

        resource_x, resource_adj, resource_edge_attr, resource_time_attr = (
            self._resource_temporal_features(current_time)
        )

        # DAG 图仍然用第一个 ready task 所属 DAG 来构造，保证兼容原来的 DAG 编码
        if candidates:
            first_current = candidates[0]
        else:
            first_current = None

        dag_x, dag_adj, current_task_x, interaction_x, _ = self._dag_features(
            first_current,
            current_time,
        )

        ready_task_x, ready_task_mask = self._ready_task_features(candidates)

        pair_interaction_x, action_mask = self._pair_interaction_features(
            candidates,
            current_time,
        )

        global_feat = self._global_features(current_time)

        obs = {
            "resource_x": resource_x,
            "resource_adj": resource_adj,
            "resource_edge_attr": resource_edge_attr,
            "resource_time_attr": resource_time_attr,

            "dag_x": dag_x,
            "dag_adj": dag_adj,

            # 为了兼容旧模型，保留 current_task_x 和 interaction_x
            "current_task_x": current_task_x,
            "interaction_x": interaction_x,

            # 新增：多个 ready task
            "ready_task_x": ready_task_x,
            "ready_task_mask": ready_task_mask,
            "pair_interaction_x": pair_interaction_x,

            "global_x": global_feat,
            "action_mask": action_mask,
        }
        ll_blend = float(getattr(self.config, "stgnn_leastload_logit_blend", 0.0))
        if ll_blend > 0.0:
            obs["stgnn_leastload_logit_blend"] = np.array([ll_blend], dtype=np.float32)
        edge_bonus = float(getattr(self.config, "stgnn_edge_logit_bonus", 0.0))
        if edge_bonus > 0.0:
            obs["stgnn_edge_logit_bonus"] = np.array([edge_bonus], dtype=np.float32)
        end_pen = float(getattr(self.config, "stgnn_end_logit_penalty", 0.0))
        if end_pen > 0.0:
            obs["stgnn_end_logit_penalty"] = np.array([end_pen], dtype=np.float32)
        self._cached_obs = obs
        self._cached_candidates = list(candidates)
        return obs

    def _execute_task(self, job_idx: int, task_name: str, target_machine: str) -> Tuple[float, Dict[str, Any]]:
        job = self.active_jobs[job_idx]
        attrs = job["graph"].nodes[task_name]

        req_cpu = max(float(attrs.get("cpu_real_peak", attrs.get("plan_cpu", 0.0))) / 100.0, 1e-3)
        req_mem = max(float(attrs.get("mem_real_peak", attrs.get("plan_mem", 0.0))), 1e-3)
        current_time = float(getattr(self, "current_time", self.current_slot * self.slot_size))
        self.current_slot = int(current_time // max(self.slot_size, 1))
        snap = self._node_snapshot(target_machine, self.current_slot, current_time)

        runtime = max(1.0, float(attrs.get("runtime_mean", attrs.get("duration", 1.0))))
        eff_cpu = max(0.1, snap.cpu_idle)
        exec_time = runtime * (req_cpu / eff_cpu)
        transfer_cost = self._task_transfer_cost(job, task_name, target_machine)
        ready_time = self._task_ready_time(job, task_name)

        start_time = max(current_time, ready_time, float(self.machine_available_time.get(str(target_machine), current_time)))
        finish_time = start_time + exec_time + transfer_cost
        response_time = finish_time - ready_time
        queue_delay = max(0.0, start_time - max(current_time, ready_time))

        job["assigned"].add(task_name)
        job["task_start_time"][task_name] = start_time
        job["task_finish_time"][task_name] = finish_time
        job["task_machine"][task_name] = target_machine
        job.setdefault("task_exec_time", {})[task_name] = float(exec_time)
        self.machine_available_time[str(target_machine)] = finish_time
        self.task_response_times.append(float(response_time))
        self.transfer_costs.append(float(transfer_cost))
        self.machine_busy_time[str(target_machine)] = self.machine_busy_time.get(str(target_machine), 0.0) + float(exec_time)

        # Paper-style memory utilization load. The paper evaluates memory utilization
        # across processors; here we integrate normalized memory occupation over the
        # task execution interval so it is comparable with CPU usage time.
        mem_util = min(1.0, float(req_mem) / max(float(snap.mem_cap), 1e-9))
        mem_usage_time = float(mem_util * exec_time)
        self.machine_mem_usage_time[str(target_machine)] = (
            self.machine_mem_usage_time.get(str(target_machine), 0.0)
            + mem_usage_time
        )
        job.setdefault("task_mem_usage_time", {})[task_name] = mem_usage_time

        # 简化能耗模型：端侧更重视能耗，边缘适中，云端因规模效应略低；传输能耗与传输时延近似正相关。
        role = str(snap.role)

        # 计算能耗：端设备单位计算能耗高，云端单位计算更高效
        compute_power = self.role_power.get(role, 1.0)
        compute_energy = compute_power * exec_time * min(req_cpu, eff_cpu)

        # 传输能耗：如果跨层传输，传输成本更明显
        tx_power = self.role_tx_power.get(role, 0.2)

        origin = str(job.get("origin_end_id", ""))
        if origin and origin != str(target_machine):
            origin_role = str(self.machine_static.get(origin, {}).get("role", "end"))
            target_role = str(snap.role)

            # 跨层传输惩罚：端-云 > 端-边 > 边-云 > 同层
            if {origin_role, target_role} == {"end", "cloud"}:
                tx_scale = 2.50
            elif {origin_role, target_role} == {"end", "edge"}:
                tx_scale = 1.50
            elif {origin_role, target_role} == {"edge", "cloud"}:
                tx_scale = 1.80
            elif origin_role == target_role:
                tx_scale = 1.00
            else:
                tx_scale = 1.30
        else:
            tx_scale = 0.30

        transfer_energy = tx_power * transfer_cost * tx_scale
        total_energy = float(compute_energy + transfer_energy)
        self.task_energy.append(total_energy)

        # 事件驱动推进：推进到当前已知最早可发生变化的时间，避免资源状态长期停在初始 slot。
        self._advance_time_after_schedule(current_time, finish_time)
        #unfinished_ready_times = []
        #for j in self.active_jobs:
          #  if j.get("done", False):
          #      continue
           # for n in ready_nodes_static(j["graph"], j["assigned"]):
          #      unfinished_ready_times.append(self._task_ready_time(j, n))
        #future_events = [t for t in list(self.machine_available_time.values()) + unfinished_ready_times if t > current_time + 1e-9]
       # self.current_time = min(future_events) if future_events else finish_time
       # self.current_slot = int(self.current_time // max(self.slot_size, 1))

        load_balance = self._compute_load_balance(self.current_time)
        self.load_balance_history.append(load_balance)

         # --- 归一化基准 ---
        slot = max(self.slot_size, 1.0)
        runtime_ref = max(float(runtime), 1.0)

        # 用 task 自身 runtime 归一化，比除以 300 秒 slot 更敏感
        resp_penalty = np.log1p(response_time / runtime_ref)
        queue_penalty = np.log1p(queue_delay / runtime_ref)
        transfer_penalty = np.log1p(transfer_cost / max(slot * 0.1, 1.0))
        energy_penalty = np.log1p(total_energy / max(slot, 1.0))

        # load_balance 现在是论文式 L_CPU + L_Mem；用 log 压缩后再进入奖励，避免数值过大。
        balance_penalty = float(np.clip(np.log1p(max(load_balance, 0.0)) / np.log1p(100.0), 0.0, 2.0))
    
        # makespan proxy：每步惩罚当前 task 完成后扩展 episode span 的程度
        #span_slots = max(0.0, finish_time - self.episode_start_time) / slot
        #makespan_penalty = np.log1p(span_slots)
        prev_finish_time = float(getattr(self, "episode_last_finish_time", self.episode_start_time))

        old_span_slots = max(0.0, prev_finish_time - self.episode_start_time) / slot
        new_finish_time = max(prev_finish_time, float(finish_time))
        new_span_slots = max(0.0, new_finish_time - self.episode_start_time) / slot

        makespan_penalty = max(
            0.0,
            np.log1p(new_span_slots) - np.log1p(old_span_slots)
        )

        self.episode_last_finish_time = new_finish_time

        
        reward_mode = str(getattr(self.config, "reward_mode", "dense")).lower()
        sparse_reward = reward_mode in {"terminal_sparse", "sparse_terminal", "episode"}

        if sparse_reward:
            # Clean scheduling signal: small per-step reward; episode metrics
            # at end + optional dense-style balance proxy every step (STGNN vs MLP).
            reward = float(getattr(self.config, "reward_task_bonus", 0.05))
            w_step_bal = float(getattr(self.config, "reward_sparse_step_balance_weight", 0.0))
            if w_step_bal > 0.0:
                _, _, ref_lb = tri_refs_from_env_config(self.config)
                lb_ratio = float(load_balance) / max(float(ref_lb), 1e-6)
                step_bal_term = float(min(4.0, np.log1p(max(0.0, lb_ratio))))
                reward -= w_step_bal * step_bal_term
            w_ent = float(getattr(self.config, "reward_sparse_busy_entropy_weight", 0.0))
            if w_ent > 0.0:
                spread_nodes = self._episode_eligible_nodes() or self._selected_nodes()
                busy = np.array(
                    [self.machine_busy_time.get(str(m), 0.0) for m in spread_nodes],
                    dtype=np.float64,
                )
                if len(busy) > 1 and float(busy.sum()) > 1e-6:
                    p = busy / float(busy.sum())
                    p = p[p > 1e-12]
                    ent = float(-np.sum(p * np.log(p + 1e-12)))
                    max_ent = float(np.log(max(len(busy), 2)))
                    reward += w_ent * float(ent / max(max_ent, 1e-9))
        else:
            reward = (
                - self.reward_latency_weight  * resp_penalty
                - self.reward_queue_weight    * queue_penalty
                - self.reward_transfer_weight * transfer_penalty
                - self.reward_balance_weight  * balance_penalty
                - self.reward_energy_weight   * energy_penalty
                - self.reward_makespan_weight * makespan_penalty
            )

        finish_bonus = 0.0
        job_penalty = 0.0
        slr_penalty = 0.0
        finished = False
        job_completion_time = 0.0

        if len(job["assigned"]) == job["graph"].number_of_nodes():
            job["done"] = True
            self.finished_jobs += 1
            finished = True

            job_finish_time = max(job["task_finish_time"].values())
            job_arrival_time = float(job.get("arrival_time", self.episode_start_time))
            job_completion_time = float(job_finish_time - job_arrival_time)

            self.job_completion_times.append(job_completion_time)

            if sparse_reward:
                finish_bonus = float(getattr(self.config, "reward_dag_finish_bonus", 5.0))
                reward += finish_bonus
                w_slr_sparse = float(getattr(self.config, "reward_sparse_per_dag_slr_weight", 0.0))
                w_ms_sparse = float(getattr(self.config, "reward_sparse_per_dag_makespan_weight", 0.0))
                w_ld_sparse = float(getattr(self.config, "reward_sparse_per_dag_load_weight", 0.0))
                if finished and (w_slr_sparse > 0.0 or w_ms_sparse > 0.0 or w_ld_sparse > 0.0):
                    task_finish = job.get("task_finish_time", {})
                    task_start = job.get("task_start_time", {})
                    if task_finish:
                        finish_t = max(float(t) for t in task_finish.values())
                        if task_start:
                            local_start = min(float(t) for t in task_start.values())
                        else:
                            local_start = float(job.get("arrival_time", self.episode_start_time))
                        job_ms = max(0.0, finish_t - local_start)
                        if w_slr_sparse > 0.0:
                            cp_m = max(float(self._job_critical_path_min_runtime(job)), 1e-9)
                            j_slr = float(job_ms / cp_m)
                            reward -= w_slr_sparse * float(np.log1p(max(0.0, j_slr)))
                        if w_ms_sparse > 0.0:
                            ref = max(float(TRI_REF_MAKESPAN_SEC), 1e-6)
                            reward -= w_ms_sparse * float(np.log1p(max(0.0, job_ms / ref)))
                        if w_ld_sparse > 0.0:
                            lb_d = self._paper_load_balance_completed_job(job)
                            if lb_d > 0.0:
                                _, _, ref_lb = tri_refs_from_env_config(self.config)
                                lb_ratio = max(0.0, lb_d) / max(float(ref_lb), 1e-6)
                                reward -= w_ld_sparse * float(
                                    np.log1p(lb_ratio) + 0.45 * (lb_ratio ** 2)
                                )
            else:
                finish_bonus = float(getattr(self.config, "reward_finish_bonus", 1.0))

                job_ref = max(slot * 20.0, 1.0)
                job_penalty = np.log1p(job_completion_time / job_ref)

                # Terminal objective reward: directly penalize the paper SLR of the
                # completed DAG. Step-level latency rewards can be locally good but
                # still produce a poor final critical-path normalized schedule.
                cp_min = max(float(self._job_critical_path_min_runtime(job)), 1e-9)
                job_slr = float(job_completion_time / cp_min)
                slr_penalty = np.log1p(max(0.0, job_slr))

                reward += finish_bonus
                reward -= self.reward_job_weight * job_penalty
                reward -= self.reward_slr_weight * slr_penalty
        '''
        # --- 1. 响应时间惩罚（针对avg_response_time）---
        norm_resp = response_time / slot
        # 线性惩罚，对大响应时间更敏感
        resp_penalty = min(5.0, norm_resp / 5.0)
    
        # --- 2. 队列惩罚 ---
        norm_queue = queue_delay / slot
        queue_penalty = min(2.0, norm_queue / 3.0)
    
        # --- 3. 传输惩罚 ---
        norm_transfer = transfer_cost / slot
        transfer_penalty = min(1.0, norm_transfer / 3.0)
    
        # --- 4. 能耗惩罚 ---
        norm_energy = total_energy / slot
        energy_penalty = min(1.0, norm_energy / 5.0)
    
        # --- 5. 负载均衡惩罚（load_balance_cv）---
        # 直接用当前时刻负载方差作为惩罚
        balance_penalty = min(3.0, load_balance)
    
        # --- 6. Makespan 惩罚（step-level proxy）---
        # 用 finish_time - episode_start_time 的归一化值
        current_span = max(0.0, finish_time - self.episode_start_time) / slot
        makespan_penalty = min(5.0, current_span / 100.0)
    
        # --- 组合奖励 ---
        reward = (
            - self.reward_latency_weight  * resp_penalty        # 响应时间（最重要）
            - self.reward_balance_weight  * balance_penalty      # 负载均衡
            - self.reward_queue_weight    * queue_penalty        # 排队时间
            - self.reward_transfer_weight * transfer_penalty     # 传输成本
            - self.reward_energy_weight   * energy_penalty       # 能耗
            - 0.50                        * makespan_penalty     # makespan proxy
        )
         # --- Job完成奖励 ---
        finish_bonus = 0.0
        finished = False
        job_completion_time = 0.0
        
        '''
        raw_reward = float(reward)
        #reward = float(np.clip(reward, -20.0, 10.0))
        reward_clip_min = float(getattr(self.config, "reward_clip_min", -20.0))
        reward_clip_max = float(getattr(self.config, "reward_clip_max", 10.0))

        reward = float(np.clip(raw_reward, reward_clip_min, reward_clip_max))

        was_clipped_low = raw_reward <= reward_clip_min
        was_clipped_high = raw_reward >= reward_clip_max

        task_demand = self._classify_task_demand(job, task_name)
        allowed_machines = self._allowed_machines_for_task(job, task_name)

        info = {
            "job_name": str(job["job_name"]),
            "task_name": str(task_name),

            "source_end": str(job.get("source_end", job.get("origin_end_id", ""))),
            "source_edge": str(job.get("source_edge", job.get("origin_edge_id", ""))),
            "task_demand": str(task_demand),
            "allowed_machine_count": float(len(allowed_machines)),

            "target_machine": str(target_machine),
            "target_role": str(snap.role),

            "ready_time": float(ready_time),
            "start_time": float(start_time),
            "finish_time": float(finish_time),
            "exec_time": float(exec_time),
            "transfer_cost": float(transfer_cost),
            "task_response_time": float(response_time),
            "job_completion_time": float(job_completion_time),
            "energy": float(total_energy),
            "load_balance": float(load_balance),
            "queue_delay": float(queue_delay),

            "reward": float(reward),
            "raw_reward": float(raw_reward),
            "was_clipped_low": bool(was_clipped_low),
            "was_clipped_high": bool(was_clipped_high),

            "resp_penalty": float(resp_penalty),
            "queue_penalty": float(queue_penalty),
            "transfer_penalty": float(transfer_penalty),
            "energy_penalty": float(energy_penalty),
            "balance_penalty": float(balance_penalty),
            "makespan_penalty": float(makespan_penalty),
            "finish_bonus": float(finish_bonus),
            "job_penalty": float(job_penalty),
            
            "avg_task_response_time": float(np.mean(self.task_response_times)) if self.task_response_times else 0.0,
            "avg_job_completion_time": float(np.mean(self.job_completion_times)) if self.job_completion_times else 0.0,
            "avg_load_balance": float(np.mean(self.load_balance_history)) if self.load_balance_history else 0.0, 
        }
        return reward, info

    def _terminal_objective_reward(self) -> Tuple[float, Dict[str, float]]:
        """Episode-level objective reward used by terminal_sparse mode.

        The final scheduling metrics are only well-defined after all jobs are
        completed (or the episode is truncated). Adding this reward to the last
        transition lets GAE propagate the objective backward through the episode
        without using a non-standard last_value target.
        """
        metrics = self.get_episode_metrics()

        completion_ratio = float(metrics.get("completion_ratio", 0.0))
        unfinished_gap = 1.0 - completion_ratio

        mm = float(metrics.get("makespan", metrics.get("raw_makespan", 0.0)))
        mx = float(metrics.get("max_dag_makespan", mm))
        blend = float(getattr(self.config, "reward_terminal_tri_makespan_max_blend", 0.0))
        makespan_sec = effective_makespan_for_tri(mm, mx, blend)
        slr_raw = float(metrics.get("SLR", metrics.get("slr", 0.0)))
        load_raw = float(metrics.get("load_balance", 0.0))

        ref_m, ref_s, ref_lb = tri_refs_from_env_config(self.config)
        m_norm, s_norm, lb_norm = tri_objective_normalized_terms(
            makespan_sec, slr_raw, load_raw,
            ref_makespan_sec=ref_m, ref_slr=ref_s, ref_load_balance=ref_lb,
        )
        wm = float(getattr(self.config, "reward_terminal_tri_w_makespan", 0.27))
        ws = float(getattr(self.config, "reward_terminal_tri_w_slr", 0.58))
        wlb = float(getattr(self.config, "reward_terminal_tri_w_load", 0.15))
        tri_core = tri_objective_weighted_scalar(
            makespan_sec, slr_raw, load_raw, wm, ws, wlb,
            ref_makespan_sec=ref_m, ref_slr=ref_s, ref_load_balance=ref_lb,
        )
        tri_penalty = float(getattr(self.config, "reward_terminal_tri_penalty", 2.75))

        terminal_reward = (
            float(getattr(self.config, "reward_terminal_completion_weight", 10.0)) * completion_ratio
            - tri_penalty * tri_core
            - float(getattr(self.config, "reward_terminal_unfinished_weight", 10.0)) * unfinished_gap
        )

        clip_min = float(getattr(self.config, "reward_terminal_clip_min", -50.0))
        clip_max = float(getattr(self.config, "reward_terminal_clip_max", 20.0))
        terminal_reward = float(np.clip(terminal_reward, clip_min, clip_max))

        return terminal_reward, {
            "terminal_completion_ratio": completion_ratio,
            "terminal_tri_m": m_norm,
            "terminal_tri_s": s_norm,
            "terminal_tri_lb": lb_norm,
            "terminal_tri_core": tri_core,
            "terminal_makespan_slots": float(makespan_sec)
            / max(float(getattr(self, "slot_size", 1.0)), 1.0),
            "terminal_log_slr": float(np.log1p(max(0.0, slr_raw))),
            "terminal_log_load_balance": float(np.log1p(max(0.0, load_raw))),
        }

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, Dict[str, Any]]:
        if self.done:
            return self._get_obs(), 0.0, True, {"msg": "episode already done"}

        obs_before = getattr(self, "_cached_obs", None)
        if obs_before is None:
            obs_before = self._get_obs()
        candidates = list(getattr(self, "_cached_candidates", getattr(self, "current_candidates", [])))

        if not candidates:
            self.done = True

            if self.finished_jobs >= self.total_jobs:
                return self._get_obs(), 0.0, True, {
                    "msg": "all jobs finished",
                    "normal_done": True,
                    "finished_jobs": self.finished_jobs,
                    "total_jobs": self.total_jobs,
                }

            unfinished_ratio = 1.0 - float(self.finished_jobs / max(self.total_jobs, 1))
            if str(getattr(self.config, "reward_mode", "dense")).lower() in {"terminal_sparse", "sparse_terminal", "episode"}:
                reward, terminal_info = self._terminal_objective_reward()
                info = {
                    "msg": "no schedulable task but unfinished jobs remain",
                    "normal_done": False,
                    "finished_jobs": self.finished_jobs,
                    "total_jobs": self.total_jobs,
                    "unfinished_ratio": unfinished_ratio,
                    "terminal_objective_reward": reward,
                }
                info.update(terminal_info)
                return self._get_obs(), reward, True, info

            reward = float(np.clip(-5.0 * unfinished_ratio, -10.0, 0.0))

            return self._get_obs(), reward, True, {
                "msg": "no schedulable task but unfinished jobs remain",
                "normal_done": False,
                "finished_jobs": self.finished_jobs,
                "total_jobs": self.total_jobs,
                "unfinished_ratio": unfinished_ratio,
                "terminal_penalty": reward,
            }

        # 非法动作检查
        if (
            action < 0
            or action >= len(obs_before["action_mask"])
            or obs_before["action_mask"][action] <= 0
        ):
            self.steps += 1
            reward = -5.0
            info = {
                "invalid_action": True,
                "penalty": reward,
            }

            if self.steps >= self.max_steps_per_episode:
                self.done = True

            return self._get_obs(), reward, self.done, info

        # defer 动作
        if self.include_defer_action and action == self.action_dim - 1:
            self.steps += 1
            self.current_time = (
                float(getattr(self, "current_time", self.current_slot * self.slot_size))
                + self.slot_size
            )
            self.current_slot = int(self.current_time // max(self.slot_size, 1))

            reward = -0.5
            info = {
                "defer": True,
                "reason": "defer_action",
                "penalty": reward,
            }

            if self.steps >= self.max_steps_per_episode:
                self.done = True

            return self._get_obs(), reward, self.done, info

        # 解码联合动作：ready task index + node index
        pair_action_dim = self.max_ready_tasks * self.max_nodes
        if action >= pair_action_dim:
            self.steps += 1
            reward = -5.0

            if self.steps >= self.max_steps_per_episode:
                self.done = True

            return self._get_obs(), reward, self.done, {
                "invalid_action": True,
                "reason": "action_out_of_pair_range",
            }

        ready_task_idx = int(action // self.max_nodes)
        node_idx = int(action % self.max_nodes)

        if ready_task_idx >= len(candidates) or node_idx >= len(self.compute_node_ids[: self.max_nodes]):
            self.steps += 1
            reward = -5.0
            return self._get_obs(), reward, self.done, {
                "invalid_action": True,
                "reason": "decoded_index_out_of_range",
                "ready_task_idx": ready_task_idx,
                "node_idx": node_idx,
            }

        job_idx, task_name = candidates[ready_task_idx]
        target_machine = str(self.compute_node_ids[node_idx])
        job = self.active_jobs[job_idx]

        if bool(getattr(self.config, "use_hierarchical_topology", True)):
            if not self._is_machine_allowed_for_task(job, task_name, target_machine):
                self.steps += 1
                reward = -5.0
                info = {
                    "invalid_action": True,
                    "invalid_reason": "topology_constraint",
                    "job_name": str(job.get("job_name", "")),
                    "task_name": str(task_name),
                    "target_machine": str(target_machine),
                    "source_end": str(job.get("source_end", "")),
                    "source_edge": str(job.get("source_edge", "")),
                    "task_demand": str(self._classify_task_demand(job, task_name)),
                }

                if self.steps >= self.max_steps_per_episode:
                    self.done = True

                return self._get_obs(), reward, self.done, info

        reward, info = self._execute_task(job_idx, task_name, target_machine)

        info["ready_task_idx"] = ready_task_idx
        info["node_idx"] = node_idx
        info["selected_task_name"] = task_name
        info["selected_job_idx"] = job_idx

        self.steps += 1

        if self.finished_jobs >= self.total_jobs:
            self.done = True
            if str(getattr(self.config, "reward_mode", "dense")).lower() in {"terminal_sparse", "sparse_terminal", "episode"}:
                terminal_reward, terminal_info = self._terminal_objective_reward()
                info["reward_before_terminal"] = float(reward)
                info["terminal_objective_reward"] = float(terminal_reward)
                info.update(terminal_info)
                reward = float(reward + terminal_reward)
                reward = float(np.clip(
                    reward,
                    float(getattr(self.config, "reward_clip_min", -50.0)),
                    float(getattr(self.config, "reward_clip_max", 20.0)),
                ))
                info["reward"] = float(reward)
                info["raw_reward"] = float(info.get("raw_reward", 0.0) + terminal_reward)

        elif self.steps >= self.max_steps_per_episode:
            self.done = True

            unfinished_ratio = 1.0 - float(self.finished_jobs / max(self.total_jobs, 1))
            if str(getattr(self.config, "reward_mode", "dense")).lower() in {"terminal_sparse", "sparse_terminal", "episode"}:
                terminal_reward, terminal_info = self._terminal_objective_reward()
                info["reward_before_terminal"] = float(reward)
                info["terminal_objective_reward"] = float(terminal_reward)
                info.update(terminal_info)
                reward = float(reward + terminal_reward)
                reward = float(np.clip(
                    reward,
                    float(getattr(self.config, "reward_clip_min", -50.0)),
                    float(getattr(self.config, "reward_clip_max", 20.0)),
                ))
                info["reward"] = float(reward)
                info["raw_reward"] = float(info.get("raw_reward", 0.0) + terminal_reward)
                info["unfinished_ratio"] = unfinished_ratio
            else:
                terminal_penalty = -5.0 * unfinished_ratio
                reward += terminal_penalty
                reward = float(np.clip(reward, -10.0, 10.0))
                info["terminal_penalty"] = terminal_penalty
                info["unfinished_ratio"] = unfinished_ratio

        obs = self._get_obs()
        return obs, reward, self.done, info

    def _task_static_exec_time_on_machine(
        self,
        job: Dict[str, Any],
        task_name: str,
        machine_id: str,
    ) -> float:
        """Static estimate w_{i,k} used by paper-style SLR.

        The paper's SLR denominator uses min_p{w_i,p}; w_i,p is the estimated
        execution cost of task i on processor p before scheduling. Therefore
        this helper uses static processor capacity rather than transient idle CPU.
        """
        attrs = dict(job.get("graph", nx.DiGraph()).nodes[task_name])
        runtime = max(
            1.0,
            float(attrs.get("runtime_mean", attrs.get("duration", 1.0))),
        )
        static = self.machine_static.get(str(machine_id), {})
        role = str(static.get("role", "edge"))
        raw_cpu_cap = float(static.get("cpu_num", 0.0))
        cpu_cap = raw_cpu_cap * float(self.role_cpu_scale.get(role, 1.0))

        mode = str(getattr(self.config, "slr_exec_cost_mode", "relative_runtime"))

        if mode == "legacy_cpu_demand":
            # Old formula. It is kept only for ablation/backward compatibility.
            # On Alibaba trace, runtime_mean is already wall-clock time, so this
            # mode often makes CP_MIN extremely small and SLR unrealistically huge.
            req_cpu = max(
                float(attrs.get("cpu_real_peak", attrs.get("plan_cpu", 0.0))) / 100.0,
                1e-3,
            )
            return float(runtime * (req_cpu / max(0.1, cpu_cap)))

        # Default paper-facing estimate: w_{i,k} is a wall-clock execution cost.
        # Alibaba runtime_mean/duration is already wall-clock duration, so we use
        # relative processor speed, not absolute host CPU capacity as a linear
        # parallel speedup. This keeps SLR in a meaningful range and in the same
        # time unit as AFT/makespan.
        caps = []
        for m in self._selected_nodes():
            st = self.machine_static.get(str(m), {})
            r = str(st.get("role", "edge"))
            caps.append(float(st.get("cpu_num", 0.0)) * float(self.role_cpu_scale.get(r, 1.0)))
        ref_cap = float(np.median(caps)) if caps else max(float(cpu_cap), 1.0)
        ref_cap = max(ref_cap, 1e-6)
        speed = max(float(cpu_cap) / ref_cap, 0.1)
        return float(runtime / speed)

    def _task_min_exec_time_for_slr(self, job: Dict[str, Any], task_name: str) -> float:
        """Compute min_{p_j in P} w_{i,j} for Eq.(31)-style SLR."""
        # To match the paper strictly, P is the full processor set available in
        # the scheduling environment, not only currently idle processors.
        machines = self._selected_nodes()
        if not machines:
            machines = list(self.compute_node_ids[: self.max_nodes])
        vals = [
            self._task_static_exec_time_on_machine(job, task_name, m)
            for m in machines
        ]
        return float(max(min(vals) if vals else 1.0, 1e-9))

    def _job_critical_path_min_runtime(self, job: Dict[str, Any]) -> float:
        """Strict CP_MIN denominator for paper SLR.

        SLR = makespan / sum_{t_i in CP_MIN} min_{p_j in P}{w_i,j}.
        CP_MIN is the critical path obtained after replacing every task weight
        by its minimum execution cost across processors.
        """
        g = job.get("graph", None)
        if g is None or g.number_of_nodes() == 0:
            return 1.0
        try:
            topo = list(nx.topological_sort(g))
        except Exception:
            topo = list(g.nodes)

        min_cost = {str(n): self._task_min_exec_time_for_slr(job, str(n)) for n in topo}
        dist: Dict[str, float] = {}
        for n in topo:
            n = str(n)
            preds = [str(p) for p in g.predecessors(n)] if hasattr(g, "predecessors") else []
            best_pred = max([dist.get(p, 0.0) for p in preds] or [0.0])
            dist[n] = best_pred + min_cost[n]
        return max(float(max(dist.values()) if dist else 1.0), 1e-9)

    def get_episode_metrics(self) -> Dict[str, float]:
        """Return paper-facing episode metrics.

        The key change is that makespan and avg_job_completion_time are computed
        on completed DAGs only, while completion_ratio is reported separately.
        This prevents all methods from being collapsed to the same episode
        horizon when a few jobs are unfinished.
        """
        completion_ratio = float(self.finished_jobs / max(self.total_jobs, 1))
        episode_horizon = float(self.max_steps_per_episode * self.slot_size)

        # =========================
        # Task response metrics
        # =========================
        if len(self.task_response_times) > 0:
            resp = np.array(self.task_response_times, dtype=np.float32)
            avg_task_response_time = float(np.mean(resp))
            median_task_response_time = float(np.median(resp))
            p95_task_response_time = float(np.percentile(resp, 95))
            norm_resp = resp / max(self.slot_size, 1.0)
            log_norm_resp = np.log1p(norm_resp)
            avg_norm_response_time = float(np.mean(log_norm_resp))
            p95_norm_response_time = float(np.percentile(log_norm_resp, 95))
            sla_threshold = float(getattr(self.config, "sla_response_threshold", 8.0 * self.slot_size))
            sla_violation_rate = float(np.mean(resp > sla_threshold))
        else:
            fallback_resp = episode_horizon
            avg_task_response_time = fallback_resp
            median_task_response_time = fallback_resp
            p95_task_response_time = fallback_resp
            avg_norm_response_time = float(np.log1p(fallback_resp / max(self.slot_size, 1.0)))
            p95_norm_response_time = avg_norm_response_time
            sla_violation_rate = 1.0

        # =========================
        # Completed-only job completion metrics
        # =========================
        completed_job_cts = [float(x) for x in self.job_completion_times]
        unfinished_jobs = max(0, int(self.total_jobs) - len(completed_job_cts))

        if completed_job_cts:
            completed_job_arr = np.array(completed_job_cts, dtype=np.float32)
            avg_job_completion_time = float(np.mean(completed_job_arr))
            median_job_completion_time = float(np.median(completed_job_arr))
            p95_job_completion_time = float(np.percentile(completed_job_arr, 95))
            avg_completed_job_completion_time = avg_job_completion_time
            p95_completed_job_completion_time = p95_job_completion_time
        else:
            avg_job_completion_time = episode_horizon
            median_job_completion_time = episode_horizon
            p95_job_completion_time = episode_horizon
            avg_completed_job_completion_time = episode_horizon
            p95_completed_job_completion_time = episode_horizon

        penalized_job_cts = completed_job_cts + [episode_horizon] * unfinished_jobs
        penalized_avg_job_completion_time = float(np.mean(penalized_job_cts)) if penalized_job_cts else episode_horizon

        # =========================
        # Energy and transfer
        # =========================
        avg_energy = float(np.mean(self.task_energy)) if self.task_energy else 0.0
        total_energy = float(np.sum(self.task_energy)) if self.task_energy else 0.0
        avg_transfer_cost = float(np.mean(self.transfer_costs)) if self.transfer_costs else 0.0
        total_transfer_cost = float(np.sum(self.transfer_costs)) if self.transfer_costs else 0.0

        # =========================
        # Paper-consistent completed-DAG makespan and SLR
        # =========================
        # For each completed DAG: makespan_DAG = max_i AFT_i - arrival_DAG.
        # The reported makespan is the average completed-DAG makespan, matching
        # the paper's "average makespan over DAGs" evaluation style. The full
        # multi-workflow episode span is still reported as episode_makespan.
        completed_finish_times = []
        completed_arrival_times = []
        dag_makespans = []
        slr_values = []
        for job in self.active_jobs:
            task_finish = job.get("task_finish_time", {})
            if job.get("done", False) and task_finish:
                finish = max(float(t) for t in task_finish.values())
                task_start = job.get("task_start_time", {})
                # Paper-style DAG makespan is the schedule length of this DAG.
                # Use the first actual task start as the local zero point so the
                # metric is not inflated by Alibaba trace inter-arrival gaps.
                if task_start:
                    local_start = min(float(t) for t in task_start.values())
                else:
                    local_start = float(job.get("arrival_time", self.episode_start_time))
                arrival = local_start
                completed_finish_times.append(finish)
                completed_arrival_times.append(arrival)
                job_makespan = max(0.0, finish - local_start)
                dag_makespans.append(job_makespan)
                cp_min = self._job_critical_path_min_runtime(job)
                slr_values.append(job_makespan / max(cp_min, 1e-9))

        if dag_makespans:
            makespan = max(float(np.mean(dag_makespans)), 1e-9)
            max_dag_makespan = max(float(np.max(dag_makespans)), 1e-9)
            raw_makespan = max(0.0, max(completed_finish_times) - min(completed_arrival_times))
        else:
            makespan = episode_horizon
            max_dag_makespan = episode_horizon
            raw_makespan = 0.0

        # This keeps a separate punishment channel without hiding completed-only differences.
        penalized_makespan = max(makespan, episode_horizon) if completion_ratio < 1.0 else makespan
        SLR = float(np.mean(slr_values)) if slr_values else float(episode_horizon / max(self.slot_size, 1.0))

        # =========================
        # Resource utilization metrics
        # =========================
        selected_nodes = self._selected_nodes()
        eligible_nodes = self._episode_eligible_nodes()
        if not eligible_nodes:
            eligible_nodes = selected_nodes
        active_nodes = [m for m in selected_nodes if self.machine_busy_time.get(str(m), 0.0) > 1e-9]
        total_busy_time = float(sum(self.machine_busy_time.get(str(m), 0.0) for m in selected_nodes))

        util_den = makespan if dag_makespans else episode_horizon
        util_global = self._safe_utilization(selected_nodes, util_den)
        util_eligible = self._safe_utilization(eligible_nodes, util_den)
        util_active = self._safe_utilization(active_nodes, util_den)

        # =========================
        # Paper-consistent load balance: L_CPU + L_Mem
        # =========================
        # The paper evaluates L_CPU and L_Mem on each DAG schedule. Because this
        # simulator packs multiple DAGs into one episode, using episode-level
        # machine_busy_time divided by the *average* DAG makespan would inflate
        # the load variance by orders of magnitude and make val_score almost
        # constant. Instead, compute load vectors per completed DAG:
        #     CPU_k = CPU busy time of this DAG on processor k / makespan_DAG
        #     MEM_k = memory occupation time of this DAG on processor k / makespan_DAG
        # Then average the variances over completed DAGs.
        per_job_cpu_vars = []
        per_job_mem_vars = []
        per_job_load_vectors = []
        per_job_mem_vectors = []
        load_nodes = selected_nodes  # P in the paper: all processors available in this experiment.

        for job in self.active_jobs:
            task_finish = job.get("task_finish_time", {})
            if not (job.get("done", False) and task_finish):
                continue

            finish = max(float(t) for t in task_finish.values())
            task_start = job.get("task_start_time", {})
            if task_start:
                local_start = min(float(t) for t in task_start.values())
            else:
                local_start = float(job.get("arrival_time", self.episode_start_time))
            job_makespan = max(1e-9, finish - local_start)

            cpu_time_by_node = {str(m): 0.0 for m in load_nodes}
            mem_time_by_node = {str(m): 0.0 for m in load_nodes}
            task_machine = job.get("task_machine", {})
            task_exec_time = job.get("task_exec_time", {})
            task_mem_usage_time = job.get("task_mem_usage_time", {})

            for task_name, machine_id in task_machine.items():
                m = str(machine_id)
                if m not in cpu_time_by_node:
                    continue
                cpu_time_by_node[m] += float(task_exec_time.get(task_name, 0.0))
                mem_time_by_node[m] += float(task_mem_usage_time.get(task_name, 0.0))

            # Use the paper-style normalized resource-load scale (0--100) before
            # computing L_CPU and L_Mem.  Makespan and SLR remain raw values.
            cpu_vec = np.array([100.0 * cpu_time_by_node[str(m)] / job_makespan for m in load_nodes], dtype=np.float64)
            mem_vec = np.array([100.0 * mem_time_by_node[str(m)] / job_makespan for m in load_nodes], dtype=np.float64)
            if len(cpu_vec) > 1:
                per_job_cpu_vars.append(self._resource_variation(cpu_vec))
                per_job_mem_vars.append(self._resource_variation(mem_vec))
                per_job_load_vectors.append(cpu_vec)
                per_job_mem_vectors.append(mem_vec)

        if per_job_cpu_vars:
            cpu_load_variance = float(np.mean(per_job_cpu_vars))
            mem_load_variance = float(np.mean(per_job_mem_vars))
            load_balance_var = float(cpu_load_variance + mem_load_variance)
            load_balance_std = float(np.sqrt(load_balance_var))
            all_cpu = np.concatenate(per_job_load_vectors) if per_job_load_vectors else np.zeros(1, dtype=np.float64)
            all_mem = np.concatenate(per_job_mem_vectors) if per_job_mem_vectors else np.zeros(1, dtype=np.float64)
            combined = all_cpu + all_mem
            load_balance_cv = float(np.std(combined) / (np.mean(combined) + 1e-9))
            load_balance_range = float(np.max(combined) - np.min(combined))
            avg_cpu_usage = float(np.mean(all_cpu))
            avg_memory_utilization = float(np.mean(all_mem))
        else:
            # Fallback for failed episodes: keep finite values but let completion_ratio
            # expose the failure explicitly.
            cpu_load, mem_load = self._paper_resource_load_vectors(selected_nodes, max(raw_makespan, self.slot_size))
            cpu_load_variance = self._resource_variation(cpu_load)
            mem_load_variance = self._resource_variation(mem_load)
            load_balance_var = float(cpu_load_variance + mem_load_variance)
            load_balance_std = float(np.sqrt(load_balance_var))
            combined = cpu_load + mem_load
            load_balance_cv = float(np.std(combined) / (np.mean(combined) + 1e-9)) if len(combined) else 0.0
            load_balance_range = float(np.max(combined) - np.min(combined)) if len(combined) else 0.0
            avg_cpu_usage = float(np.mean(cpu_load)) if len(cpu_load) else 0.0
            avg_memory_utilization = float(np.mean(mem_load)) if len(mem_load) else 0.0

        avg_load_balance = float(load_balance_var)

        # =========================
        # Layer-wise utilization
        # =========================
        role_utils = {}
        for role in ["cloud", "edge", "end"]:
            role_nodes_global = [m for m in selected_nodes if str(self.machine_static.get(str(m), {}).get("role", "")) == role]
            role_nodes_eligible = [m for m in eligible_nodes if str(self.machine_static.get(str(m), {}).get("role", "")) == role]
            role_nodes_active = [m for m in active_nodes if str(self.machine_static.get(str(m), {}).get("role", "")) == role]
            role_utils[f"util_{role}"] = self._safe_utilization(role_nodes_global, util_den)
            role_utils[f"util_{role}_eligible"] = self._safe_utilization(role_nodes_eligible, util_den)
            role_utils[f"util_{role}_active"] = self._safe_utilization(role_nodes_active, util_den)

        # =========================
        # Scheduling distribution by role
        # =========================
        role_task_counts = {
            "scheduled_cloud_tasks": 0.0,
            "scheduled_edge_tasks": 0.0,
            "scheduled_end_tasks": 0.0,
        }
        for job in self.active_jobs:
            for m in job.get("task_machine", {}).values():
                role = str(self.machine_static.get(str(m), {}).get("role", ""))
                if role == "cloud":
                    role_task_counts["scheduled_cloud_tasks"] += 1.0
                elif role == "edge":
                    role_task_counts["scheduled_edge_tasks"] += 1.0
                elif role == "end":
                    role_task_counts["scheduled_end_tasks"] += 1.0

        scheduled_task_num = float(len(self.task_response_times))
        if scheduled_task_num > 0:
            role_task_counts["scheduled_cloud_ratio"] = role_task_counts["scheduled_cloud_tasks"] / scheduled_task_num
            role_task_counts["scheduled_edge_ratio"] = role_task_counts["scheduled_edge_tasks"] / scheduled_task_num
            role_task_counts["scheduled_end_ratio"] = role_task_counts["scheduled_end_tasks"] / scheduled_task_num
        else:
            role_task_counts["scheduled_cloud_ratio"] = 0.0
            role_task_counts["scheduled_edge_ratio"] = 0.0
            role_task_counts["scheduled_end_ratio"] = 0.0

        return {
            "makespan": float(makespan),
            "raw_makespan": float(raw_makespan),
            "episode_makespan": float(raw_makespan),
            "max_dag_makespan": float(max_dag_makespan),
            "penalized_makespan": float(penalized_makespan),
            "episode_horizon": float(episode_horizon),
            "SLR": float(SLR),
            "slr": float(SLR),

            "avg_response_time": avg_task_response_time,
            "avg_task_response_time": avg_task_response_time,
            "median_task_response_time": median_task_response_time,
            "p95_task_response_time": p95_task_response_time,
            "avg_norm_response_time": avg_norm_response_time,
            "p95_norm_response_time": p95_norm_response_time,
            "sla_violation_rate": sla_violation_rate,

            "avg_job_completion_time": avg_job_completion_time,
            "median_job_completion_time": median_job_completion_time,
            "p95_job_completion_time": p95_job_completion_time,
            "avg_completed_job_completion_time": avg_completed_job_completion_time,
            "p95_completed_job_completion_time": p95_completed_job_completion_time,
            "penalized_avg_job_completion_time": penalized_avg_job_completion_time,

            "resource_utilization": util_eligible,
            "resource_utilization_global": util_global,
            "resource_utilization_eligible": util_eligible,
            "resource_utilization_active": util_active,
            "eligible_node_num": float(len(eligible_nodes)),
            "active_node_num": float(len(active_nodes)),
            "total_busy_time": total_busy_time,

            "load_balance": avg_load_balance,
            "avg_load_balance": avg_load_balance,
            "L_CPU": float(cpu_load_variance),
            "L_Mem": float(mem_load_variance),
            "cpu_load_variance": float(cpu_load_variance),
            "mem_load_variance": float(mem_load_variance),
            "avg_cpu_usage": float(avg_cpu_usage),
            "avg_memory_utilization": float(avg_memory_utilization),
            "load_balance_std": load_balance_std,
            "load_balance_cv": load_balance_cv,
            "load_balance_var": load_balance_var,
            "load_balance_range": load_balance_range,

            **role_utils,

            "avg_energy": avg_energy,
            "total_energy": total_energy,
            "avg_transfer_cost": avg_transfer_cost,
            "total_transfer_cost": total_transfer_cost,

            "finished_jobs": float(self.finished_jobs),
            "total_jobs": float(self.total_jobs),
            "completion_ratio": completion_ratio,
            "unfinished_jobs": float(unfinished_jobs),
            "scheduled_task_num": scheduled_task_num,

            **role_task_counts,
        }

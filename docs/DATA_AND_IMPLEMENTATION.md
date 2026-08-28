# Data and Implementation

## Supplied Processed Workload

The locally supplied `new500DAG/meta.json` describes the processed workload (not redistributed here):

| Item | Value |
| :--- | ---: |
| Workflows | 500 |
| Training / validation / test | 398 / 48 / 54 |
| Tasks per DAG | 3–50 |
| Resource nodes | 90 |
| Cloud / edge / end nodes in the data pool | 2 / 8 / 80 |
| Resource-state rows | 181,890 |
| Background-service rows | 181,890 |
| Slot size | 300 |
| Dataset construction seed in metadata | 42 |

`jobs_*.jsonl` contains workflow tasks, dependency edges, and topological orders. `nodes.csv` describes the resource pool, `node_state.csv` contains resource time series, `service_bg.csv` contains background service information, and `dag_stats.csv` summarizes DAGs.

The source references Alibaba trace data, but the exact upstream snapshot, preprocessing program, and upstream license are not included in the supplied package. The public release does not redistribute the processed files, assert a verified upstream reconstruction, or assign them a new license. Check upstream attribution and redistribution terms before republishing the dataset independently.

## Data Access

No public data or checkpoint download is provided in this release. Contact the repository maintainer about availability and permitted use. If you already have an authorized copy, use `--data-dir /path/to/new500DAG`; for tests, set `PPO_STGNN_DATA_DIR` to that directory. Do not place data or weights into a public commit without separate authorization. The synthetic model tests do not require the research dataset.

## Runtime Configuration Is Separate

The default experiment builder selects **2 cloud + 6 edge + 30 end nodes**, for **38 candidate execution nodes**. It sets four ready-task slots and a defer action, yielding `4 × 38 + 1 = 153` action slots. Not all action slots are valid in every state.

The current configuration uses a resource history length of five and a DAG observation capacity of 48 nodes. The dataset includes DAGs with up to 50 tasks; the fixed-size observation must not be described as an unrestricted representation of every task in those larger DAGs.

The default non-quick setting samples 18 workflows per episode. `paper_static_arrivals=True` makes the sampled workflows available at the start; resource disturbances are enabled separately. These choices should be included in any experimental report.

## Scheduling Path

1. The simulator constructs resource histories, DAG features, candidate task features, task–node interaction features, and an action mask.
2. The policy encodes the resource and task observations. The supplied temporal model uses graph attention and temporal attention; see the [paper/code differences](REPRODUCIBILITY.md#paper-and-implementation).
3. An action selects a task slot and node through `action = task_slot × max_nodes + node_index`, or selects the defer slot.
4. The simulator advances the schedule and returns `(observation, reward, done, info)`.
5. PPO uses rollout data, generalized advantage estimation, and a clipped policy objective to update the actor–critic.

Behavior cloning uses demonstrations collected from scheduling heuristics before PPO training. The experiment utilities also contain optional refiners, presets, and architecture-specific adjustments. Check the effective run configuration before attributing a result solely to the encoder.

## Preserved Artifacts

The local `results/` and `checkpoints/` folders form a research archive, not a single clean benchmark run; neither folder is published here. Machine-specific paths in archived JSON files reflect the original configuration and are not portable instructions. Use `runs/` for new experiments.

`docs/source-manifest.json` contains content hashes of the 15 original source/documentation files included in this public release. It can detect accidental changes during release preparation; it is not evidence of historical development dates or authorship.

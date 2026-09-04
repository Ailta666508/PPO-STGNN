# Reproducibility Notes

This document separates the manuscript's claims, locally supplied artifacts, and software checks performed while preparing the repository. It is not a certification that the paper's experiments have been reproduced.

## Paper and Implementation

| Component | Manuscript | Supplied implementation |
| :--- | :--- | :--- |
| Temporal resource encoder | Graph attention followed by a GRU (Eq. 4). | Graph attention followed by temporal attention and a temporal gate in `StructuralTemporalResourceEncoder`. No GRU is present. |
| Task DAG encoder | GCN formulation (Eq. 5). | Edge-aware graph attention layers in `TaskGraphEncoder`. |
| Main makespan terminology | Describes completion of the overall workflow workload. | `makespan` is the mean completion duration of completed DAGs; separate fields report overall and maximum durations. |
| Architecture comparison | Compares MLP, static graph, and temporal graph representations. | Multiple archived experiment modes exist, including modes with architecture-specific training adjustments. Use `unified_fair` for the intended controlled comparison. |

The manuscript and implementation must therefore not be treated as interchangeable specifications. Resolving the encoder differences requires identifying the exact implementation associated with the manuscript experiments; renaming the current modules would not resolve them.

## Paper and Artifact Results

The manuscript Table 1 and the CSV summaries contain the following PPO-STGNN values:

| Source | Makespan | SLR | Load balance |
| :--- | ---: | ---: | ---: |
| Manuscript Table 1 | 124.8 | 11.6 | 127.9 |
| Local `results/ppo_results.csv` | 137.27560 | 11.17499 | 146.74820 |
| Local `results/ppo_results_PPO-STGNN.csv` | 402.40526 | 38.22068 | 153.26339 |

The baseline rows in the local `baseline_results.csv` match the manuscript's rounded FCFS, LeastLoad, HEFT, and Greedy rows. That agreement alone does not establish the provenance of the manuscript's PPO row.

The supplied summary CSVs do not establish which checkpoint, configuration, seed set, or code revision produced the manuscript's PPO result. Historical data, logs, and weights are retained locally and are not redistributed in this public repository. Their reported values are not relabeled as newly reproduced measurements.

## Metrics and Completion

See `CloudEdgeDagEnv.get_episode_metrics` in [`env_cec_dag.py`](../cecoppo/env_cec_dag.py).

- **`makespan`**: mean per-DAG duration over completed DAGs, measured from each DAG's first task start to its final task finish.
- **`max_dag_makespan`**: maximum of the completed DAG durations.
- **`raw_makespan` / `episode_makespan`**: overall episode duration as computed by the simulator; this is not the same quantity as the mean per-DAG `makespan`.
- **`SLR`**: mean per-DAG scheduling length ratio over completed DAGs, using the critical-path lower bound calculated by the simulator.
- **`load_balance`**: the sum of CPU and memory load variances, averaged over completed DAGs. The implementation uses percentage-scaled utilization, so its numeric scale must be preserved when comparing results.
- **`completion_ratio`**: must be reported alongside time and load metrics. A short run that leaves workflows unfinished must not be presented as a successful scheduling result merely because its completed-workflow mean is low.

Do not mix metrics with different definitions or completion filters in one comparison table.

## Baseline Integrity

The original source includes an optional `PaperBaselineWrapper` that replaces some heuristic actions when a positive degradation probability is requested. This is **not appropriate for a standard baseline comparison**.

- `run_baselines.py` has an effective CLI default of **0**.
- The locally supplied `baseline_config.json` records **`baseline_degrade_prob: 0.0`**.
- For standard comparisons, explicitly use **`--baseline-degrade-prob 0`**.
- The wrapper constructor's separate default and the CLI's legacy help text should not be interpreted as evidence that the archived experiment used a nonzero probability.

The original code is preserved for inspection. No result is strengthened by removing unfavorable baselines or concealing experimental switches.

## Configuration and Plotting Pitfalls

1. **Original environment unavailable.** The package did not include a dependency lockfile. The requirements in this release record a newly tested environment.
2. **Different defaults and archived configurations.** `build_default_config` overrides the dataclass defaults; archived `train_config*.json` files also contain settings that differ from the current builder. Record the configuration actually used for each run.
3. **Static arrivals by default.** `paper_static_arrivals=True` makes sampled workflows available at time zero. Resource disturbances can still vary over time. Do not describe this default as an experiment with dynamic workflow arrivals.
4. **Keep old artifacts intact.** Default script output paths point to `results/` and `checkpoints/`. Use a new `--project-root ./runs/<run-name>` to separate each new run from any local historical artifacts.
5. **Avoid mixed summaries.** The plotting loader can read both `ppo_results.csv` and `ppo_results_PPO-STGNN.csv` and keep the later duplicate. In the supplied archive this can select the 402.40526 run. Plot a fresh, internally consistent run directory and inspect its CSVs first.
6. **Controlled architecture comparisons.** The default `unified` mode includes method-specific adjustments. `unified_fair` is the mode intended to hold the training setup common while varying the encoder; inspect its saved plan and configurations before drawing conclusions.
7. **Checkpoints are historical artifacts.** Their presence does not establish a checkpoint-to-paper-result mapping. Only load model files from sources you trust.

## Validation Scope

The repository provides small CPU tests for dataset structure, environment interaction, action validity, encoder masking, and PPO optimization. The source verifier checks packaged-file hashes and Python syntax. These checks do not validate scientific claims, test-set performance, training convergence, or publication status.

On 2026-08-28, release validation passed on Python 3.12.13 / macOS arm64 / CPU: all 8 tests with local data, integrity checking of the 131-file local package, parsing of all 14 original Python files, and all four CLI help entry points. The documented `--quick` training command also completed heuristic behavior cloning, two PPO epochs, checkpoint saving/loading, and final evaluation successfully; its two quick test episodes had completion ratio 1.0, but this small functional run is not a benchmark.

The current public suite runs 13 deterministic data-free tests and skips five dataset-dependent checks. GitHub Actions executes the source and publication-asset integrity verifier together with this CPU suite on every push and pull request. The public manifest covers the 15 redistributed original source/documentation files; separate SHA-256 checks bind the repository PDF and all four displayed figures to the arXiv v1 paper. The [machine-readable record](release-validation.json) lists the dependency versions and current verification scope.

Full paper reproduction remains open until the manuscript-specific implementation, configuration, checkpoint, evaluation episodes, and raw results can be matched. Future reproduction reports should include commit ID, environment, seeds, completion ratios, and raw per-episode outputs, including any results that do not improve on the baselines.

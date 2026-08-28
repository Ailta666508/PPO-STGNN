# Running Experiments

Run commands from the repository root after installing the dependencies. These commands exercise the supplied implementation; they do not promise the manuscript's numerical results. Consult [Reproducibility Notes](REPRODUCIBILITY.md) before comparing methods.

## Provide the Dataset

The public repository does not include processed data, model weights, or historical logs. If you have authorized access to the workload, place a copy at `./new500DAG` or replace `--data-dir ./new500DAG` in the commands below with its actual location. Do not publish those files without separate authorization.

To run data-dependent checks against a copy outside the repository:

```bash
PPO_STGNN_DATA_DIR=/path/to/new500DAG python -m pytest -q
```

Without data, the three synthetic model tests still run and the other five checks are skipped.

## Small Functional Run

```bash
python train_ppo.py \
  --data-dir ./new500DAG \
  --project-root ./runs/quick \
  --quick --device cpu --seed 42
```

The quick preset reduces PPO training, rollout length, episode size, and behavior cloning. Its purpose is to check the end-to-end path, not to produce a publishable score. Runtime depends on the CPU and may exceed the much smaller `pytest` checks.

## PPO-STGNN Training and Evaluation

```bash
python train_ppo.py \
  --data-dir ./new500DAG \
  --project-root ./runs/main-seed42 \
  --device cpu --seed 42 \
  --encoder-type stgnn \
  --teacher-samples 6000 --bc-epochs 5 \
  --eval-episodes 20 --test-episodes 20 \
  --val-every 5 --reward-mode terminal_sparse
```

Change `--device cpu` to `--device cuda` only with a compatible GPU environment. The builder currently sets 150 PPO epochs, 2,048 rollout steps per epoch, and hidden dimension 144 for the non-quick path. The saved JSON is the record of the effective settings; inspect it before starting a long experiment.

The entry point uses training seed 42 above and a fixed test evaluation seed of 500. Validation is used for checkpoint selection. Preserve the best checkpoint, configuration, history, per-episode test output, and aggregate summary together.

## Non-learning Baselines

```bash
python run_baselines.py \
  --data-dir ./new500DAG \
  --project-root ./runs/main-seed42 \
  --device cpu --seed 500 --eval-episodes 20 \
  --baseline-degrade-prob 0
```

Use the same environment and test workload settings as the PPO evaluation. If you override resource disturbance or reward-related environment settings in training, reconcile the baseline configuration explicitly before comparing the results. A shared folder name alone does not guarantee equivalent settings.

## Controlled Encoder Comparison

```bash
python run_ablations.py \
  --data-dir ./new500DAG \
  --project-root ./runs/architecture-seed42 \
  --experiment-type architecture --arch-bc-mode unified_fair \
  --device cpu --seed 42 \
  --teacher-samples 6000 --bc-epochs 3 \
  --eval-episodes 20 --test-episodes 20 \
  --force-retrain
```

This selects the intended common-training comparison of MLP-PPO, PPO-StaticGNN, and PPO-STGNN. It is distinct from the original `unified` mode, which applies architecture-specific adjustments. Review the generated experiment plan, not only the mode name.

For component ablations, use `--experiment-type ablation` and inspect `python run_ablations.py --help`. Do not aggregate runs with different reward presets as if only one component changed.

## Plot a Fresh Run

```bash
python plot_results.py \
  --result-dir ./runs/main-seed42/results \
  --plot-type baseline
```

Inspect the input CSVs first. Do not run this command against a historical `results/` archive expecting it to regenerate the exact paper figures; the local source package contains artifacts from multiple runs. The original paper figures are preserved separately under `docs/assets/`.

## Reporting Checklist

- Record the commit, Python and dependency versions, device, and all seeds.
- Retain the effective environment and training configurations.
- Keep the test split separate from training and checkpoint selection.
- Disable baseline degradation and disclose any action refiners or method-specific presets.
- Report completion ratios with makespan, SLR, and load balance.
- Use repeated independent training seeds and per-episode outputs before making robust performance claims.
- Label paper-reported, archived, and newly reproduced results separately.

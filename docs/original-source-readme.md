# PPO-STGNN 云边端 DAG 调度实验运行说明

本项目用于运行云-边-端环境下的 DAG 任务调度实验，主要包含传统 baseline、PPO-STGNN 训练、架构对比、消融实验和结果绘图。

## 1. 项目结构

```text
.
├── cecoppo/                 # 环境、baseline、PPO agent、绘图与实验工具
├── new500DAG/               # 默认数据集目录
├── checkpoints/             # 训练得到的模型权重
├── results/                 # CSV 结果、训练历史和图片
├── run_baselines.py         # 运行传统调度 baseline
├── train_ppo.py             # 训练并测试 PPO-STGNN
├── run_ablations.py         # 运行架构对比和消融实验
├── plot_results.py          # 根据已有 CSV 生成图表
└── README.md                # 当前统一说明文件
```

默认数据目录为 `new500DAG`。如果该目录就在项目根目录下，运行脚本时可以省略 `--data-dir`；否则需要手动传入数据路径。


##2 .正式运行

完整实验一般按以下顺序执行：

```bash
python run_baselines.py --data-dir new500DAG --eval-episodes 20
python train_ppo.py --data-dir new500DAG --eval-episodes 20 --test-episodes 30 --val-every 5
python plot_results.py --plot-type baseline
```

## 3. 各脚本使用

### 3.1 运行 baseline

```bash
python run_baselines.py --data-dir new500DAG --eval-episodes 20
```


主要输出：

```text
results/baseline_config.json
results/baseline_results.csv
results/detail_<method>.csv
```

### 3.2 训练 PPO-STGNN

```bash
python train_ppo.py --data-dir new500DAG --eval-episodes 20 --test-episodes 30 --val-every 5
```



主要输出：

```text
checkpoints/best_PPO-STGNN.pt
checkpoints/latest_PPO-STGNN.pt
results/train_config.json
results/PPO-STGNN_training_history.csv
results/ppo_results.csv
results/detail_PPO-STGNN.csv
```

### 3.3 绘图

已有 CSV 结果后，可以单独重新绘图：

```bash
python plot_results.py --plot-type baseline
```

可选绘图类型：

- `baseline`：传统 baseline + PPO-STGNN 对比。
- `architecture`：架构对比。
- `ablation`：消融实验对比。
- `training`：训练曲线。
- `all`：生成全部可用图表。

常见输出：

```text
results/method_comparison_makespan.png
results/method_comparison_SLR.png
results/method_comparison_load_balance.png
results/method_comparison_panel.png
```

### 3.4 架构对比与消融实验

运行架构对比：

```bash
python run_ablations.py --data-dir new500DAG --experiment-type architecture --arch-bc-mode unified --eval-episodes 15 --test-episodes 15
python plot_results.py --plot-type architecture --arch-bc-mode unified
```

运行消融实验：

```bash
python run_ablations.py --data-dir new500DAG --experiment-type ablation --eval-episodes 15 --test-episodes 15
python plot_results.py --plot-type ablation
```








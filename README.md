# AT-Dec-POSG Support Materials

本压缩包只保留实验复现所需的核心代码、保留的 baseline 模型和一个 `best` 模型。

## 目录

- `doudizhu/`: 斗地主实验代码
- `option/`: 第二个执行任务环境代码

## 斗地主实验

### 1. 安装

```bash
cd doudizhu
python3 -m pip install -r requirements.txt
```

### 2. 训练

基础训练：

```bash
python3 train.py --seed 2026
```

如果希望更接近当前保留的 `best` 模型配置，建议使用单独农民席位：

```bash
python3 train.py --seed 2026 --separate_farmer_seats true
```

常见训练模板：

```bash
# ADP 目标
python3 train.py --objective adp --seed 2026

# WP 目标
python3 train.py --objective wp --seed 2026

# 共享农民参数
python3 train.py --seed 2026 --separate_farmer_seats false

# 单独农民席位
python3 train.py --seed 2026 --separate_farmer_seats true

# 关闭某个模块做消融
python3 train.py --seed 2026 --enable_module_a false
python3 train.py --seed 2026 --enable_module_b false
python3 train.py --seed 2026 --enable_module_c false

# 从已有 checkpoint 恢复
python3 train.py --seed 2026 --load_model --savedir dmc_checkpoints --xpid your_run_name
```

`train.py` 完整选项速查：

通用设置：

- `--xpid`: 实验名，决定保存子目录名
- `--seed`: 随机种子
- `--save_interval`: checkpoint 保存间隔，单位分钟
- `--objective`: 回报目标，`adp` / `wp` / `logadp`

设备与训练流程：

- `--actor_device_cpu`: 强制 actor 走 CPU
- `--gpu_devices`: 可见 GPU 列表，例如 `0` 或 `0,1`
- `--num_actor_devices`: 用多少个设备做模拟
- `--num_actors`: 每个模拟设备上的 actor 数
- `--training_device`: learner 使用的设备编号，或 `cpu`
- `--load_model`: 从已有实验目录恢复
- `--disable_checkpoint`: 训练时不保存 checkpoint
- `--savedir`: 实验输出根目录

核心超参数：

- `--total_frames`: 总训练帧数
- `--exp_epsilon`: acting 时的 epsilon-greedy 探索概率
- `--replay_buffer_size`: 每个位置的优先回放容量
- `--replay_warmup_size`: 优先回放开始采样前的最小热身大小
- `--priority_alpha`: PER 优先级指数
- `--priority_beta`: PER 重要性采样修正强度
- `--priority_epsilon`: PER 更新优先级时加到 TD 误差上的常数

模块与任务开关：

- `--enable_module_a true|false`: 是否启用 Module A
- `--enable_module_b true|false`: 是否启用 Module B
- `--enable_module_c true|false`: 是否启用 Module C
- `--separate_farmer_seats true|false`: 是否给 `landlord_up` / `landlord_down` 分开建模
- `--train_bidding true|false`: 是否训练 bidding 模型
- `--train_multiply`: 是否把 multiply 阶段样本也并入训练

辅助损失：

- `--belief_coef`: 隐藏手牌 belief loss 权重
- `--coord_sender_coef`: farmer 发信端 coordination loss 权重
- `--coord_receiver_coef`: farmer 收信端 coordination loss 权重

联赛 / 对手池：

- `--opponent_pool_size`: 旧对手池兼容参数
- `--league_snapshot_size`: 联赛中近期 snapshot 数量
- `--league_exploiter_size`: 联赛中 exploiter 数量
- `--league_main_prob`: 采样当前主策略作为对手的概率
- `--league_snapshot_prob`: 采样近期 snapshot 的概率
- `--league_exploiter_prob`: 采样 exploiter 的概率
- `--external_opponent`: 外部对手，目前支持 `perfectdou`
- `--external_opponent_prob`: 外部对手的采样概率
- `--perfectdou_repo_root`: PerfectDou 仓库根目录
- `--perfectdou_dir`: PerfectDou ONNX 权重目录
- `--league_exploiter_train_prob`: 一局 self-play 训练 exploiter 的概率
- `--league_exploiter_reset_interval`: exploiter 从主策略重置的帧间隔

minimax exploiter 相关：

- `--minimax_exploiter_alpha`: exploiter shaping 强度
- `--minimax_exploiter_gamma`: exploiter shaping 折扣因子
- `--minimax_value_floor`: opponent value 平移下界
- `--minimax_exploiter_enabled true|false`: 是否启用 minimax exploiter shaping

对手采样刷新与退火：

- `--opponent_refresh_episodes`: actor 刷新对手池元数据的回合间隔
- `--sa_initial_temperature`: simulated annealing 初始温度
- `--sa_final_temperature`: simulated annealing 末端温度
- `--sa_decay_episodes`: 温度退火回合数

learner 批处理：

- `--batch_size`: learner batch 大小
- `--unroll_length`: actor 产生的时间展开长度
- `--num_buffers`: 共享内存 buffer 数量
- `--num_threads`: learner 线程数
- `--max_grad_norm`: 梯度裁剪上限

优化器：

- `--learning_rate`: 学习率
- `--alpha`: RMSProp 平滑常数
- `--momentum`: RMSProp 动量
- `--epsilon`: RMSProp 数值稳定常数

布尔参数写法：

- 可以写成 `--enable_module_a false`
- 也可以写成 `--separate_farmer_seats true`
- 不写时使用默认值

### 3. 生成评测数据

```bash
python3 generate_eval_data.py --seed 2026
```

### 4. 使用保留的 best 模型评测

`best` 模型目录：

- `doudizhu/pre-train/best/resnet_init_adp_noC_stable_best_10006528/`

该目录中保留了：

- `model.tar`

如果你要直接做全局对战评测，通常使用同目录下的 `model.tar` 配合训练/恢复流程，或者使用训练导出的 `general_*.ckpt` 做显式对战。

### 5. 模仿 DouZero 的 baseline 对战

保留的 DouZero baseline：

- `baselines/douzero_WP/`
- `baselines/douzero_ADP/`

示例：把你的模型放在地主位，对战 `douzero_WP`：

```bash
python3 evaluate.py \
  --landlord path/to/general_landlord.ckpt \
  --landlord_down baselines/douzero_WP/landlord_down.ckpt \
  --landlord_up baselines/douzero_WP/landlord_up.ckpt \
  --seed 2026
```

示例：对战 `douzero_ADP`：

```bash
python3 evaluate.py \
  --landlord path/to/general_landlord.ckpt \
  --landlord_down baselines/douzero_ADP/landlord_down.ckpt \
  --landlord_up baselines/douzero_ADP/landlord_up.ckpt \
  --seed 2026
```

### 6. 模仿 PerfectDou 的 baseline 对战

保留的 PerfectDou baseline：

- `baselines/perfectdou/`

示例：把你的模型放在地主位，对战 PerfectDou：

```bash
python3 evaluate.py \
  --landlord path/to/general_landlord.ckpt \
  --landlord_down baselines/perfectdou/landlord_down.onnx \
  --landlord_up baselines/perfectdou/landlord_up.onnx \
  --seed 2026
```

如果要反向测试农民位，同理把 PerfectDou 放在地主位、你的模型放在 `landlord_up` 和 `landlord_down`。

### 7. 冒烟测试

```bash
bash scripts/smoke_test.sh
```

## Option 实验

基础运行：

```bash
python3 -m option.train --solver benchmark --episodes 120 --eval-episodes 20 --benchmark-seeds 5
```

完整 solver：

```bash
python3 -m option.train --solver full --episodes 200 --eval-episodes 30
```

## 说明

- 包内不包含训练日志、测试样本和运行输出。
- 斗地主目录已移除 teacher/distillation 相关逻辑与独立脚本。
- 包内只保留一个最外层 `README.md`。

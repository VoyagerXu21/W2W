# W2W 启动说明

这个文件整理了当前仓库最常用的启动命令，直接复制即可使用。

## 训练入口

统一入口：

```bash
accelerate launch trainval.py --cfg <配置文件> --dataset <数据集> --tag <实验名> --train_mode <sft|ppo>
```

可选数据集：`eth`、`hotel`、`univ`、`zara1`、`zara2`。

---

## SFT 训练示例

### 1. meter 配置训练 eth

```bash
accelerate launch trainval.py --cfg ./config/sft-meter.json --dataset eth --tag LMTraj-sft-meter-eth --train_mode sft
```

### 2. pixel 配置训练 eth

```bash
accelerate launch trainval.py --cfg ./config/sft-pixel.json --dataset eth --tag LMTraj-sft-pixel-eth --train_mode sft
```

### 3. meter deterministic 配置训练 hotel

```bash
accelerate launch trainval.py --cfg ./config/sft-meter-deterministic.json --dataset hotel --tag LMTraj-sft-meter-hotel-det --train_mode sft
```

### 4. pixel deterministic 配置训练 zara1

```bash
accelerate launch trainval.py --cfg ./config/sft-pixel-deterministic.json --dataset zara1 --tag LMTraj-sft-pixel-zara1-det --train_mode sft
```

---

## PPO 训练示例

> 运行 PPO 前，请先确认 `config/ppo-pixel.json` 中的 `model_name_or_path`、`ce_checkpoint_dir`、`rl_checkpoint_dir` 已按你的实际路径配置好。

### 5. 使用 PPO 配置训练 eth

```bash
accelerate launch trainval.py --cfg ./config/ppo-pixel.json --dataset eth --tag LMTraj-ppo-eth --train_mode ppo
```

### 6. 使用 PPO 配置训练 hotel

```bash
accelerate launch trainval.py --cfg ./config/ppo-pixel.json --dataset hotel --tag LMTraj-ppo-hotel --train_mode ppo
```

### 7. 使用 PPO 配置训练 zara2

```bash
accelerate launch trainval.py --cfg ./config/ppo-pixel.json --dataset zara2 --tag LMTraj-ppo-zara2 --train_mode ppo
```

---

## 测试 / 评估示例

### 8. 用 SFT 配置做测试

```bash
accelerate launch trainval.py --cfg ./config/sft-meter.json --dataset eth --tag LMTraj-sft-meter-eth --test
```

### 9. 用 PPO 配置做测试

```bash
accelerate launch trainval.py --cfg ./config/ppo-pixel.json --dataset eth --tag LMTraj-ppo-eth --test
```

---

## 推荐使用方式

- 跑 SFT：优先使用 `config/sft-*.json`
- 跑 PPO：优先使用 `config/ppo-pixel.json`
- 如果你要做新实验，建议新建你自己的配置文件，例如：

```bash
config/ppo-pixel-exp1.json
config/sft-meter-exp1.json
```

然后用 `_base_` 继承现有配置，只覆盖你想改的字段。

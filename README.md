# CVPR2026

# W2W: Language-Model-Based Trajectory Prediction with Reinforcement Learning

本仓库为论文 **W2W: Language-Model-Based Trajectory Prediction with Reinforcement Learning** 的代码实现与实验说明。

## 项目简介

W2W 通过将轨迹预测任务建模为语言建模问题，并结合强化学习进行后训练优化，以提升多步轨迹预测性能。

## 论文流程图

> 你上传图片后，下面这行要**直接写在正文里**（不能放在代码块里），这样 Markdown 才会渲染图片。

![W2W Pipeline](./assets/w2w_pipeline.png)

如果仍不显示，请检查：
- 图片文件是否真实存在于 `assets/w2w_pipeline.png`。
- 文件名大小写是否完全一致（`W2W_Pipeline.png` 与 `w2w_pipeline.png` 会被视为不同文件）。
- 图片是否已经被 `git add` 并提交。

## 环境依赖

建议使用 Python 3.10+。

核心依赖库（按当前代码结构整理）：

- `torch`
- `transformers`
- `accelerate`
- `datasets`
- `numpy`
- `scipy`
- `pandas`
- `tqdm`
- `matplotlib`

你可以按需创建 `requirements.txt`，或先手动安装常用依赖：

```bash
pip install torch transformers accelerate datasets numpy scipy pandas tqdm matplotlib
```

## 数据与配置

- 配置文件位于 `config/` 目录。
- 常用数据集参数：`eth`、`hotel`、`univ`、`zara1`、`zara2`。
- 若使用 PPO，请先检查 `config/ppo-pixel.json` 中关键路径（如模型权重与 checkpoint 路径）是否已替换为本地有效路径。

## 如何使用

### 1) 统一训练入口

```bash
accelerate launch trainval.py --cfg <配置文件> --dataset <数据集> --tag <实验名> --train_mode <sft|ppo>
```

### 2) SFT 训练示例

```bash
accelerate launch trainval.py --cfg ./config/sft-meter.json --dataset eth --tag W2W-sft-meter-eth --train_mode sft
```

### 3) PPO 训练示例

```bash
accelerate launch trainval.py --cfg ./config/ppo-pixel.json --dataset eth --tag W2W-ppo-eth --train_mode ppo
```

### 4) 评估示例

```bash
accelerate launch trainval.py --cfg ./config/sft-meter.json --dataset eth --tag W2W-sft-meter-eth --test
```

## 目录结构

```text
.
├── config/           # 训练/测试配置
├── model/            # 模型与训练/评估逻辑
├── utils/            # 数据处理与工具函数
├── script/           # 批处理脚本
├── trainval.py       # 训练/测试主入口
├── trainrl.py        # 强化学习相关入口
└── README.md
```

## 致谢

如果本项目对你的研究有帮助，欢迎在论文公开后引用本工作。

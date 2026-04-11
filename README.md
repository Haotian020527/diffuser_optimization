为了确保您可以一键无缝复制，以下是纯净的 Markdown 代码块，未添加任何其他对话文本。请直接点击代码框右上角的“复制”按钮：

````markdown
# diffuser_optimization

> `diffuser_optimization` 是一个面向移动操作的研究代码仓库。它以 `M2Diffuser` 为核心，统一实现了扩散式整段轨迹生成、物理约束优化、任务目标引导，以及 `MPiNets` / `MPiFormer` 基线对比。其中 **AMOGG** 是当前代码快照里最关键、最值得关注的采样引导模块。

---

## 📖 目录

- [项目简介](#-项目简介)
- [✨ 核心特性](#-核心特性)
- [🧠 AMOGG 深入解析](#-amogg-深入解析)
- [⚙️ 安装与环境配置](#️-安装与环境配置)
- [🚀 快速开始](#-快速开始)
- [📂 文件结构](#-文件结构)
- [📌 已知说明](#-已知说明)
- [🤝 贡献与协议](#-贡献与协议)
- [📑 引用与致谢](#-引用与致谢)

---

## 💡 项目简介

### 一句话说明
`M2Diffuser` 采用扩散模型生成整段机器人轨迹，并在采样阶段持续注入任务目标与物理约束，让生成的轨迹不仅“**到得了**”，还更可能“**走得通**”。

### 解决了什么问题？
在移动操作任务中，单纯预测下一步动作通常会面临以下挑战：
1. **累积误差**：一步偏离，后续整条轨迹都会产生漂移。
2. **目标冲突**：任务目标和物理约束容易产生冲突（例如，离目标越近可能意味着越容易发生碰撞）。
3. **约束滞后**：传统“先生成，再后处理”的流程将约束修正放得太晚，导致轨迹的物理可行性差。

**本仓库的核心解决方案：**
- 放弃单步预测，使用 `DDPM + UNet` 生成**整段轨迹**。
- 使用 `planner` 表达**任务目标**（如 `goal-reach`, `pick`, `place`）。
- 使用 `optimizer` 表达**物理约束**（如防碰撞、关节限制、动作幅度、平滑度）。
- 使用 **AMOGG** 在采样时自适应融合多个目标的梯度，解决硬性相加导致的冲突问题。

### 当前公开快照重点
本仓库目前提供最完整、可直接运行的 preset 为 **`goal-reach`**。
同时包含 `pick` / `place` 的预处理与结果汇总代码，以及 `M2Diffuser`、`MPiNets` 和 `MPiFormer` 三套完整的模型实现框架。

---

## ✨ 核心特性

- 🌪️ **扩散式整段轨迹生成**：`M2Diffuser` 从噪声出发，逐步反向采样出完整轨迹，而非单步贪心预测。
- ⚖️ **AMOGG 多目标引导**：在 `DDPM.p_sample()` 内部智能协调任务梯度与碰撞梯度，化解多目标冲突。
- 📊 **统一的基线对比框架**：集成 `M2Diffuser`、`MPiNets`、`MPiFormer`，确保公平一致的对比实验。
- 🧊 **3D 场景条件建模**：直接使用场景、物体、目标的 3D 点云作为几何输入条件。
- 🛡️ **物理感知评测**：内置基于 PyBullet 的评测脚本，精准统计碰撞率、越界、平滑度及耗时等核心物理指标。
- 🎛️ **Hydra 驱动配置**：模型、扩散器、任务、规划器与优化器均实现高度解耦的模块化管理。

<details>
<summary><strong>🔍 核心模块速览（点击展开）</strong></summary>

| 模块 | 核心作用 | 入口文件 |
|---|---|---|
| **Train** | 统一训练入口 | `train.py` |
| **Inference** | M2Diffuser 推理 | `inference_m2diffuser.py` |
| **DDPM** | 扩散模型与采样主逻辑 | `models/m2diffuser/ddpm.py` |
| **AMOGG** | AMOGG 算法数学实现 | `models/m2diffuser/amogg.py` |
| **Planner** | 任务目标引导 | `models/planner/mk_motion_policy_planning.py` |
| **Optimizer**| 物理约束优化 | `models/optimizer/mk_motion_policy_optimization.py` |
| **Env** | 评估与可视化环境 | `env/mk_motion_policy_env.py` |
| **Test** | AMOGG smoke test | `scripts/test_amogg_integration.py` |

</details>

---

## 🧠 AMOGG 深入解析

> AMOGG 是本仓库中最核心的创新增强模块。它并不替代 `planner` 或 `optimizer`，而是专注于解决**在采样时如何正确、高效地组合多个目标梯度**。

### 什么是 AMOGG？
AMOGG（**Adaptive Multi-Objective Gradient Guidance**）实现了：
- 时间感知加权 (Time-aware weighting)
- 冲突规避投影 (Conflict-avoidance projection)
- 可选的方差感知引导注入 (Variance-aware guidance injection)

### 为什么需要它？
在 `M2Diffuser` 采样中，主要存在两类引导信号：
1. **任务目标梯度** ($g_{task}$)：如靠近目标。
2. **物理约束梯度** ($g_{collision}$)：如避免碰撞。

若直接相加 ($g = g_{task} + g_{collision}$)，当两目标冲突时，会导致梯度抵消、采样震荡，或导致早期采样被约束压死。AMOGG 将两者融合为一个稳定的、带时序偏置的复合 Guidance。

### 处理流程

```mermaid
flowchart TD
    A["Conditioned DDPM Sampling"] --> B["Compute model_mean & variance"]
    B --> C["Task objective gradient (Planner)"]
    B --> D["Collision objective gradient (Optimizer)"]
    C --> E["AMOGG"]
    D --> E["AMOGG"]
    E --> F["Guided model_mean update"]
    F --> G["Sample x_(t-1)"]
````

*注：AMOGG 不参与训练损失，仅在**推理采样阶段**生效。*

### 核心数学逻辑

1.  **梯度计算**：分别计算 $g_1 = \nabla c_{task}$ 与 $g_2 = \nabla c_{collision}$。
2.  **冲突投影**：若 $\langle g_1, g_2 \rangle < 0$（存在冲突），将 $g_2$ 投影到 $g_1$ 的法平面，得到安全的非冲突分量 $g_{2\_safe}$。
3.  **时序融合**：利用激活函数（如 Sigmoid）根据当前扩散步数动态分配权重，确保早期注重整体任务，晚期注重精细约束。

### 如何验证 AMOGG

仓库内置了 Smoke test 来验证核心数学逻辑和 DDPM 分支：

```bash
python scripts/test_amogg_integration.py
```

\<details\>
\<summary\>\<strong\>📚 AMOGG 推荐阅读源码顺序\</strong\>\</summary\>

1.  `models/m2diffuser/amogg.py`
2.  `models/m2diffuser/ddpm.py`
3.  `models/planner/mk_motion_policy_planning.py`
4.  `models/optimizer/mk_motion_policy_optimization.py`
5.  `scripts/test_amogg_integration.py`

\</details\>

-----

## ⚙️ 安装与环境配置

**推荐环境**: Ubuntu 20.04 | NVIDIA GPU | Python 3.8.18 | CUDA 11.6 | PyTorch 1.13.1

### 1\. 克隆仓库

```bash
git clone [https://github.com/Haotian020527/diffuser_optimization.git](https://github.com/Haotian020527/diffuser_optimization.git)
cd diffuser_optimization
```

### 2\. 创建并安装环境

```bash
bash ./setup_env.sh
```

*该脚本将自动安装对应的 PyTorch, PyTorch-Lightning, Kaolin, PointNet2 ops 等依赖。*

### 3\. 修补 `yourdfpy`

根据项目要求，请修改 `yourdfpy/urdf.py` 中的 mesh scale 逻辑：

```python
# 修改前
new_s = new_s.scaled(geometry.mesh.scale)
# 修改后
new_s = new_s.scaled([geometry.mesh.scale[0], geometry.mesh.scale[1], geometry.mesh.scale[2]])
```

### 4\. 数据与路径配置

  * 下载 URDF / USD / Dataset / Checkpoints: [Hugging Face 资源入口](https://huggingface.co/datasets/M2Diffuser/mec_kinova_mobile_manipulation/tree/main)
  * 更新本地路径：在 `utils/path.py` 和 `configs/task/*.yaml` 中修改 `data_dir` 为你的本地绝对路径。

-----

## 🚀 快速开始

### 0\. 验证 AMOGG 环境

```bash
python scripts/test_amogg_integration.py
```

### 1\. 训练 M2Diffuser (Goal-reach 任务)

```bash
bash ./scripts/model-m2diffuser/goal-reach/train.sh 1
```

### 2\. 推理 M2Diffuser

```bash
# <CKPT_DIR> 为包含 last.ckpt 的路径
bash ./scripts/model-m2diffuser/goal-reach/inference.sh <CKPT_DIR>
```

\<details\>
\<summary\>\<strong\>🛠️ 3. 启用 AMOGG 的高级推理配置 (Hydra CLI)\</strong\>\</summary\>

你可以通过覆盖 Hydra 参数在推理中完整激活 AMOGG 特性：

```bash
python inference_m2diffuser.py hydra/job_logging=none hydra/hydra_logging=none \
  exp_dir=<CKPT_DIR> \
  task=mk_m2diffuser_goal_reach \
  diffuser=ddpm \
  diffuser.timesteps=50 \
  model=m2diffuser_mk \
  model.use_position_embedding=true \
  optimizer=mk_motion_policy_optimization \
  optimizer.scale_type=div_var \
  optimizer.collision=true \
  optimizer.collision_weight=0.03 \
  optimizer.collision_margin=0.02 \
  optimizer.joint_limits=true \
  optimizer.joint_limits_weight=0.1 \
  optimizer.smoothness=true \
  optimizer.smoothness_weight=0.1 \
  planner=mk_motion_policy_planning \
  planner.goal_reach_energy=true \
  planner.goal_reach_energy_type=last_frame \
  planner.goal_reach_energy_weight=0.005 \
  planner.goal_reach_energy_method=chamfer_distance \
  diffuser.sample.converage.optimization=true \
  diffuser.sample.converage.planning=true \
  diffuser.sample.converage.ksteps=2 \
  diffuser.sample.fine_tune.optimization=true \
  diffuser.sample.fine_tune.planning=true \
  diffuser.sample.fine_tune.timesteps=20 \
  diffuser.sample.fine_tune.ksteps=2 \
  diffuser.sample.amogg.enabled=true \
  diffuser.sample.amogg.kappa=10.0 \
  diffuser.sample.amogg.use_variance=true \
  diffuser.sample.amogg.task_scale=1.0 \
  diffuser.sample.amogg.collision_scale=1.0 \
  task.environment.sim_gui=false \
  task.environment.viz=false
```

\</details\>

### 4\. 运行 Baseline

```bash
# MPiNets
bash ./scripts/model-mpinets/goal-reach/train.sh 1
bash ./scripts/model-mpinets/goal-reach/inference.sh <CKPT_DIR>

# MPiFormer
bash ./scripts/model-mpiformer/goal-reach/train.sh 1
bash ./scripts/model-mpiformer/goal-reach/inference.sh <CKPT_DIR>
```

### 5\. 结果评估后处理

```bash
python ./postprocessing/eval_all_result_goal_reach.py \
  --result_dir ./results/<task_name>/<timestamp> \
  --dataset_test_dir <YOUR_DATASET_PATH>/goal-reach/test
```

-----

## 📂 文件结构

```text
diffuser_optimization/
├── assests/                         # 文档图示资源
├── configs/                         # Hydra 配置 (模型/任务/AMOGG等)
├── datamodule/                      # 数据集加载逻辑
├── env/                             # PyBullet环境、机器人与场景定义
├── eval/                            # 评测指标逻辑
├── models/
│   ├── m2diffuser/                  # M2Diffuser与AMOGG核心算法
│   ├── model/                       # UNet 与 Scene Model
│   ├── mpinets/                     # MPiNets 基线
│   ├── mpiformer/                   # MPiFormer 基线
│   ├── optimizer/                   # 物理约束优化器
│   └── planner/                     # 任务目标规划器
├── postprocessing/                  # 评测结果汇总脚本
├── preprocessing/                   # 数据预处理脚本
├── scripts/                         # 训练与推理 Bash 脚本
├── train.py                         # 统一训练入口
├── inference_*.py                   # 对应模型的推理入口
└── setup_env.sh                     # 环境一键配置脚本
```

-----

## 📌 已知说明

  - **快照状态**：`goal-reach` 是当前最完整、开箱即用的任务 preset。`pick` / `place` 的完整评测流程依赖的本地数据目录需用户自行补齐。
  - **路径配置**：当前 `utils/path.py` 使用了硬编码的绝对路径，运行前**必须替换**为你本地的数据源目录。
  - **评测引擎**：目前代码中的物理评测依赖于 PyBullet。README 中提及的 Isaac Sim / Tongverse 评测代码将在后续更新中补全。

-----

## 🤝 贡献与协议

### 贡献指南

我们非常欢迎任何形式的贡献，包括但不限于：Bug 修复、引入新任务场景、实现更强效的物理约束目标，或者开发 AMOGG 的新变体。

1.  Fork 本仓库。
2.  创建您的特性分支。
3.  提交代码后，请务必运行 `python scripts/test_amogg_integration.py` 确保核心逻辑未被破坏。
4.  提交 PR，并附上改动说明与复现命令。

### 协议

*(待补充：维护者将在后续更新中提供正式的 LICENSE 文件，预计采用 MIT 协议)*

-----

## 📑 引用与致谢

如果您在研究中使用了本项目，请引用我们的工作：

```bibtex
@article{yan2025m2diffuser,
  title={M2Diffuser: Diffusion-based Trajectory Optimization for Mobile Manipulation in 3D Scenes},
  author={Yan, Sixu and Zhang, Zeyu and Han, Muzhi and Wang, Zaijin and Xie, Qi and Li, Zhitian and Li, Zhehan and Liu, Hangxin and Wang, Xinggang and Zhu, Song-Chun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
```

**相关工作：**

  - [M³Bench: Benchmarking Whole-Body Motion Generation...](https://www.google.com/search?q=https://ieeexplore.ieee.org/document/10839185) (Zhang et al., IEEE RA-L 2025)
  - [Efficient task planning for mobile manipulation...](https://ieeexplore.ieee.org/document/9636665) (Jiao et al., IROS 2021)

**致谢：**
本仓库的实现参考了以下优秀开源项目：[SceneDiffuser](https://github.com/scenediffuser/Scene-Diffuser/), [MPiNets](https://github.com/NVlabs/motion-policy-networks), [Decision Transformer](https://github.com/kzl/decision-transformer), [VKC](https://github.com/zyjiao4728/Planning-on-VKC) 以及 [PhyScene](https://github.com/PhyScene/PhyScene)。

-----

\<div align="center"\>
\<i\>如果这个项目对你的研究有帮助，欢迎给它点个 ⭐️ Star！这将帮助更多扩散模型、移动操作、多目标引导领域的研究者看到它。\</i\>
\</div\>

```
```
# 从零训练一只机器狗：UIKA 强化学习工程入门

这是一份面向强化学习初学者的工程教程。你只需要会基础 Python 或 C 语言，不需要预先了解 PPO、Isaac Sim、Isaac Lab 或四足机器人控制。

教程使用 UIKA 四足机器人和 HimLoco 作为完整案例。完成环境配置后，你会依次理解并修改 observation、action、command、reward、termination、随机化和训练参数，最后能够设计一个可复现的对照实验。

本文侧重“如何把 RL 跑起来并改对”，理论推导可配合另行提供的视频或其他 RL 理论资料学习。

## 1. 开始前先确认版本

强化学习仿真项目对版本非常敏感。本教程固定使用下面的软件组合，不要把命令中的版本号改成 `latest`。

| 组件 | 本教程版本 |
|---|---|
| 操作系统 | Ubuntu 22.04 LTS |
| Python | 3.11 |
| Isaac Sim | 5.1.0 |
| PyTorch | 2.7.0，CUDA 12.8 构建 |
| Isaac Lab | v2.3.0 |
| 训练项目 | UIKA Lab，提交 `c1af265` |

开始前请确认：

- 电脑安装了 Ubuntu 22.04；
- NVIDIA GPU 可以正常使用；
- `nvidia-smi` 能显示显卡信息；
- 至少准备 50 GB 可用磁盘空间；
- 网络可以访问 GitHub、PyPI 和 NVIDIA Python Package Index。

确认上述条件后，直接从 conda 环境开始。

## 2. 先看懂整套环境的关系

后面会安装多个名字相似的工具。它们的关系如下：

```text
conda 环境
└── Python 3.11
    ├── pip
    ├── Isaac Sim 5.1      物理仿真器
    ├── PyTorch 2.7        神经网络与 GPU 计算
    ├── Isaac Lab 2.3      机器人学习框架
    └── himloco_lab        本教程的 UIKA 训练项目
```

可以把它们理解为：

- **conda**：为项目准备一个与其他项目隔离的工具箱；
- **Python**：运行训练代码的语言环境；
- **pip**：把 Python 软件包装进当前工具箱；
- **Isaac Sim**：负责机器人、地形、关节和接触的物理仿真；
- **PyTorch**：负责神经网络和 GPU 张量计算；
- **Isaac Lab**：在 Isaac Sim 上组织机器人、传感器、强化学习环境和并行仿真；
- **himloco_lab**：定义 UIKA 机器人任务、HimLoco 网络和训练流程。

安装必须按这个顺序进行。上一层没有验证成功时，不要继续安装下一层。

---

# 第一部分：配置 RL 环境

## 3. conda 是什么

同一台电脑上，不同项目可能需要不同版本的 Python 和 PyTorch。如果把所有软件都安装到系统 Python 中，升级一个项目时很容易破坏另一个项目。

conda 可以创建彼此隔离的环境。例如：

```text
base                conda 自己的基础环境
isaac_lab_51        本教程使用的 Python 3.11 环境
another_project     另一个项目自己的环境
```

本教程使用轻量的 Miniconda。官方资料：

- [Miniconda 官方页面](https://docs.conda.io/miniconda.html)
- [conda 官方 Linux 安装说明](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [conda 入门指南](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)

### 3.1 检查 conda 是否已经安装

打开终端，运行：

```bash
conda --version
```

如果输出类似：

```text
conda x.y.z
```

说明 conda 已经安装，可以跳到第 4 节。

如果提示 `conda: command not found`，继续安装 Miniconda。

### 3.2 安装 Miniconda

下载 Linux x86_64 安装脚本：

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

这里的 `latest` 只表示获取当前 Miniconda 安装器；项目使用的 Python 仍由下一节明确固定为 3.11。

运行安装程序：

```bash
bash Miniconda3-latest-Linux-x86_64.sh
```

安装过程中：

1. 按 Enter 阅读许可协议；
2. 输入 `yes` 接受协议；
3. 安装路径不确定时使用默认值；
4. 询问是否初始化 conda 时输入 `yes`。

安装完成后关闭并重新打开终端，再运行：

```bash
conda --version
```

如果仍然找不到 conda，可以执行：

```bash
source ~/.bashrc
```

然后再次检查。

## 4. 创建独立的 Python 环境

创建名为 `isaac_lab_51` 的环境，并指定 Python 3.11：

```bash
conda create -n isaac_lab_51 python=3.11
```

看到确认提示时输入 `y`。创建完成后激活环境：

```bash
conda activate isaac_lab_51
```

激活成功后，终端提示符前面通常会出现：

```text
(isaac_lab_51)
```

检查 Python：

```bash
python --version
which python
```

预期结果：

- Python 版本以 `3.11` 开头；
- Python 路径包含 `envs/isaac_lab_51`。

以后每次打开新终端，都要先执行：

```bash
conda activate isaac_lab_51
```

如果忘记激活环境，后面安装的软件可能进入错误的 Python 环境。

## 5. pip 是什么

pip 是 Python 的软件包安装工具。它可以从 Python Package Index（PyPI）或指定的软件源下载并安装 Python 包。

最常见的命令是：

```bash
python -m pip install 包名
```

本教程始终写成 `python -m pip`，而不是只写 `pip`。这样可以明确告诉系统：使用“当前这个 Python”对应的 pip，避免把包安装到其他 conda 环境。

例如：

```bash
python -m pip install "some-package==1.2.3"
```

其中：

- `install` 表示安装；
- `some-package` 是包名；
- `==1.2.3` 表示必须安装指定版本。

官方资料：

- [Python Packaging User Guide：安装 Python 包](https://packaging.python.org/en/latest/tutorials/installing-packages/)
- [pip 官方文档](https://pip.pypa.io/en/stable/)

先升级当前环境中的 pip：

```bash
python -m pip install --upgrade pip
```

检查 pip 属于哪个 Python：

```bash
python -m pip --version
```

输出路径应包含 `envs/isaac_lab_51`。

## 6. 安装 Isaac Sim 5.1

### 6.1 Isaac Sim 是什么

Isaac Sim 是 NVIDIA 的机器人仿真平台。它负责：

- 加载 UIKA 的 URDF 和网格模型；
- 模拟关节、电机、重力和碰撞；
- 生成地形与传感器数据；
- 在 GPU 上并行运行许多机器人环境。

本项目固定使用 Isaac Sim 5.1。请使用版本页面，不要照抄 `latest` 页面中的 6.x 命令。

官方资料：

- [Isaac Sim 5.1 安装总览](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/index.html)
- [Isaac Sim 5.1 Python/pip 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Sim 5.1 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html)

### 6.2 安装 Isaac Sim

确认终端前面有 `(isaac_lab_51)`，然后执行：

```bash
python -m pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

这个包体积很大，安装时间取决于网络和磁盘速度。`extscache` 会同时安装常用扩展缓存，减少第一次运行时的在线下载。

安装完成后检查版本：

```bash
python -c "import importlib.metadata as m; print(m.version('isaacsim'))"
```

预期输出以 `5.1.0` 开头。

先不要启动 Isaac Sim。按官方安装顺序装好下一节的 CUDA 版 PyTorch，再进行第一次启动。

## 7. 安装 CUDA 版 PyTorch

PyTorch 负责神经网络训练和 GPU 张量计算。Isaac Lab v2.3.0 的 x86_64 安装文档为这一软件组合指定 PyTorch 2.7.0 和 CUDA 12.8 wheel。

官方安装依据：

- [Isaac Lab v2.3.0：使用 Isaac Sim pip 包安装](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)

执行：

```bash
python -m pip install --upgrade torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

验证 PyTorch 和 CUDA：

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

正确结果应满足：

- PyTorch 版本以 `2.7.0` 开头；
- `CUDA available` 为 `True`；
- `GPU` 后显示实际显卡名称。

如果 `CUDA available` 为 `False`，不要继续安装 Isaac Lab。先确认当前环境中的 PyTorch 是否来自 `cu128` 索引，并确认 `nvidia-smi` 能正常工作。

### 7.1 第一次启动 Isaac Sim

运行：

```bash
isaacsim
```

第一次启动时会显示 NVIDIA Omniverse EULA。阅读后按提示接受，程序随后会准备扩展和缓存。第一次启动通常比后续启动慢。

看到 Isaac Sim 图形窗口说明基础安装成功。关闭窗口后继续下一步。

如果终端提示 `isaacsim: command not found`，先检查：

```bash
conda activate isaac_lab_51
python -m pip show isaacsim
```

## 8. 安装 Isaac Lab v2.3.0

### 8.1 Isaac Lab 是什么

Isaac Lab 是构建在 Isaac Sim 上的机器人学习框架。它提供：

- 机器人和场景配置；
- observation、action、reward 等 manager；
- 大规模并行强化学习环境；
- 与不同 RL 算法库连接的接口；
- 训练、回放和测试工具。

Isaac Sim 负责“物理世界”，Isaac Lab 负责“怎样把这个物理世界组织成一个学习任务”。

官方资料：

- [Isaac Lab v2.3.0 文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/)
- [Isaac Lab v2.3.0 pip 安装](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
- [Isaac Lab：创建自己的项目或任务](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/own-project/index.html)

### 8.2 准备 Git

Git 是版本控制工具。本教程用它下载 Isaac Lab 和 UIKA Lab，并记录每次实验对应的代码版本。

检查 Git：

```bash
git --version
```

如果提示 `git: command not found`，执行：

```bash
sudo apt update
sudo apt install git
```

官方资料：

- [Pro Git：Git 是什么](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)
- [Pro Git：安装 Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

### 8.3 获取 Isaac Lab 源码

先选择一个专门保存项目的目录：

```bash
mkdir -p ~/projects
cd ~/projects
```

克隆固定版本：

```bash
git clone --branch v2.3.0 --depth 1 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
```

检查版本：

```bash
git describe --tags --always
```

输出应包含 `v2.3.0`。

### 8.4 安装 Isaac Lab

Isaac Lab 的完整安装需要两个基础编译工具。先执行：

```bash
sudo apt install cmake build-essential
```

然后确认 conda 环境仍然是 `isaac_lab_51`，在 Isaac Lab 根目录执行：

```bash
./isaaclab.sh -i
```

安装完成后，运行官方空场景示例：

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

看到 Isaac Sim 窗口且程序没有 Python 异常，说明 Isaac Lab 已经能调用 Isaac Sim。

关闭仿真窗口后再继续，避免多个 Isaac Sim 进程同时占用显存。

## 9. 安装 himloco_lab

### 9.1 下载固定版本

```bash
cd ~/projects
git clone https://github.com/SAIKi0125/UIKA_lab.git himloco_lab
cd himloco_lab
git switch --create rl_course c1af2657f3c2f284429729a33407e59837ce5cad
```

最后一条命令从本教程核对过的提交创建本地 `rl_course` 分支，避免远端默认分支更新后出现任务名或参数不一致。学习教程时不要把远端默认分支合并进来。

确认当前目录：

```bash
pwd
ls
```

`ls` 应能看到 `README.md`、`scripts`、`source` 和 `docs` 等内容。

检查项目版本：

```bash
git rev-parse --short HEAD
```

预期输出为 `c1af265`。

### 9.2 安装项目包

```bash
python -m pip install -e source/himloco_lab
```

`-e` 表示 editable install。以后修改 `source/himloco_lab` 中的 Python 文件，一般不需要重新安装。

检查项目是否能被 Python 找到：

```bash
python -c "import himloco_lab; print(himloco_lab.__file__)"
```

输出路径应指向刚刚克隆的 `himloco_lab/source/himloco_lab`。

## 10. 环境总验收

不要只凭“安装命令没有报错”判断成功。按顺序执行下面的检查。

### 10.1 检查当前 Python 环境

```bash
conda activate isaac_lab_51
python --version
which python
python -m pip --version
```

Python 和 pip 路径都应包含 `envs/isaac_lab_51`。

### 10.2 检查关键包版本

```bash
python -c "import importlib.metadata as m; print('Isaac Sim:', m.version('isaacsim')); print('Isaac Lab package:', m.version('isaaclab')); print('PyTorch:', m.version('torch'))"
```

这里的 `Isaac Lab package` 是已安装 Python 扩展包的版本。源码发布版本仍以第 8.3 节中 Isaac Lab 仓库的 `git describe` 输出为准，两项都应与 v2.3.0 兼容线一致。

### 10.3 检查 GPU 计算

```bash
python -c "import torch; x=torch.ones(3, device='cuda'); print(x); print(torch.cuda.get_device_name(0))"
```

如果能输出 CUDA tensor 和显卡名称，说明 PyTorch GPU 计算可用。

### 10.4 检查任务注册

在 `himloco_lab` 根目录运行：

```bash
python - <<'PY'
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

import gymnasium as gym
import himloco_lab.tasks  # 注册 UIKA 任务

for task_id in sorted(spec.id for spec in gym.registry.values() if spec.id.startswith("UIKA-")):
    print(task_id)

simulation_app.close()
PY
```

列表中至少应出现：

```text
UIKA-Velocity
UIKA-Velocity-Play
UIKA-Flat-Velocity
UIKA-Flat-Velocity-Play
```

### 10.5 运行最小训练

```bash
python scripts/himloco_rsl_rl/train.py \
  --task UIKA-Flat-Velocity \
  --num_envs 16 \
  --max_iterations 1 \
  --headless \
  --run_name smoke
```

参数含义：

- `--task`：选择训练任务；
- `--num_envs 16`：并行模拟 16 只机器人；
- `--max_iterations 1`：只更新一次，用于检查链路；
- `--headless`：不打开图形窗口；
- `--run_name smoke`：给这次检查命名。

这一阶段不看机器人是否学会走路，只检查：

- 环境成功创建；
- 输出显示 `num_envs: 16`；
- 能完成 rollout 和一次参数更新；
- `logs/himloco_rsl_rl/uika_flat/` 中生成 `_smoke` 日志目录；
- 日志目录中存在 `params/env.yaml`、`params/agent.yaml` 和模型文件。

完成这一步后，RL 环境配置结束。

---

# 第二部分：认识 UIKA 强化学习工程

## 11. 先认识项目结构

你暂时不需要读完所有代码，只需要知道不同问题应该去哪里找。

```text
himloco_lab/
├── scripts/
│   ├── list_envs.py
│   └── himloco_rsl_rl/
│       ├── train.py
│       ├── play.py
│       └── play_interactive.py
├── source/himloco_lab/himloco_lab/
│   ├── assets/uika.py
│   ├── envs/
│   ├── rsl_rl/
│   └── tasks/locomotion/
│       ├── agents/
│       ├── mdp/
│       └── robots/uika/
└── logs/
```

常见入口：

| 想解决的问题 | 先看哪里 |
|---|---|
| 怎样启动训练 | `scripts/himloco_rsl_rl/train.py` |
| 怎样播放策略 | `scripts/himloco_rsl_rl/play.py` |
| UIKA 模型和执行器 | `assets/uika.py` |
| observation/action/reward 配置 | `robots/uika/velocity_env_cfg.py` |
| reward 函数实现 | `mdp/rewards.py` |
| PPO 和网络参数 | `agents/himloco_rsl_rl_cfg.py` |
| HimLoco estimator | `rsl_rl/modules/him_estimator.py` |
| PPO 更新逻辑 | `rsl_rl/algorithms/him_ppo.py` |

## 12. 一步强化学习中发生了什么

强化学习不是单独一个神经网络，而是一个不断重复的数据闭环：

```text
环境生成 observation
        ↓
策略根据 observation 输出 action
        ↓
Isaac Sim 执行 action 并推进物理世界
        ↓
环境计算 reward、下一步 observation 和 done
        ↓
PPO 使用收集到的数据更新策略
        ↓
进入下一轮
```

核心术语：

| 术语 | 在本项目中的含义 |
|---|---|
| environment | UIKA、地形、传感器和任务规则组成的环境 |
| observation | 策略当前能获得的信息 |
| action | 策略输出的 12 维关节动作 |
| command | 期望机器人执行的目标速度 |
| reward | 对当前行为的数值反馈 |
| done | 当前回合是否结束 |
| policy/actor | 根据 observation 生成 action 的网络 |
| critic | 估计当前状态长期价值的网络 |
| rollout | 用当前策略连续采集的一批数据 |

本项目中的主要数据流：

```text
Isaac Lab managers
  → HimlocoManagerBasedRLEnv
  → HimlocoVecEnvWrapper
  → HIMOnPolicyRunner
  → HIMPPO
  → HIMActorCritic / HIMEstimator
```

## 13. task 和 Scene：训练的是哪一个环境

训练命令中的：

```bash
--task UIKA-Flat-Velocity
```

不是随意字符串。它在：

```text
source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/__init__.py
```

中通过 `gym.register()` 注册。

`UIKA-Flat-Velocity` 会连接：

```text
任务名称
  → FlatRobotEnvCfg
  → UIKAFlatPPORunnerCfg
  → gym.make(...)
  → HimlocoVecEnvWrapper
  → HIMOnPolicyRunner
```

`RobotEnvCfg` 由多个 Isaac Lab manager 配置组成：

```python
scene
observations
actions
commands
rewards
terminations
events
curriculum
```

`scene` 中包含地形、UIKA 机器人、高度扫描器、接触传感器和灯光。平地任务通过继承基础 scene，把复杂地形替换为无限平面。

### 动手检查

分别运行一次最小训练：

```bash
python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity --num_envs 16 --max_iterations 1 --headless --run_name flat_smoke
python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity --num_envs 16 --max_iterations 1 --headless --run_name rough_smoke
```

比较两个日志目录中的 `params/env.yaml`，观察 terrain 和 curriculum 的区别。

## 14. Observation：策略看到了什么

HimLoco 的策略主要使用机器人本体感知信息。当前 UIKA policy 的单帧 observation 为：

| observation term | 含义 | 维度 |
|---|---|---:|
| `velocity_commands` | 目标 x/y 线速度与 z 角速度 | 3 |
| `base_ang_vel` | 机体角速度 | 3 |
| `projected_gravity` | 机体坐标系中的重力方向 | 3 |
| `joint_pos_rel` | 12 个关节相对默认角度 | 12 |
| `joint_vel_rel` | 12 个关节速度 | 12 |
| `last_action` | 上一步动作 | 12 |
| 合计 |  | 45 |

配置位于 `ObservationsCfg.PolicyCfg`。

critic 使用 `ObservationsCfg.CriticCfg`。它继承 policy observation，并加入仿真中可获得的线速度、外力和地形高度等 privileged information。

这叫 asymmetric actor-critic：

- actor 只使用部署时能够获得的信息；
- critic 在训练时使用更多信息，帮助估计 value；
- 真正部署时只运行 actor 和 estimator。

### scale、noise 和 clip

一个 observation term 可能包含：

```python
func
scale
noise
clip
```

阅读时依次问：

1. `func` 返回哪个物理量；
2. `scale` 为什么缩放；
3. `noise` 模拟什么传感误差；
4. `clip` 如何限制极端值。

### 历史 observation

当前 runner 配置：

```python
history_length = 5
```

这里的 `5` 表示历史区间长度 `H`。wrapper 按闭区间收集“当前帧 + 5 个过去时刻”，因此实际共有 6 帧：

```text
单帧维度：45
历史帧数：6
历史输入维度：45 × 6 = 270
```

论文写作 `o_{t-H:t}`，当 `H=5` 时同样包含 6 帧。这个例子不是论文与代码的差异，而是提醒你：不要把 `history_length` 误读成张量的总帧数，要沿代码和公式检查 shape。

## 15. Action：12 个输出怎样控制关节

UIKA 有 12 个主动关节，每条腿包含 hip、thigh 和 calf 三个关节。

策略 action 的数据流：

```text
12 维神经网络输出
  → action scale
  → 加到默认关节角
  → 得到目标关节位置
  → PD 控制器计算力矩
  → 电机模型限制最终输出
```

因此当前 action 表示目标关节位置偏置，不是直接输出力矩。

`ActionsCfg` 中的缩放为：

```python
scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
```

髋关节动作范围比其他关节更小。调整 action scale 时还必须一起检查：

- `assets/uika.py` 中的默认关节角；
- URDF 中的关节方向和限制；
- 执行器的 stiffness、damping、effort 和 velocity limit。

动作范围过小可能限制步幅，过大可能造成冲击、越界或仿真不稳定。

## 16. Command：告诉机器人做什么

command 定义任务目标。当前速度任务会采样：

```python
lin_vel_x=(-1.0, 1.0)
lin_vel_y=(-1.0, 1.0)
ang_vel_z=(-1.0, 1.0)
```

三个量分别表示：

- 前后方向线速度；
- 左右方向线速度；
- 绕竖直轴旋转的角速度。

command 通过 `velocity_commands` observation 告诉策略，同时被速度跟踪 reward 当作目标。

```text
command 采样
  ├── 进入 observation，告诉策略目标
  └── 进入 reward，计算实际速度与目标的差距
```

训练配置通常采样较宽的 command 范围；play 配置可以固定速度，便于观察某个动作。不要为了让回放只向前走，就把训练范围改成单一速度。

## 17. Reward：为什么机器人会形成某种行为

一个 reward term 包含三部分：

```python
track_lin_vel_xy = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=3.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
)
```

- `func`：如何计算原始数值；
- `params`：目标、阈值、传感器或 body；
- `weight`：这一项对总 reward 的影响。

当前 reward 可以分为：

| 目标 | 代表项 |
|---|---|
| 跟踪任务 | 线速度、角速度跟踪 |
| 保持机体稳定 | 竖直速度、姿态、机体高度 |
| 保护关节和电机 | 力矩、功率、加速度、关节限制 |
| 动作平滑 | `action_rate_l2` |
| 接触安全 | 非足端接触、足端冲击、打滑 |
| 塑造步态 | 腾空时间、足端高度、对角步态 |

reward 是多个目标的折中。某一项改善，可能以另一项变差为代价。

### 推荐的第一个 reward 实验

从 `action_rate_l2` 开始：

1. 先保留原权重训练 baseline；
2. 只修改 `action_rate_l2` 权重；
3. 两次实验使用相同 task、seed、环境数和迭代数；
4. 比较动作变化率、速度跟踪和回放抖动；
5. 检查平滑性是否换来了更慢的速度响应。

一次只改变一个主要变量，才能说明结果可能由什么造成。

## 18. Termination、Event 和 Curriculum

三者都会改变采集到的数据，但作用不同：

| 模块 | 作用 |
|---|---|
| termination | 判断一个 episode 何时结束 |
| event | 在启动、reset 或固定间隔执行随机化/扰动 |
| curriculum | 根据表现逐步改变任务难度 |

### Event 的时间模式

本项目包含：

- `startup`：启动时随机化摩擦、质量和质心；
- `reset`：每回合重置关节、base 和执行器参数；
- `interval`：训练中周期性施加外力或推机器人。

这些随机化用于让策略适应模型误差和环境变化。随机范围不是越大越好：范围太小可能缺乏鲁棒性，范围太大可能让训练任务难以学习或产生不真实状态。

curriculum 则让机器人从较简单地形开始，表现达到条件后再进入更困难的地形。

## 19. HimLoco：为什么要使用历史信息

本项目参考 ICLR 2024 论文：

- [Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response](https://arxiv.org/abs/2312.11460)
- [HimLoco 项目主页](https://junfeng-long.github.io/HIMLoco/)
- [HimLoco 官方代码](https://github.com/InternRobotics/HIMLoco)

真实机器人难以直接获得准确的地形高度、摩擦等外部状态。HimLoco 从连续多帧本体感知中提取机器人对环境的响应：

- 3 维显式速度估计；
- 16 维隐式 latent representation。

当前网络数据流：

```text
6 帧历史 observation（270 维）
        ↓
HIMEstimator
  ├── 3 维 velocity
  └── 16 维 latent
        ↓
当前 observation（45）+ velocity（3）+ latent（16）
        ↓
Actor MLP（64 维输入）
        ↓
12 维 action
```

论文中的概念与本仓库代码对应：

| 论文概念 | 本仓库代码 |
|---|---|
| partial observation | `ObservationsCfg.PolicyCfg` |
| privileged information | `ObservationsCfg.CriticCfg` |
| historical observation | `HimlocoVecEnvWrapper` |
| explicit velocity + implicit latent | `HIMEstimator` |
| velocity loss | `F.mse_loss(pred_vel, vel)` |
| prototype contrastive objective | `proto`、`sinkhorn()`、`swap_loss` |
| actor input | `HIMActorCritic.update_distribution()` |
| HIO + PPO | `HIMPPO.update()` |

论文和本仓库不是逐行一致的实现。两者的 `H=5` 都表示 `o_{t-H:t}` 的 6 帧闭区间；一个实际差异是论文训练使用 100 步 rollout，而当前 UIKA 配置使用每环境 24 步。工程判断应以本仓库实际代码和日志为准。

## 20. PPO 训练循环

UIKA 使用 on-policy PPO。每次训练迭代大致分为：

1. 多个环境并行运行 `num_steps_per_env` 步；
2. 保存 observation、action、reward、done、value 和 log probability；
3. 根据最后一个 value 计算 return 和 advantage；
4. 将 rollout 切成多个 mini-batch；
5. 更新 HIM estimator；
6. 计算 PPO clipped surrogate loss；
7. 计算 value loss 和 entropy bonus；
8. 更新 actor 和 critic；
9. 写入日志并定期保存 checkpoint。

主要参数位于：

```text
source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_rsl_rl_cfg.py
```

UIKA 当前关键参数包括：

```text
num_steps_per_env = 24
history_length = 5
clip_param = 0.2
gamma = 0.99
lam = 0.95
learning_rate = 5e-4
```

第一次学习这些参数时，先理解它们控制哪一部分，不要同时调多个超参数。

---

# 第三部分：训练、观察与实验

## 21. 根据显卡调整并行环境数

项目默认可能使用 4096 个并行环境，但不同显卡的可用显存不同。不要一开始直接使用最大配置。

建议逐级测试：

| 用途 | 建议起点 |
|---|---:|
| 安装冒烟测试 | 16 |
| 代码调试 | 64 |
| 短实验 | 256 或 512 |
| 正式训练 | 逐步尝试 1024、2048、4096 |

通过命令行覆盖：

```bash
python scripts/himloco_rsl_rl/train.py \
  --task UIKA-Flat-Velocity \
  --num_envs 256 \
  --max_iterations 100 \
  --headless \
  --run_name short_log_demo
```

如果出现 CUDA OOM，先关闭其他 Isaac Sim 进程，再降低 `--num_envs`。

这条命令用于练习查看日志和 checkpoint。100 次迭代通常不足以让策略稳定行走，不要把它当作训练完成。

准备正式训练时，使用单独的 run 名称，并从机器能够稳定承受的环境数开始。例如：

```bash
python scripts/himloco_rsl_rl/train.py \
  --task UIKA-Flat-Velocity \
  --num_envs 1024 \
  --headless \
  --run_name flat_baseline
```

不写 `--max_iterations` 时使用当前配置的上限 50000 次；这不是保证必须跑满的固定答案。项目每 100 次迭代保存一次模型。结合 mean reward、episode length、速度跟踪项和阶段性回放判断是否继续，而不要只看训练时间。

环境数减少后，每轮收集的样本也会减少。因此 16 或 64 环境适合检查程序，不适合直接与 4096 环境的完整训练结果比较。

## 22. 使用 TensorBoard 看训练

在项目根目录启动：

```bash
tensorboard --logdir logs/himloco_rsl_rl --port 6006
```

浏览器打开：

```text
http://localhost:6006
```

可以从四类指标开始：

| 类别 | 关注内容 |
|---|---|
| 任务表现 | episode reward、速度跟踪、episode length |
| PPO | surrogate loss、value loss、learning rate、mean noise std |
| HimLoco | estimation loss、swap loss |
| 性能 | FPS、collection time、learning time |

不要要求所有 loss 单调下降。强化学习的数据分布会随策略变化。判断训练是否正常，需要结合长期趋势、是否出现 NaN、最终 checkpoint 回放以及各 reward term 的变化。

## 23. 播放 checkpoint

训练日志默认保存在：

```text
logs/himloco_rsl_rl/实验名/运行目录/
```

推荐明确指定要检查的 run：

```bash
python scripts/himloco_rsl_rl/play.py \
  --task UIKA-Flat-Velocity-Play \
  --load_run 2026-01-01_12-00-00_flat_baseline \
  --num_envs 4
```

将示例中的 run 目录替换为自己的实际名称。没有指定 checkpoint 文件时，脚本会从该 run 中选择匹配的最新模型。

也可以省略 `--load_run`，让脚本播放 `uika_flat` 下名称排序最后的 run：

```bash
python scripts/himloco_rsl_rl/play.py \
  --task UIKA-Flat-Velocity-Play \
  --num_envs 4
```

这种便利写法可能选中最新的 smoke 或短练习目录，因此做正式比较时应显式写 `--load_run`。短练习 checkpoint 可以用来验证回放链路，但机器人不会因此必然已经学会稳定行走。

回放时同时观察：

- 是否跟随目标速度；
- 机体是否稳定；
- 步态是否平滑；
- 足端是否打滑；
- 是否出现关节极限或异常接触；
- 改动改善了什么，又损失了什么。

## 24. 做一个可靠的对照实验

一个清楚的实验从具体问题开始，例如：

- 机器人平地行走时动作抖动明显；
- 低速指令跟踪较差；
- 静止指令下仍频繁抬脚；
- 足端打滑较多。

不要使用“把 reward 调好”作为问题，因为它没有可验证的终点。

推荐流程：

1. 用一句话定义问题；
2. 选择一个主要指标和一个副作用指标；
3. 记录 baseline 的 task、seed、环境数和训练预算；
4. 写下修改前的预测；
5. 一次只改变一个主要因素；
6. 使用相同条件训练 treatment；
7. 检查两个 run 保存的 `env.yaml` 和 `agent.yaml`；
8. 同时比较 TensorBoard 和回放行为；
9. 写明结论只适用于哪些 task、seed 和训练预算。

推荐记录格式：

```markdown
# 实验标题

## 问题

## 假设与预测

## Baseline

## 唯一主要改动

## 运行命令

## TensorBoard 结果

## 回放行为

## 结论与副作用

## 局限和下一步
```

---

# 第四部分：常见问题

## 25. 为什么 `python` 找不到包

先执行：

```bash
conda activate isaac_lab_51
which python
python -m pip --version
```

Python 和 pip 必须来自同一个 `isaac_lab_51` 环境。

如果找不到 `himloco_lab`，在项目根目录重新执行：

```bash
python -m pip install -e source/himloco_lab
```

## 26. 为什么找不到 task

重新执行第 10.4 节的任务注册检查。如果 UIKA task 不在输出中，检查：

- 是否位于正确仓库；
- 项目是否安装到当前 conda 环境；
- `himloco_lab.tasks` 是否能导入；
- task 注册文件是否包含对应 ID。

## 27. CUDA Out of Memory

处理顺序：

1. 关闭其他 Isaac Sim 和训练进程；
2. 把 `--num_envs` 降到一半；
3. 调试时使用 `--headless`；
4. 确认没有启用不需要的相机或渲染；
5. 再次运行。

Isaac Lab 官方训练指南也建议在 OOM 时优先减少并行环境数：

- [Isaac Lab 调试与训练指南](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/overview/reinforcement-learning/training_guide.html)

## 28. 出现 NaN

NaN 常见来源：

- observation 中出现非法数值；
- action 过大导致仿真不稳定；
- 机器人 reset 到无效姿态；
- 关节限制、PD 参数或物理参数不合理；
- reward 计算包含除零或无效输入；
- 学习率或梯度异常。

先找到 traceback 中第一个属于本项目的文件位置，再检查该位置附近的张量。不要只看最后一行异常，也不要同时修改多个参数碰运气。

## 29. 为什么总 reward 上升但机器人行为不好

总 reward 是多个 term 的加权和。它上升可能表示策略找到了某种容易获得分数、但不符合最终目标的行为。

检查：

- 每个 reward term 的变化；
- 速度跟踪是否真的改善；
- episode 是否因为异常终止变短；
- 是否牺牲动作平滑性或能耗；
- 回放行为是否满足原始工程目标。

“程序能跑”“reward 上升”“机器人达到目标”是三个不同结论。

---

# 术语速查

| 术语 | 简明解释 |
|---|---|
| conda environment | 相互隔离的 Python 和软件包环境 |
| pip | Python 包安装工具 |
| package | 可安装和复用的软件模块 |
| simulator | 计算机器人与物理世界变化的软件 |
| environment | 强化学习中的机器人、世界和任务规则 |
| observation | 策略做决定时获得的信息 |
| action | 策略交给环境的控制输出 |
| command | 当前要求机器人完成的目标 |
| reward | 对当前行为的数值反馈 |
| episode | 从 reset 到下一次结束的一段交互 |
| rollout | 用当前策略收集的一批连续数据 |
| actor/policy | 根据 observation 产生 action 的网络 |
| critic/value network | 估计状态长期价值的网络 |
| privileged information | 训练可用、部署时不直接给 actor 的信息 |
| domain randomization | 随机改变仿真参数以增强鲁棒性 |
| curriculum | 根据表现逐步改变任务难度 |
| checkpoint | 训练过程中保存的模型文件 |
| baseline | 用于比较的原始设置 |
| ablation | 移除或屏蔽一个因素以检验其作用 |

# 官方资料索引

- [Miniconda 官方页面](https://docs.conda.io/miniconda.html)
- [conda Linux 安装](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/installing-packages/)
- [pip 官方文档](https://pip.pypa.io/en/stable/)
- [Isaac Sim 5.1 Python 安装](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
- [Isaac Lab v2.3.0 pip 安装](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/pip_installation.html)
- [Isaac Lab v2.3.0 文档](https://isaac-sim.github.io/IsaacLab/v2.3.0/)
- [Pro Git](https://git-scm.com/book/en/v2)
- [HimLoco 论文](https://arxiv.org/abs/2312.11460)
- [HimLoco 官方代码](https://github.com/InternRobotics/HIMLoco)

版本页面中的 5.1 和 v2.3.0 是本教程的一部分。即使官方已经发布更新版本，也不要在没有完成兼容性验证时单独升级某个组件。

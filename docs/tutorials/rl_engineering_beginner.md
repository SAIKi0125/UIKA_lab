# 从零训练一只机器狗：UIKA 强化学习工程入门

> 面向只学过基础 Python/C、第一次使用 Ubuntu 和强化学习框架的读者

## 开始之前

这份教程不要求你先学完所有强化学习理论。你会一边运行 UIKA 四足机器人，一边认识 command、observation、action、reward、termination 和 PPO。理论视频与阅读材料由课程主页统一维护；本文只在需要时指出“现在应该先看哪类理论”。

学完后，你应该能独立回答并验证以下问题：

- `--task UIKA-Flat-Velocity` 最终加载了哪些 Python 类？
- 策略每一步能看到什么，critic 为什么能看到更多信息？
- 神经网络输出的 12 个数怎样变成关节目标角？
- 机器人为什么会追踪速度、抬脚、少打滑？
- 如何只改变一个因素，比较两次训练？
- OOM、任务未注册、包导入失败和 NaN 应该从哪里查？

不要把“命令没有报错”等同于“实验成功”。本课程区分三层结果：

1. **程序成功**：环境能够创建，训练循环能够执行；
2. **学习成功**：策略指标随训练发生合理变化；
3. **任务成功**：回放行为满足预先定义的工程目标。

---

## 课程地图

| 课次 | 主题 | 你要解决的问题 | 主要文件 |
|---|---|---|---|
| 0 | Ubuntu 与环境 | 如何证明电脑能运行课程项目？ | 本章命令 |
| 1 | 第一次训练 | 一条训练命令背后发生了什么？ | `train.py`、任务注册 |
| 2 | RL 数据闭环 | 一步仿真中数据怎样流动？ | env、wrapper、runner |
| 3 | Scene 与 task | UIKA、地形和传感器从哪里来？ | `velocity_env_cfg.py` |
| 4 | Observation | actor 和 critic 分别看到了什么？ | `ObservationsCfg` |
| 5 | Action | 网络输出怎样控制 12 个关节？ | `ActionsCfg`、`uika.py` |
| 6 | Command | “向前走”如何成为训练任务？ | `CommandsCfg` |
| 7 | Reward | 为什么机器人会形成某种行为？ | `RewardsCfg`、`rewards.py` |
| 8 | Termination/Event/Curriculum | 何时重置，为什么环境每次不同？ | 三类 manager 配置 |
| 9 | HimLoco 与 PPO | 论文中的 HIM 和 PPO落在何处？ | estimator、actor、PPO |
| 10 | 综合实验 | 如何有证据地改进一个行为？ | 完整实验链路 |

建议每次课都保留四样东西：你的预测、完整命令、实验日志、对结果的解释。

---

# 第 0 课：Ubuntu、终端与可复现环境

## 0.1 本节完成标准

本节结束时，你需要提交一份环境验收记录，其中每条命令都有输出，并满足：

- Ubuntu 为课程指定版本；
- `nvidia-smi` 能识别 NVIDIA GPU；
- 当前终端使用课程 conda 环境中的 Python；
- Isaac Sim、Isaac Lab 和 `himloco_lab` 都能被同一个 Python 找到；
- `scripts/list_envs.py` 能列出 UIKA 任务；
- 16 个并行环境能够完成 1 次训练迭代。

## 0.2 先学会读一条命令

打开终端后，你通常会看到类似下面的提示符：

```text
student@computer:~/project$
```

`$` 前面是提示信息，不需要输入。下面这条命令由三部分组成：

```bash
python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity
```

- `python`：要运行的程序；
- `scripts/himloco_rsl_rl/train.py`：交给 Python 的脚本路径；
- `--task UIKA-Flat-Velocity`：传给脚本的选项和值。

本教程中：

- 代码块里的命令可以输入终端；
- `#` 后面是解释，不是命令的必要部分；
- 不理解的命令先查含义，不要在路径开头随意加 `sudo`。

## 0.3 五个够用的 Ubuntu 命令

```bash
pwd                 # 我现在位于哪个目录？
ls                  # 当前目录有哪些文件？
cd path/to/folder   # 进入一个目录
cd ..               # 回到上一级目录
mkdir experiment    # 新建一个目录
```

路径有两种写法：

- `/home/student/project` 是绝对路径，从系统根目录 `/` 开始；
- `scripts/train.py` 是相对路径，从当前目录开始。

训练前先运行 `pwd` 和 `ls`。如果当前目录不是 `himloco_lab` 仓库根目录，相对路径就可能找不到。

## 0.4 课程冻结的软件栈

本仓库当前工程线使用以下基线：

| 组件 | 课程基线 |
|---|---|
| 操作系统 | Ubuntu 22.04 LTS |
| Python | 3.11 |
| Isaac Sim | 5.0.0 |
| Isaac Lab | 课程固定 commit `d94504bcf91cb7ab7ff956a2d48ecd1bca82797a` |
| 项目 | 与本文档位于同一个 Git commit 的 `himloco_lab` |

不要把版本号中的“旧”理解成“错误”。2026 年的最新 Isaac Lab 已经面向 Isaac Sim 6.x，但本项目包含 Isaac Sim 5.0/5.1 的兼容逻辑。入门课首先追求全班可复现；升级框架属于独立迁移任务。

教师发布课程时应给项目 commit 打标签，并要求学生记录 `git rev-parse HEAD` 的输出。文档和代码必须一起冻结，不能只复制本文而继续使用其他版本的项目代码。

## 0.5 安装前检查

先执行：

```bash
lsb_release -ds
nvidia-smi
ldd --version
free -h
df -h
```

你在检查五件事：系统版本、GPU/驱动、GLIBC、内存、磁盘空间。Isaac Sim 5.0 的 pip 包要求 Python 3.11 和 GLIBC 2.35 以上；Ubuntu 22.04 满足对应 GLIBC 基线。

如果 `nvidia-smi` 报错，先处理驱动，不要继续安装 Python 包。此时问题仍在“硬件/驱动层”，重装项目代码不能解决它。

## 0.6 conda 是什么

conda 环境可以理解为一个独立的 Python 工具箱。不同项目把不同版本的 Python 和包放在各自工具箱中，避免互相覆盖。

安装 Miniconda 后创建课程环境：

```bash
conda create -n isaac_lab_50 python=3.11
conda activate isaac_lab_50
python --version
which python
```

最后一条应指向类似下面的路径：

```text
.../miniconda3/envs/isaac_lab_50/bin/python
```

如果打开新终端后命令失效，首先重新执行：

```bash
conda activate isaac_lab_50
```

## 0.7 安装 Isaac Sim 5.0

在已激活的 `isaac_lab_50` 环境中执行：

```bash
python -m pip install --upgrade pip
python -m pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
```

这里坚持使用 `python -m pip`，是为了明确把包安装到当前 `python` 所属环境。

验证版本：

```bash
python -c "import importlib.metadata as m; print(m.version('isaacsim'))"
```

预期输出以 `5.0.0` 开头。第一次启动需要接受 NVIDIA EULA，也可能花较长时间准备扩展缓存；这不等同于程序卡死。

## 0.8 安装固定版本的 Isaac Lab

选择一个专门放代码的目录：

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
git checkout d94504bcf91cb7ab7ff956a2d48ecd1bca82797a
./isaaclab.sh -i
```

验证当前 commit：

```bash
git rev-parse HEAD
```

验证 Isaac Lab 能创建应用：

```bash
python scripts/tutorials/00_sim/create_empty.py
```

出现 Isaac Sim 窗口并能正常关闭，说明 Isaac Sim 与 Isaac Lab 的基本连接成立。关闭仿真后再继续，避免多个 Isaac Sim 进程占用显存。

## 0.9 安装本项目

回到代码目录，克隆课程指定仓库：

```bash
cd ~/projects
git clone https://github.com/SAIKi0125/UIKA_lab.git himloco_lab
cd himloco_lab
python -m pip install -e source/himloco_lab
```

`-e` 表示 editable install。修改 `source/himloco_lab` 下的 Python 代码后，一般不需要重复安装。

查看项目任务：

```bash
python scripts/list_envs.py
```

至少应看到：

```text
UIKA-Velocity
UIKA-Velocity-Play
UIKA-Flat-Velocity
UIKA-Flat-Velocity-Play
```

## 0.10 最小训练验收

```bash
python scripts/himloco_rsl_rl/train.py \
  --task UIKA-Flat-Velocity \
  --num_envs 16 \
  --max_iterations 1 \
  --headless \
  --run_name smoke
```

反斜杠 `\` 表示这条命令还没有结束，下一行仍属于同一条命令。

验收重点不是 reward 高低，而是：

- 环境成功创建；
- 输出包含 `num_envs: 16`；
- 输出显示单帧 observation、历史 observation 和 action 维度；
- 完成一次 rollout 和 update；
- `logs/himloco_rsl_rl/uika_flat/` 下出现带 `_smoke` 的目录；
- 目录中存在 `params/env.yaml`、`params/agent.yaml` 和模型文件。

## 0.11 环境问题分层

| 现象 | 先检查 | 常见层级 |
|---|---|---|
| `nvidia-smi` 失败 | 驱动是否加载 | 驱动/硬件 |
| `python` 版本不对 | `which python`、conda 是否激活 | Python 环境 |
| `No module named isaacsim` | `pip show isaacsim` | Python 环境 |
| `No module named isaaclab` | Isaac Lab 是否安装在同一环境 | Isaac Lab |
| 找不到 `himloco_lab` | editable install、当前解释器 | 项目安装 |
| task 不存在 | `scripts/list_envs.py`、任务注册 | 项目配置 |
| CUDA OOM | 降低 `--num_envs`、关闭其他仿真进程 | 资源 |
| observation 中出现 NaN | reset、关节限制、动作、物理参数 | 项目/仿真 |

排错时一次只改变一件事，并保存完整错误的第一处 traceback。最后一行告诉你异常类型，最前面的项目文件位置通常告诉你问题从哪里进入。

---

# 第 1 课：第一次训练——从命令追到环境

## 1.1 本节问题

当你运行：

```bash
python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity
```

程序怎样知道要加载 UIKA、平地和 HimLoco PPO？

## 1.2 先画出入口链

按下面顺序打开文件，不要一开始就阅读每一行：

1. `scripts/himloco_rsl_rl/train.py`
2. `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/__init__.py`
3. `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/flat_env_cfg.py`
4. `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py`
5. `source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_rsl_rl_cfg.py`

`gym.register()` 把字符串 ID 与环境配置、算法配置连接起来。对于 `UIKA-Flat-Velocity`：

```text
任务字符串
  → FlatRobotEnvCfg
  → UIKAFlatPPORunnerCfg
  → gym.make(...)
  → HimlocoVecEnvWrapper
  → HIMOnPolicyRunner
```

## 1.3 认识命令行覆盖

配置中默认 `num_envs=4096`，但训练入口执行：

```python
env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
```

因此：

```bash
--num_envs 64
```

会在运行时覆盖默认值。这类覆盖适合临时实验；需要成为长期任务定义的改动才应写入配置文件。

## 1.4 实验：证明覆盖确实生效

分别运行：

```bash
python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity --num_envs 16 --max_iterations 1 --headless --run_name env16
python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity --num_envs 64 --max_iterations 1 --headless --run_name env64
```

记录两次输出中的：

- `num_envs`；
- 每次迭代采集的样本数；
- FPS；
- 显存占用；
- 一次迭代耗时。

这一实验只能说明吞吐和资源差异，不能用 1 次迭代判断哪个策略更好。

---

# 第 2 课：RL 不是一个网络，而是一个数据闭环

## 2.1 本节问题

策略为什么能够“学”？答案不在某一个类里，而在循环中：

```text
observation → policy → action → simulator → reward/next observation
       ↑                                      ↓
       └──────────── PPO/HIO 更新参数 ─────────┘
```

在本项目中，关键数据流为：

```text
Isaac Lab managers
  → HimlocoManagerBasedRLEnv
  → HimlocoVecEnvWrapper（堆叠历史帧）
  → HIMOnPolicyRunner（采集 rollout）
  → HIMPPO（计算 return、advantage 和 loss）
  → HIMActorCritic / HIMEstimator
  → 产生下一批 action
```

## 2.2 把术语落到张量

| 术语 | 本项目中的含义 | 常见形状 |
|---|---|---|
| observation | 策略当前可用的信息 | `[num_envs, obs_dim]` |
| action | 每个环境中 12 个关节的策略输出 | `[num_envs, 12]` |
| reward | 每个环境当前一步的标量反馈 | `[num_envs]` |
| done | 哪些环境需要结束并重置 | `[num_envs]` |
| rollout | 多个环境连续若干步的数据 | 环境数 × 步数 |
| policy | 根据 observation 给出 action 分布 | 神经网络 |
| critic | 估计当前状态价值 | 神经网络 |

形状中的第一维是并行环境。Isaac Lab 同时模拟许多 UIKA，不是为了在画面里热闹，而是为了并行收集经验。

## 2.3 代码追踪练习

在以下位置找到相邻的四步：

1. `HIMOnPolicyRunner.learn()` 调用 `self.alg.act(...)`；
2. runner 调用 `self.env.step(actions)`；
3. `HIMPPO.process_env_step()` 把 transition 放入 storage；
4. rollout 结束后执行 `compute_returns()` 和 `update()`。

提交一张你自己画的数据流图。每条箭头都要写数据名，不能只画类名。

---

# 第 3 课：Scene、task 与 manager 配置

## 3.1 一个环境由什么组成

`RobotEnvCfg` 把多个 manager 配置组合在一起：

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

这正是后续课程的工程地图。找不到一个行为的配置时，先判断它属于哪类 manager。

## 3.2 Scene 中有哪些东西

`RobotSceneCfg` 至少包含：

- terrain：地面或生成地形；
- robot：UIKA articulation；
- height scanner：地形高度射线；
- contact sensor：接触力、触地和腾空时间；
- light：可视化光源。

`FlatRobotSceneCfg` 继承它，但把 terrain 改成无限平面。这说明继承不是“复制代码”，而是保留共同配置并替换差异。

## 3.3 实验：平地与复杂地形不是同一个变量

先比较两个任务：

```bash
python scripts/himloco_rsl_rl/train.py --task UIKA-Flat-Velocity --num_envs 16 --max_iterations 1 --headless --run_name flat_smoke
python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity --num_envs 16 --max_iterations 1 --headless --run_name rough_smoke
```

比较保存的 `params/env.yaml`，回答：

- terrain type 有什么不同？
- curriculum 是否生效？
- critic 的 height scanner 是否仍然存在？
- 两个任务是否使用同一套 action、observation 和 reward？

这一步练习“读取最终配置”，不要仅凭类名猜测。

---

# 第 4 课：Observation——策略究竟看到了什么

## 4.1 论文问题

HimLoco 论文强调：真实机器人难以直接获得准确地形摩擦、恢复系数和高度图，因此 policy 主要依赖关节编码器和 IMU 的本体感知；训练阶段的 value network 可以使用额外 privileged information。

本仓库用两个 observation group 表达这一点：

- `PolicyCfg`：给 actor；
- `CriticCfg`：继承 policy 内容，再加入额外信息给 critic。

## 4.2 policy 单帧 observation

| term | 含义 | 维度 |
|---|---|---:|
| `velocity_commands` | 目标 x/y 线速度与 z 角速度 | 3 |
| `base_ang_vel` | 机体角速度 | 3 |
| `projected_gravity` | 机体坐标系中的重力方向 | 3 |
| `joint_pos_rel` | 12 个关节相对默认角度 | 12 |
| `joint_vel_rel` | 12 个关节速度 | 12 |
| `last_action` | 上一步 12 维动作 | 12 |
| 合计 |  | 45 |

`base_lin_vel` 没有放入 policy，而是出现在 critic group 中。不要把“仿真中能读取”误认为“部署时 policy 应该直接使用”。

## 4.3 scale、noise、clip 的顺序意识

一个 observation term 不只有函数：

```python
base_ang_vel = ObsTerm(
    func=mdp.base_ang_vel,
    scale=0.25,
    clip=(-100, 100),
    noise=Unoise(n_min=-0.2, n_max=0.2),
)
```

读配置时分别问：

- 原始物理量是什么？
- 为什么缩放？
- 噪声模拟哪类传感误差？
- clip 是正常范围设计，还是极端值保护？

## 4.4 历史长度的工程陷阱

runner 中写着：

```python
history_length = 5
```

但 wrapper 的定义是“当前帧 + `history_length` 个过去帧”，所以：

```text
单帧维度 = 45
实际帧数 = 5 + 1 = 6
actor 历史输入维度 = 45 × 6 = 270
```

论文写默认 `H=5`，本仓库实际传给 estimator 的帧数是 6。这里不要争论变量名“应该是什么意思”，而要沿构造函数和张量 shape 证明实际行为。

## 4.5 实验：屏蔽一项观测前先做预测

选择一个风险较低的观测，例如在教学分支中把 `last_action` 临时替换为同维零向量。实验前写下：

- 你预测动作平滑性、训练速度或最终行为会怎样？
- 哪些 TensorBoard 指标可能变化？
- 哪些现象不能由短训练判断？

保持 task、seed、`num_envs`、迭代数和 reward 不变，运行 baseline 与 ablation。不要同时改变 observation noise。

---

# 第 5 课：Action——12 个数怎样控制机器人

## 5.1 action 不是力矩

本任务使用 `JointPositionActionCfg`。策略输出经过 scale，再加到默认关节角上，成为 PD 控制器的目标位置。概念链如下：

```text
policy output
  → action scale
  → default joint position offset
  → target joint position
  → PD actuator
  → joint torque（受电机能力限制）
```

论文将动作描述为“目标关节位置相对 nominal position 的偏置”，本仓库与这一思想对应。

## 5.2 为什么髋关节和其他关节 scale 不同

当前配置中：

```python
scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
```

髋关节的动作幅度更小。动作缩放越大，策略可探索的目标角范围越大，但动作冲击、越界和仿真不稳定风险也可能增加。

## 5.3 三个必须一起核对的文件

- `ActionsCfg`：策略输出如何缩放；
- `assets/uika.py`：默认姿态、关节顺序、执行器与限制；
- UIKA URDF：关节轴、上下限和几何结构。

只改 action scale 而不看关节限制，属于无依据调参。

## 5.4 实验：动作缩放与行为

在独立实验分支中，把非 hip scale 从 `0.25` 改为一个更保守的值。保持其他设置不变，比较：

- `action_rate_l2`；
- 关节限位惩罚；
- 速度跟踪；
- 回放时的步幅、抖动和通过能力。

如果短训练策略走不起来，只能说明“在当前预算下学习更慢或失败”，不能直接证明较小动作范围永远更差。

---

# 第 6 课：Command——告诉机器人做什么

## 6.1 command 与 observation 的关系

`CommandsCfg` 负责采样目标：

```python
lin_vel_x=(-1.0, 1.0)
lin_vel_y=(-1.0, 1.0)
ang_vel_z=(-1.0, 1.0)
```

采样结果通过 `velocity_commands` observation 告诉策略，又被速度跟踪 reward 用作目标。command 不直接控制关节，它定义任务。

## 6.2 训练范围与播放命令不是一回事

训练环境随机采样较宽范围，使策略学习多种目标；play 配置可固定一个速度，便于观察。不要为了让回放只向前走，就把整个训练范围改成单一前进速度。

## 6.3 实验：从简单指令开始

设计两组训练：

- baseline：当前 x/y/yaw 范围；
- simplified：缩小横移和旋转范围，保留前进后退。

比较早期线速度跟踪和最终泛化。你需要回答：simplified 更快学会前进，是否意味着它是更好的通用策略？

---

# 第 7 课：Reward——把“想要的行为”写成可计算反馈

## 7.1 reward 的三个层次

读取一个 reward term 时分开看：

```python
track_lin_vel_xy = RewTerm(
    func=mdp.track_lin_vel_xy_exp,
    weight=3.0,
    params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
)
```

1. `func`：原始物理量如何计算；
2. `params`：目标、阈值、body 或 sensor 是什么；
3. `weight`：这个目标在总 reward 中占多大方向和强度。

正权重通常鼓励，负权重通常惩罚，但必须先看函数返回值。不能只看到负号就下结论。

## 7.2 当前 reward 的行为分组

- 任务目标：线速度与角速度跟踪；
- 机体稳定：竖直速度、横滚俯仰角速度、高度、朝上；
- 关节约束：力矩、功率、加速度、位置限制、默认姿态；
- 动作平滑：相邻动作变化；
- 接触安全：非足端接触、过大足端力、打滑；
- 步态塑形：腾空时间、触地、足端高度、对角步态。

reward 是多目标折中。某个分项变好，可能以另一个目标变差为代价。

## 7.3 推荐的第一个 reward 实验

选择行为和指标都容易观察的 `action_rate_l2`：

1. baseline 使用当前权重；
2. treatment 只改变该权重；
3. 两组使用相同 seed、task、环境数、迭代数；
4. 比较动作变化率、速度跟踪、总 reward 和回放抖动；
5. 解释“更平滑”是否牺牲了响应速度。

不要把复杂足端高度项作为第一个练习。高度项可能涉及世界坐标/机体坐标、接触相位和速度权重，初学者很容易根据单一 episode reward 做错误反推。

## 7.4 一份合格的 reward 实验结论

不合格：

> treatment 的总 reward 更高，所以更好。

合格：

> 在相同 seed 和训练预算下，treatment 的动作变化率惩罚绝对值下降，回放抖动减小；但线速度跟踪误差增大。因此该权重改善了平滑性，却降低了响应能力。当前结果只覆盖平地和一个 seed，不能推广到复杂地形。

---

# 第 8 课：Termination、Event 与 Curriculum

## 8.1 三者分别解决什么

- termination：这一回合何时结束；
- event：何时随机化或施加扰动；
- curriculum：何时提高或降低任务难度。

它们都会改变采样分布，但含义不同。

## 8.2 EventCfg 的时间模式

本仓库包含：

- `startup`：环境启动时随机化摩擦、质量和质心；
- `reset`：每次回合重置关节、base、执行器参数；
- `interval`：训练中周期性施加力或推机器人。

阅读 event 时，先看 `mode`，再看采样范围，最后看作用对象 `asset_cfg`。

## 8.3 为什么要 domain randomization

真实机器人与仿真模型不会完全一致。HimLoco 论文将质量、摩擦、执行器、延迟和外力等随机化，用不同环境响应训练鲁棒策略。本仓库把这些因素放在 `EventCfg` 和执行器配置中。

随机范围不是越大越好。范围过小可能覆盖不了现实差异，范围过大可能让任务难以学习或产生不真实样本。

## 8.4 实验：随机化不是噪声开关

选择一项容易解释的随机化，例如 base mass：

- baseline：当前范围；
- narrow：缩小范围；
- evaluation：在训练范围内和范围外分别回放。

比较学习速度和鲁棒性。结论应区分“训练更快”与“面对模型误差更稳定”。

---

# 第 9 课：从 HimLoco 论文回到本仓库

## 9.1 论文的核心工程问题

腿式机器人部署时只能获得有限且有噪声的传感信息。HimLoco 不要求 policy 直接回归所有外部环境参数，而是从历史本体感知中提取机器人响应：

- 显式部分：3 维机体线速度估计；
- 隐式部分：16 维归一化 latent，用于表达稳定性和环境动态响应。

policy 使用“当前单帧 observation + 速度估计 + latent”输出动作。

## 9.2 论文—代码映射

| 论文内容 | 本仓库代码 |
|---|---|
| partial observation | `ObservationsCfg.PolicyCfg` |
| privileged value input | `ObservationsCfg.CriticCfg` |
| history observation | `HimlocoVecEnvWrapper.obs_history_buf` |
| hybrid internal embedding | `HIMEstimator.encoder` |
| explicit velocity loss | `F.mse_loss(pred_vel, vel)` |
| prototype contrastive objective | `proto`、`sinkhorn()`、`swap_loss` |
| actor input | `torch.cat((current_obs, vel, latent))` |
| HIO + PPO | `HIMPPO.update()` |

## 9.3 estimator 数据形状

当前实现中：

```text
历史 observation：6 × 45 = 270
encoder 输出：3 维 velocity + 16 维 latent
actor 输入：45 + 3 + 16 = 64
actor 输出：12 维 action mean
```

训练时 estimator 使用下一时刻 critic observation 中的真实 base linear velocity 作为监督目标，同时用 source/target encoder、prototype 和 Sinkhorn assignment 计算 swap loss。

## 9.4 HIO 与 PPO 如何交替

在每个 mini-batch 中：

1. 根据 observation 计算动作分布与 value；
2. 调用 `estimator.update(...)` 更新 HIM；
3. 计算 PPO clipped surrogate loss；
4. 计算 value loss 与 entropy bonus；
5. 更新 actor/critic。

这与论文的“HIO 与 PPO 交替优化”思想对应，但具体 rollout 长度和实现细节以本仓库为准。论文报告 4096 个环境、100 步 rollout；UIKA 当前配置是 4096 个环境、每环境 24 步。不要把论文超参数直接复制成课程默认值。

## 9.5 TensorBoard：先问问题，再选曲线

启动：

```bash
tensorboard --logdir logs/himloco_rsl_rl --port 6006
```

浏览器访问：

```text
http://localhost:6006
```

常见指标分四类看：

- 任务表现：速度跟踪、episode reward、episode length；
- 策略优化：surrogate、value function、KL、learning rate；
- HIM：estimation loss、swap loss；
- 工程性能：FPS、collection time、learning time。

不要要求所有 loss 单调下降。RL 的数据分布随策略变化，reward 也是多个目标的加权结果。异常判断应结合突变、长期趋势、NaN、策略回放和配置变化。

---

# 第 10 课：综合项目——有证据地改变一个行为

## 10.1 项目题目模板

从一个可观察问题开始，例如：

- 机器人平地行走时动作抖动明显；
- 机器人低速指令跟踪较差；
- 静止指令下仍频繁抬脚；
- 足端打滑较多；
- 复杂地形训练早期频繁失败。

不要用“把 reward 调好”作为题目，它没有可检验终点。

## 10.2 实验协议

### 第一步：定义指标

至少选择：

- 一个主要任务指标；
- 一个可能产生副作用的指标；
- 一个可回放观察的行为标准。

### 第二步：建立 baseline

记录：

```text
git commit / git diff
task
seed
num_envs
max_iterations
run_name
GPU
训练开始与结束时间
```

### 第三步：提出机制假设

示例：

> 我认为机器人抖动主要来自相邻 action 变化过大。提高 action-rate penalty 的绝对权重会降低抖动，但可能使速度响应变慢。

### 第四步：单变量修改

只修改能检验该假设的主要因素。格式化、注释等不影响运行的改动可以存在，但不能同时改变 command 范围、observation 和多个 reward。

### 第五步：运行 treatment

使用与 baseline 一致的训练预算和 seed。用明确名称：

```bash
--run_name action_rate_stronger_seed1
```

### 第六步：比较与复查

先确认两个 run 的 `params/env.yaml` 和 `params/agent.yaml`，证明预期改动真正生效，再看曲线和回放。

### 第七步：写结论

按以下结构：

1. 观察到了什么；
2. 哪些证据支持；
3. 是否支持原假设；
4. 有什么副作用；
5. 结论适用于哪些 task、seed 和训练预算；
6. 下一轮只准备改变什么。

## 10.3 实验报告模板

```markdown
# 实验标题

## 问题

## 机制假设与预测

## 基线配置

## 唯一主要改动

## 运行命令与环境

## 结果

### TensorBoard 证据

### 回放行为

## 结论

## 局限

## 下一步
```

## 10.4 评分标准

| 项目 | 合格标准 |
|---|---|
| 可复现 | 同学能根据记录找到配置并重复命令 |
| 变量控制 | baseline 与 treatment 的主要差异唯一且明确 |
| 证据 | 同时使用配置、曲线和行为，不只看总 reward |
| 解释 | 能沿代码说明改动如何进入 RL 闭环 |
| 边界意识 | 不把短训练、单 seed、单地形结论过度推广 |

---

# 附录 A：GPU 不统一时怎样选择 `num_envs`

从小到大试，不按显卡型号猜：

1. 先用 16 个环境完成冒烟测试；
2. 用 64 个环境进行 GUI 或代码调试；
3. 依次尝试 256、512、1024；
4. 记录显存、FPS 与迭代时间；
5. 只有资源足够时才使用默认 4096。

出现 OOM 后先结束残留 Isaac Sim 进程并降低环境数。环境数过少时，一轮 PPO 收集到的样本也更少，因此小环境数实验适合检查代码和方向，不一定适合得出最终性能结论。

`--headless` 关闭画面渲染，适合正式训练；需要看行为时使用 play，避免一边渲染大量环境一边训练。

---

# 附录 B：实验纪律清单

开始前：

- [ ] 我能用一句话描述本次问题；
- [ ] 我写下了改变后会发生什么；
- [ ] baseline 能正常复现；
- [ ] 当前 Git 状态已经记录；
- [ ] 我只计划改变一个主要因素。

运行后：

- [ ] 我保存了完整命令；
- [ ] 我检查了实际生成的 YAML；
- [ ] 我没有只看总 reward；
- [ ] 我看过 checkpoint 回放；
- [ ] 我区分了观察、推断和结论；
- [ ] 我写明了 seed、task、训练预算和局限。

---

# 附录 C：术语表

| 英文 | 本教程固定译法 | 简明含义 |
|---|---|---|
| environment | 环境 | 机器人与仿真世界组成的交互对象 |
| observation | 观测 | 策略做决定时获得的信息 |
| action | 动作 | 策略输出并交给控制器的量 |
| command | 指令 | 当前任务要求，如目标速度 |
| reward | 奖励 | 对当前行为的标量反馈 |
| termination | 终止 | 当前 episode 结束的条件 |
| episode | 回合 | 从一次 reset 到下一次结束 |
| rollout | 轨迹采集 | 用当前策略连续收集的一批交互数据 |
| actor / policy | 策略网络 | 根据观测产生动作分布 |
| critic / value network | 价值网络 | 估计当前状态的长期价值 |
| privileged information | 特权信息 | 训练时可用、部署时不直接给策略的信息 |
| proprioception | 本体感知 | IMU、关节位置、关节速度等自身传感信息 |
| domain randomization | 域随机化 | 随机改变仿真参数以提高鲁棒性 |
| curriculum | 课程学习 | 根据表现逐步改变任务难度 |
| checkpoint | 检查点 | 某次训练保存的模型与优化状态 |
| ablation | 消融实验 | 移除或屏蔽一个因素以检验其作用 |
| baseline | 基线组 | 用于比较的原始设置 |

---

# 参考资料

- Junfeng Long et al., [Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response](https://arxiv.org/abs/2312.11460), ICLR 2024.
- [HimLoco 项目主页](https://junfeng-long.github.io/HIMLoco/)
- [HimLoco 官方代码](https://github.com/InternRobotics/HIMLoco)
- [Isaac Sim 5.0 Python 环境安装](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/install_python.html)
- [Isaac Sim 5.0 系统要求](https://docs.isaacsim.omniverse.nvidia.com/5.0.0/installation/requirements.html)
- [Isaac Lab RL 调试与训练指南](https://isaac-sim.github.io/IsaacLab/main/source/overview/reinforcement-learning/training_guide.html)

阅读外部资料时先确认版本。网页的 `latest` 文档可能已经面向 Isaac Sim 6.x，不一定适用于本课程冻结的 5.0 工程线。

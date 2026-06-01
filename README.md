# UIKA Lab

UIKA Lab 是华东理工大学 Robocon 无贰战队参加仿生足式比赛的代码仓库。仓库基于 Isaac Lab 搭建 UIKA 四足机器人强化学习训练、策略导出、仿真验证与 Sim2Real 部署流程。

## 项目说明

本项目面向 UIKA 仿生足式机器人，主要包含：

- UIKA 机器人 URDF、网格资源与 Isaac Lab articulation 配置
- 基于 HimLoco 思路的强化学习网络结构与训练流程
- 参考 RobotLab 的速度跟踪、稳定性、能耗与动作平滑等奖励项设计
- 面向仿生足式比赛场景的平地、坡面、台阶、离散障碍等地形训练配置
- 策略导出、Sim2Sim 验证与基于 rl_sar 的 Sim2Real 部署流程
- 使用 PACE 参数辨识方法进行机器人模型与控制参数校准

## 技术路线

### 训练框架

训练环境使用 Isaac Lab，强化学习算法实现保留 HimLoco 的双网络结构思想，将运动策略与隐变量估计结合，用于提升机器人在模型误差、地形变化和接触扰动下的鲁棒性。

### 奖励设计

奖励函数参考 RobotLab 的腿式运动任务设计，围绕以下目标进行约束：

- 跟踪线速度与角速度指令
- 保持机体姿态、足端接触节律和运动稳定性
- 抑制关节力矩、关节速度、动作变化率等高能耗或高冲击行为
- 提升复杂地形下的通过能力与比赛任务适应性

### Sim2Real

Sim2Real 部署使用 rl_sar 作为策略运行和实机控制框架。部署前先在 MuJoCo 中完成 Sim2Sim 验证，再结合 PACE 参数辨识结果修正质量、质心、惯量、关节阻尼、PD 参数等关键参数，降低训练仿真与实机之间的差异。

## 仓库结构

```text
himloco_lab/
├── scripts/
│   ├── himloco_rsl_rl/
│   │   ├── train.py                 # 训练入口
│   │   ├── play.py                  # 推理、回放与策略导出
│   │   └── play_interactive.py      # 键盘交互控制
│   └── list_envs.py                 # 查看已注册环境
├── source/himloco_lab/himloco_lab/
│   ├── assets/
│   │   ├── uika.py                  # UIKA articulation 配置
│   │   └── uika/                    # UIKA URDF 与 mesh 资源
│   ├── rsl_rl/                      # HimLoco 风格 RL 算法实现
│   ├── terrains/                    # 自定义地形配置
│   └── tasks/locomotion/
│       ├── agents/                  # PPO / HimLoco runner 配置
│       ├── mdp/                     # 观测、命令、奖励、事件、终止项
│       └── robots/uika/             # UIKA 训练与播放环境
├── deploy/                          # 部署相关代码与第三方依赖
└── docs/                            # 项目文档
```

## 环境安装

### 1. 安装 Isaac Lab

请先按照 Isaac Lab 官方文档安装 Isaac Sim 与 Isaac Lab，并确认当前 Python 环境可以正常运行 Isaac Lab。

### 2. 克隆仓库

```bash
git clone https://github.com/SAIKi0125/UIKA_lab.git
cd UIKA_lab
```

### 3. 安装本项目

```bash
python -m pip install -e source/himloco_lab
```

### 4. 查看环境注册情况

```bash
python scripts/list_envs.py
```

确认列表中包含：

- `UIKA-Velocity`
- `UIKA-Velocity-Play`

## 快速开始

### 训练 UIKA 策略

```bash
python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity --headless
```

### 播放与导出策略

```bash
python scripts/himloco_rsl_rl/play.py --task UIKA-Velocity-Play
```

导出结果会保存在训练日志目录中，常见文件包括：

- `policy.pt`：TorchScript 策略文件
- `encoder.onnx` 与 `policy.onnx`：ONNX 格式模型

### 交互式控制

```bash
python scripts/himloco_rsl_rl/play_interactive.py --task UIKA-Velocity-Play
```

## UIKA 训练配置

UIKA 的主要训练配置位于：

- `source/himloco_lab/himloco_lab/assets/uika.py`
- `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py`
- `source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_rsl_rl_cfg.py`

地形配置默认提供平地训练模式，也保留了坡面、台阶、障碍等复杂地形配置，可在 `velocity_env_cfg.py` 中切换。

## 部署流程

推荐部署流程：

1. 在 Isaac Lab 中完成策略训练。
2. 导出 `policy.pt` 或 ONNX 模型。
3. 将策略和 UIKA 模型配置同步到 rl_sar。
4. 在 MuJoCo 中完成 Sim2Sim 验证。
5. 根据 PACE 参数辨识结果修正模型参数和控制参数。
6. 在低速、限幅、有人保护条件下进行实机 Sim2Real 测试。

实机部署前必须确认急停、限位、关节方向、关节顺序、默认站立角、PD 参数、动作缩放、观测归一化和历史帧顺序均与训练配置一致。

## 参考项目

- [HimLoco](https://github.com/RoboLoco/HimLoco)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
- [RobotLab](https://github.com/fan-ziqi/robot_lab)
- [rl_sar](https://github.com/fan-ziqi/rl_sar)
- PACE 参数辨识方法

# himloco_lab 项目开发指南

## 项目路径
/home/esd/project/himloco_lab

## Python 环境
- conda env: isaac_lab
- 路径: /home/esd/miniconda3/envs/isaac_lab/bin/python
- 注意: `~/.local/lib/python3.10/site-packages/torch` 是 CPU-only 版本，会遮蔽 conda 环境的 CUDA torch，导致训练报 `Torch not compiled with CUDA enabled`。跑训练/list_envs 必须加 `PYTHONNOUSERSITE=1` 屏蔽用户级 site-packages：
  ```bash
  PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity-Rough --headless
  ```

## 关键配置

### UNITREE_ROS_DIR
- 配置文件: source/himloco_lab/himloco_lab/assets/unitree.py
- 值: /tmp/unitree_ros (symlink -> /home/esd/project/HuangCang-project/source/robot_lab/data/Robots/unitree)
- 注意: 需要 merge_fixed_joints=False 保留 foot body

### UIKA 环境迁移 (已完成)
- URDF资源: source/himloco_lab/himloco_lab/assets/uika/ (从 /home/esd/project/UIKA_HIMloco/resources/robots/uika/ 复制)
- 机器人配置: source/himloco_lab/himloco_lab/assets/uika.py (UIKAArticulationCfg, DCMotorCfg, merge_fixed_joints=False)
- 环境配置: source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py
- Gym注册: UIKA-Velocity, UIKA-Velocity-Play
- Agent配置: UIKAPPORunnerCfg (num_steps_per_env=24, entropy_coef=0.005, lr=5e-4)

### UIKA 机器人参数
- base_height: 0.3357m
- 12个关节 (FL/FR/RL/RR x hip/thigh/calf)
- hip effort: 17N·m, vel: 28.80 rad/s
- calf effort: 31.7N·m, vel: 15.43 rad/s
- PD: stiffness=30, damping=1
- action_scale: 0.25
- 关节限位已验证，所有default_joint_pos在限位范围内

## 项目结构
- 核心代码: `source/himloco_lab/himloco_lab/`
- 机器人定义: `assets/` — `assets/uika.py` 定义 `UIKAArticulationCfg` 和 `DCMotorCfg`，`assets/uika/` 存放 URDF 和 mesh
- 运动控制逻辑: `tasks/locomotion/`
  - `mdp/`: observations, rewards, commands, events, terminations
  - `agents/`: PPO/HimLoco runner 配置
  - `robots/uika/velocity_env_cfg.py`: UIKA train/play 设置
- 训练脚本: `scripts/himloco_rsl_rl/`
- 部署代码: `deploy/`
- `logs/` 和 `outputs/` 视为生成产物

## 常用命令

### 安装与检查
```bash
python -m pip install -e source/himloco_lab
python scripts/list_envs.py  # 应显示 UIKA-Velocity 和 UIKA-Velocity-Play
```

### 训练
注意: 必须加 `PYTHONNOUSERSITE=1` 屏蔽用户级 CPU-only torch（见「Python 环境」）。
```bash
# Go2
PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless
# UIKA 平地
PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity --headless
# UIKA rough 地形
PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity-Rough --headless
# 冒烟测试（少量 env + 迭代）
PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity-Rough --headless --num_envs 64 --max_iterations 5
```

### Play
```bash
# 平地
PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/play.py --task UIKA-Velocity-Play --experiment_name uika
# rough 地形
PYTHONNOUSERSITE=1 python scripts/himloco_rsl_rl/play.py --task UIKA-Velocity-Rough-Play --experiment_name uika
# 指定 run: --load_run 2026-05-23_15-02-49
# 速度在 RobotPlayEnvCfg.__post_init__ 中 commands.base_velocity.ranges 设置
```

### TensorBoard
```bash
tensorboard --logdir /home/esd/project/himloco_lab/logs/himloco_rsl_rl --port 6006 --bind_all
```

## 日志路径
- /home/esd/project/himloco_lab/logs/himloco_rsl_rl/uika/
- /home/esd/project/himloco_lab/logs/himloco_rsl_rl/go2_rough/

## 编码风格
- Python 3.10+，遵循 Isaac Lab 风格: 4 空格缩进、带类型注解的 config 类、描述性 reward/observation 命名
- Gym task ID 形如 `UIKA-Velocity`、`UIKA-Velocity-Play`
- pre-commit: Black (line length 120)、isort (black profile)、flake8、pyupgrade、codespell、whitespace/YAML/TOML 检查
- 大范围改动前运行 `pre-commit run --all-files`（如可用）

## 测试
- 当前无专用测试套件，用聚焦的冒烟测试验证:
  - 安装包 → 运行 `scripts/list_envs.py` → 短时 headless 训练触及的 task → 对策略加载路径运行 `play.py`
- UIKA 资源或配置改动: 验证 12 个关节 (`FL/FR/RL/RR` x `hip/thigh/calf`)、default joint pos、限位、body 名称、foot body 存在

## 提交与 PR
- 使用简洁的 conventional 风格 subject，格式 `type: summary`（如 `feat: adapt to IsaacLab 2.3.2`、`fix: correct calculation in smoothness reward function`、`docs: update UIKA README setup`）
- PR 应描述涉及的 task 或机器人、列出验证命令、注明生成产物或日志，并在适用时关联 issue

## 注意事项
- URDF 中 foot 链接通过 fixed joint 连接，必须设置 merge_fixed_joints=False 才能保留 foot body
- /tmp/unitree_ros 是临时 symlink，重启后需要重建

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

### UIKA Takeoff / SpringJump 门控记忆
- takeoff 分支以 SpringJump 定点跳为基准；HuangCang takeoff 只作参考，不作为最终 reward 权重标准。
- takeoff 不修改 robot default joint pose；`robot = ROBOT_CFG.replace(...)`，action=0、reset dof、`before_setting`、`dof_pos`、`dof_hip_pos` 都以 UIKA 资产默认站姿为基准，和 SpringJump 使用 `default_joint_angles` 的结构一致。
- takeoff 不使用 adaptation 网络；当前 runner 是 `rsl_rl_takeoff_cfg:UIKATakeoffPPORunnerCfg`，普通 RSL-RL MLP actor-critic，actor 输入 `actor_history` 10 帧历史，critic 输入 `critic` 3 帧历史；仓库中不应再引用 `rsl_rl_takeoff_adapt_cfg`、`takeoff_adapt_model`、`privileged_target` 或 `adaptation_loss_coef`。
- 命令 `base_velocity` 是 1 维: `[jump_flag]`。不要再把目标落点作为 command 或 obs 传给 actor/critic。
- `jump_flag == 0.0`: 起跳前准备/下蹲阶段。
- `jump_flag == 1.0`: 起跳触发后阶段。
- `_sj_was_in_flight`: 曾经四足离地，条件是 `all_air & fresh & (jump_flag == 1.0)` 后保持为 True。
- `_sj_last_contact_xy` / `_sj_takeoff_xy`: 仅保留为内部状态或调试辅助；当前 critic 不接收 `landing_xy_from_start` 或 `landing_xy_from_takeoff`。
- `_sj_has_jumped`: 已经飞行后再次接触地面，条件是 `_sj_was_in_flight & any_contact & ~_sj_has_jumped & fresh`。
- `_sj_landing_xy`: 只在 `just_landed` 时记录当前 root xy。
- 随机向上辅助冲量只在 `jump_flag == 1.0 & ~_sj_push_applied & ~_sj_has_jumped & fresh` 候选时触发；训练按原生 SpringJump：范围 `(1.5, 2.2)`，概率 `max(8 - int(common_step_counter / 1200), 0) / 10`，并由 `push_initial_prob=0.8` 上限控制；Play 中 `push_initial_prob=0.0` 禁用推动。
- `before_setting`: `jump_flag == 0.0`，奖励接近 robot default joint pose，不单独传下蹲 target。
- `line_z`: `~_sj_has_jumped & jump_flag == 1.0`，奖励向上速度。
- `flight`: `_sj_was_in_flight`，曾离地后生效。
- `base_height_flight`: `_sj_was_in_flight & ~_sj_has_jumped`，飞行中高度奖励。
- `base_height_flight` / `base_height_stance_sj` 的 active reward 结构按 SpringJump，但高度数值保留 UIKA 物理尺寸：`target_height=0.50`、`stance_target=0.3357`、`setting_target=0.24`；不要为了逐字对齐 GO2 改成 GO2 高度。
- `land_pos`: `_sj_has_jumped & upright & (max_height > min_height)`；目标落点使用固定 reward 参数 `target_xy=(1.0, 0.0)`，计算为 `_sj_init_xy + target_xy`，不要从 command 读取 xy。
- critic 不传 landing position；除 command 从原生 3 维改为 1 维 `jumpflag` 外，critic 按原生 SpringJump 顺序对齐：
  `velocity_commands(1), joint_pos_rel(12), joint_pos_abs(12), joint_vel_rel(12), last_action(12), base_lin_vel(3), base_ang_vel(3), base_euler_xyz(3), contact_mask(4), has_jumped(1)`。
  单帧 63 维，`history_length=3` 后为 189 维。
- `tracking_lin_vel_jump`: `_sj_was_in_flight & ~_sj_has_jumped`，只在飞行未落地阶段按固定 `target_forward_velocity=1.6` 奖励，不再从 command x 读取目标速度。
- `line_vel_stance`: `_sj_has_jumped`，落地后惩罚水平速度用于站稳。
- `foot_clearance_jump`: `_sj_was_in_flight & ~_sj_has_jumped`，飞行未落地阶段惩罚脚部高度偏差。
- 当前 active reward 不启用 `line_vel_x_setting`、`line_vel_y_setting`、`dof_pos_penalty_prepare_sj`，因为它们不是原生 SpringJump reward 表。
- `dof_pos_penalty_sj`: 无 jump 阶段门控，全程惩罚偏离默认关节姿态，和 SpringJump 原版一致。
- `dof_hip_pos_penalty_sj`: 无 jump 阶段门控，全程惩罚 hip 偏离默认姿态，和 SpringJump 原版一致。
- `action_rate_l2_sj` 函数里门控为 `jump_flag == 1.0 & ~_sj_has_jumped` 时用 `jump_action_rate_scale` 缩放；当前 active takeoff reward 使用的是 SpringJump 基准 `action_rate_l2`，不是 `action_rate_l2_sj`。
- `lin_vel_z_stance`、`stand_still_crouch_setting`、`stand_still_setting` 函数保留为 HuangCang 参考函数；当前 active takeoff reward 表未启用。
- `successful_jump_sj` 函数保留但 active takeoff reward 表已删除，当前训练不使用一次性成功奖励。
- SpringJump 通用惩罚 `ang_vel_xy`、`torques`、`joint_pos_limits`、`dof_vel_limits`、`dof_vel`、`collision`、`action_rate_l2`、`feet_contact_forces` 没有 jump 阶段门控，按每步通用惩罚生效。
- termination 结构按 SpringJump：timeout、base contact、too low；当前 too low 阈值按用户要求对齐 SpringJump 为 `0.15`。

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

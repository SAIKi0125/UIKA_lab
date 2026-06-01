# himloco_lab 项目开发指南

## 项目路径
/home/esd/project/himloco_lab

## Python 环境
- conda env: isaac_lab
- 路径: /home/esd/miniconda3/envs/isaac_lab/bin/python

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

## 常用命令

### 训练
```bash
# Go2
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless
# UIKA
python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity --headless
```

### Play
```bash
python scripts/himloco_rsl_rl/play.py --task UIKA-Velocity-Play --experiment_name uika
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

## 注意事项
- URDF 中 foot 链接通过 fixed joint 连接，必须设置 merge_fixed_joints=False 才能保留 foot body
- /tmp/unitree_ros 是临时 symlink，重启后需要重建

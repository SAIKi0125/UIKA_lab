# Repository Guidelines

## Project Structure & Module Organization

This repository extends Isaac Lab for UIKA quadruped locomotion training, policy export, and Sim2Real deployment. Core Python code lives in `source/himloco_lab/himloco_lab/`. Robot definitions are in `assets/`: `assets/uika.py` defines `UIKAArticulationCfg` and `DCMotorCfg`, while `assets/uika/` contains UIKA URDF and meshes. Locomotion logic is under `tasks/locomotion/`: `mdp/` has observations, rewards, commands, events, and terminations; `agents/` contains PPO/HimLoco runner configs; `robots/uika/velocity_env_cfg.py` contains UIKA train/play settings. Training scripts are in `scripts/himloco_rsl_rl/`; deployment code is under `deploy/`. Treat `logs/` and `outputs/` as generated artifacts.

## Build, Test, and Development Commands

Use the `isaac_lab` conda environment, or `/home/esd/miniconda3/envs/isaac_lab/bin/python` directly.

```bash
python -m pip install -e source/himloco_lab
python scripts/list_envs.py
python scripts/himloco_rsl_rl/train.py --task Unitree-Go2-Velocity --headless
python scripts/himloco_rsl_rl/train.py --task UIKA-Velocity --headless
python scripts/himloco_rsl_rl/play.py --task UIKA-Velocity-Play --experiment_name uika
python scripts/himloco_rsl_rl/play.py --task UIKA-Velocity-Play --experiment_name uika --load_run 2026-05-23_15-02-49
tensorboard --logdir /home/esd/project/himloco_lab/logs/himloco_rsl_rl --port 6006 --bind_all
```

`list_envs.py` should show `UIKA-Velocity` and `UIKA-Velocity-Play`. Adjust play command speeds in `RobotPlayEnvCfg.__post_init__` via `commands.base_velocity.ranges`.

## Coding Style & Naming Conventions

Python targets 3.10+. Follow Isaac Lab style: 4-space indentation, typed config classes, descriptive reward/observation names, and Gym task IDs such as `UIKA-Velocity` and `UIKA-Velocity-Play`. Pre-commit uses Black line length 120, isort with the Black profile, flake8, pyupgrade, codespell, and whitespace/YAML/TOML checks. Run `pre-commit run --all-files` before broad changes when available.

## Testing Guidelines

There is no dedicated test suite in the current tree. Validate changes with focused smoke tests: install the package, run `scripts/list_envs.py`, start short headless training for the touched task, and run `play.py` for policy-loading paths. For UIKA asset or config changes, verify the 12 joints (`FL/FR/RL/RR` x `hip/thigh/calf`), default joint positions, limits, body names, and foot body presence.

## Commit & Pull Request Guidelines

Recent history uses concise conventional-style subjects, for example `feat: adapt to IsaacLab 2.3.2`, `fix: correct calculation in smoothness reward function`, and `docs: update UIKA README setup`. Use the same `type: summary` format. PRs should describe the task or robot affected, list validation commands, note generated artifacts or logs, and link related issues when applicable.

## Configuration Notes

`UNITREE_ROS_DIR` is configured in `source/himloco_lab/himloco_lab/assets/unitree.py` as `/tmp/unitree_ros`, a symlink to `/home/esd/project/HuangCang-project/source/robot_lab/data/Robots/unitree`; recreate it after reboot. Keep `merge_fixed_joints=False` for UIKA and Unitree URDF imports so fixed-joint foot links remain bodies. Current UIKA reference values are `base_height=0.3357`, hip effort `17 N*m`, calf effort `31.7 N*m`, PD `stiffness=30`, `damping=1`, and `action_scale=0.25`. UIKA runner settings are `num_steps_per_env=24`, `entropy_coef=0.005`, and `lr=5e-4`. Main logs are under `logs/himloco_rsl_rl/uika/` and `logs/himloco_rsl_rl/go2_rough/`.

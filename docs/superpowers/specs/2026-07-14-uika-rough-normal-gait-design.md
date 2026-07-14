# UIKA Rough Normal-Gait Reward Environment Design

## Goal

Add an isolated UIKA rough-terrain experiment that keeps the current normal standing robot, action, observation, command, event, terrain, curriculum, and PPO semantics while using the gait-producing reward recipe from `uika/master@c1af265` `LowerRewardsCfg`.

This is a normal walking task, not a crouched lower task. The normal base-height target is `0.33 m`, and every joint-reference reward uses the current UIKA asset default joint positions.

## Source Baseline

- GitHub remote: `git@github.com:SAIKi0125/UIKA_lab.git`
- Reference: `uika/master@c1af265`
- Reference file: `source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/lower_env_cfg.py`
- Target repository state: the current local `master`, preserving all uncommitted user changes.

The upstream `origin/master` does not contain UIKA and is not the reward source.

## Architecture

Create `rough_normal_gait_env_cfg.py` beside the existing UIKA environment configs. It will contain three classes:

1. `RoughNormalGaitRewardsCfg(RewardsCfg)` snapshots the lower reward overrides from the reference commit.
2. `RoughNormalGaitEnvCfg(RobotEnvCfg)` replaces only `rewards` for training.
3. `RoughNormalGaitPlayEnvCfg(RobotPlayEnvCfg)` replaces only `rewards` for playback.

Because both environment classes inherit the current normal UIKA configs, the new task keeps:

- `UIKA_CFG` normal initialization and default joint positions;
- the current joint-position action offset and scale;
- standard actor/critic observations and history layout;
- current commands, reset/randomization events, terminations, terrain curriculum, simulation timing, and control rate;
- the existing `COBBLESTONE_ROAD_CFG` rough terrain;
- the existing UIKA PPO/model hyperparameters.

No existing UIKA environment class is modified.

## Reward Semantics

`RoughNormalGaitRewardsCfg` follows the effective lower reward recipe. It copies the lower overrides, including disabled terms, weights, thresholds, command gates, sensor/body selectors, and standard inherited reward behavior.

The only posture adaptations are:

| Reward | Master lower value | New normal-gait value | Reason |
|---|---:|---:|---|
| `base_height_l2.target_height` | `0.25` | `0.33` | The new task uses normal standing height. |
| `stand_still.target_joint_pos` | lower crouch map | current `UIKA_CFG.init_state.joint_pos` | Stand-still reference must match the normal action offset. |
| `joint_pos_penalty.target_joint_pos` | lower crouch map | current `UIKA_CFG.init_state.joint_pos` | Moving/standing posture penalty must match the normal robot. |

Key copied lower overrides include:

- `flat_orientation_l2=-0.2`, `base_height_l2=-1.0`, and `body_lin_acc_l2=-1e-4`;
- `joint_pos_limits=-1.0`, `stand_still=-2.0`, and `joint_pos_penalty=-0.3`;
- `track_lin_vel_xy=1.0` and `track_ang_vel_z=0.5`;
- `feet_air_without_cmd=-2.0` and `single_foot_air_time=-2.0` with `threshold=0.25`;
- the same `None` or zero-weight settings for termination reward, joint velocity terms, joint mirror, contact-force penalty, air-time shaping, prolonged swing, foot height/lift, and gait synchronization terms.

The new config must not import or reference `UIKA_LOWER_JOINT_POS_TARGET`.

## Task and Log Isolation

Register two new task IDs:

- `UIKA-Rough-Normal-Gait`
- `UIKA-Rough-Normal-Gait-Play`

Add `UIKARoughNormalGaitPPORunnerCfg`, inheriting `UIKAPPORunnerCfg` with:

```python
experiment_name = "uika_rough_normal_gait"
```

This keeps checkpoints under `logs/himloco_rsl_rl/uika_rough_normal_gait` and prevents accidental resume/play selection from the existing `uika`, `uika_flat`, or `uika_lower` experiments.

## Data Flow

Training resolves `UIKA-Rough-Normal-Gait` to `RoughNormalGaitEnvCfg` and `UIKARoughNormalGaitPPORunnerCfg`. The environment produces the same normal UIKA observations and action mapping as `UIKA-Velocity`; only the reward manager receives the new reward configuration. Playback resolves the `-Play` task to the matching play config and loads checkpoints from the isolated experiment directory.

## Compatibility

The actor observation shape/order, history length, action shape/order/scale, and model architecture remain unchanged from normal `UIKA-Velocity`. A normal UIKA checkpoint is dimensionally compatible, but the new experiment is intended to train from scratch because its reward objective differs. Lower-task checkpoints are not treated as compatible because lower changes action offsets and joint-position observation references.

## Testing

Implementation follows RED-GREEN-REFACTOR:

1. Add a focused static/config test that initially fails because the new file, task registrations, and runner config do not exist.
2. Verify registration names and train/play entry points.
3. Verify the training and play configs inherit the normal rough environment classes and replace only rewards.
4. Verify `0.33 m` base height and normal `UIKA_CFG.init_state.joint_pos` references.
5. Verify all copied lower override weights, parameters, and disabled terms.
6. Verify the new source never references `UIKA_LOWER_JOINT_POS_TARGET`.
7. Verify the isolated PPO experiment name.
8. Run Python compilation and the relevant UIKA test suite.
9. If the Isaac Sim runtime is available, resolve the registered config and perform a short task-construction smoke test.

## Non-goals

- Do not alter existing `UIKA-Velocity`, `UIKA-Flat-Velocity`, `UIKA-Lower-Velocity`, or parkour tasks.
- Do not change the UIKA asset default pose, action scale, observation layout, terrain generator, events, commands, PPO hyperparameters, or reward function implementations.
- Do not merge/rebase the dirty worktree or copy the old local `lower` branch implementation wholesale.
- Do not claim gait quality before training curves and playback are evaluated.

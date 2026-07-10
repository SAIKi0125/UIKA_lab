# UIKA Lower Velocity Design

## Goal

Add isolated `UIKA-Lower-Velocity` training and Play tasks without changing the existing UIKA Velocity or Parkour tasks. The Lower task migrates the stabilized crouch locomotion semantics from commit `81fea26` on the `takeoff` branch while using the current master branch's PACE actuator model.

## Posture And Commands

The Lower task uses a 0.25 m initial base height and a 0.25 m base-height reward target. Its joint-space reference is:

| Joint type | Left target | Right target |
| --- | ---: | ---: |
| hip | -0.70 | 0.70 |
| thigh | 0.30 | 0.30 |
| calf | 0.20 | 0.20 |

The velocity command is resampled every 10 seconds with `vx` in `[-1.0, 1.0]` m/s, `vy` in `[-0.5, 0.5]` m/s, and yaw rate in `[-0.5, 0.5]` rad/s. Heading command is disabled and two percent of environments receive standing commands.

## Architecture

Create `lower_env_cfg.py` beside the existing UIKA velocity configuration. It reuses common scene sensors, events, curriculum, simulation settings, and manager terms from `velocity_env_cfg.py`, but supplies Lower-specific scene, commands, actions, observations, rewards, terminations, and Play configuration.

The Lower scene replaces only the robot initial state, so other UIKA tasks keep their 0.3357 m nominal base height. The Lower action uses the crouch target as an explicit joint-position offset. Actor and critic joint-position observations are measured relative to the same target. Stand-still and joint-position rewards also use that target.

The Lower reset event preserves the current domain-randomization terms, including all PACE events, but resets root pose and velocity without the large orientation perturbations used by the current Velocity configuration.

## Actuator Semantics

The Lower scene reuses the current `UIKA_CFG` actuator dictionary unchanged:

- 12 single-joint `DelayedDCMotor` instances;
- independent per-joint delay in the two-to-three physics-step range;
- independent per-joint PACE armature, static/dynamic friction, and viscous friction samples;
- nominal damping gain of 1.5;
- no encoder bias.

The Play task disables the PACE physical reset events and fixes delay to two physics steps through the same helper used by the current Play task.

## Compatibility

The Lower actor observation dimension, critic observation dimension, action dimension, and joint order match `UIKA-Velocity`. Existing checkpoints are tensor-shape compatible, although the action and joint-position-observation reference changes to the crouch posture, so a normal Velocity checkpoint is not behaviorally equivalent.

## Validation

Tests will verify task registration, exact posture and command ranges, target-relative observations/rewards, stable reset, PACE actuator reuse, and unchanged observation/action dimensions. An Isaac Sim 5.1 four-environment smoke test will instantiate both training and Play tasks and inspect the resolved scene, actuator, command, manager, and tensor shapes.

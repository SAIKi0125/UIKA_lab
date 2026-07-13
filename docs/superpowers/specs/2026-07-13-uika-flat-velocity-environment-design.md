# UIKA Flat Velocity Environment Design

## Goal

Add separately registered flat-ground train and play environments for the normal-height UIKA velocity policy. The existing rough-terrain, lower-body, and parkour environments must remain unchanged.

## Task Registration

Register two new Gymnasium task IDs:

- `UIKA-Flat-Velocity` for training.
- `UIKA-Flat-Velocity-Play` for policy evaluation.

Both tasks use `HimlocoManagerBasedRLEnv`. Training uses the standard RSL-RL entry point already used by `UIKA-Velocity`.

## Environment Configuration

Create `flat_env_cfg.py` in the UIKA robot task package. The flat configurations inherit the existing normal-height velocity configurations so observations, actions, rewards, events, terminations, simulation settings, and robot initialization stay aligned with `UIKA-Velocity`.

The flat scene overrides only the terrain:

- `terrain_type="plane"`
- `terrain_generator=None`
- Same collision group and rigid-body material settings as the existing UIKA environments.

Because there is no terrain generator, the inherited post-initialization behavior disables terrain-level curriculum and terrain-out-of-bounds termination.

## Commands

The training task inherits the full omnidirectional command distribution:

- `lin_vel_x=(-1.0, 1.0)` m/s
- `lin_vel_y=(-1.0, 1.0)` m/s
- `ang_vel_z=(-1.0, 1.0)` rad/s
- Command resampling every 10 seconds.
- Two percent standing environments.
- Heading commands disabled.
- Linear and angular command curricula disabled.

The play task inherits the current normal-height play command override:

- `lin_vel_x=(1.0, 1.0)` m/s
- `lin_vel_y=(0.0, 0.0)` m/s
- `ang_vel_z=(0.0, 0.0)` rad/s

## Rewards and Training Behavior

The flat task does not define or modify reward terms. It inherits the complete current `RewardsCfg` from `UIKA-Velocity`, including both body-frame foot-height terms. This isolates the terrain change and allows direct comparison between rough- and flat-ground training.

Robot initial pose, actuator configuration, action scale, observations, domain randomization, and command thresholding also remain unchanged.

## Runner Isolation

Add `UIKAFlatPPORunnerCfg` as a subclass of `UIKAPPORunnerCfg` with:

- `experiment_name="uika_flat"`

All other PPO and HIM settings remain inherited. Separate experiment names prevent flat checkpoints and logs from mixing with the existing `uika` rough-terrain runs.

## Verification

Add focused configuration tests that verify:

- Both new task IDs are registered.
- Train and play scenes use a plane with no terrain generator.
- The train task preserves the full omnidirectional command ranges.
- The play task fixes the command to 1 m/s forward.
- The flat task inherits the normal-height reward configuration without overriding reward terms.
- The flat runner uses the `uika_flat` experiment name.

Run the focused tests and Python compilation checks for changed modules. No simulator rollout is required for configuration-level verification.


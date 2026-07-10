# UIKA velocity curriculum configuration design

## Goal

Adjust the standard `UIKA-Velocity` training configuration so velocity commands use their full target range from the start, while terrain difficulty still advances through curriculum learning from level 0.

## Changes

- Disable both linear-velocity and angular-velocity command curriculum terms.
- Set the base command `lin_vel_x` range to `(-1.0, 1.0)` m/s.
- Keep terrain curriculum learning enabled.
- Set `RobotSceneCfg.terrain.max_init_terrain_level` to `0`.

## Scope

Only the regular UIKA velocity training configuration in `velocity_env_cfg.py` is changed. Parkour, play, reward terms, command ranges other than `lin_vel_x`, and terrain composition remain unchanged.

## Validation

- Compile `velocity_env_cfg.py` with Python to catch syntax errors.
- Inspect the relevant configuration values after editing.
- Confirm the diff contains only the intended training configuration changes.

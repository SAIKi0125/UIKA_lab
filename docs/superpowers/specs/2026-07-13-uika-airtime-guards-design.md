# UIKA Air-Time Guards Design

## Goal

Keep the existing positive foot air-time and air-time variance terms while preventing two failure modes:

- A moving robot holding one foot airborne for too long.
- A stationary robot lifting one or more feet instead of standing on all four feet.

Apply the new behavior to the normal-height Velocity, Flat, and Parkour tasks. Preserve the Lower task's existing stationary-air-foot penalty and do not stack a second moving-air-time penalty on Lower.

## Moving Prolonged-Swing Penalty

Port the `prolonged_swing` reward from the `takeoff` branch into the current shared reward module. Read each selected foot's continuously accumulating `current_air_time` and calculate:

```python
excess = clamp(current_air_time - 0.60, min=0.0)
penalty = sum(excess ** 2)
```

The term is active only when the planar velocity command norm is greater than `0.1`. Multiply it by the existing `_upright_gate(env)` because the current Velocity environment does not have the `base_contact` termination used by the takeoff rough environment. This prevents fallen robots from accumulating unbounded air-time penalties while preserving the penalty for upright tripod behavior.

Register the term in the shared normal-height `RewardsCfg` with:

- `weight=-1.5`
- `max_swing_time=0.60`
- Foot contact-sensor body selection.
- `command_name="base_velocity"`

Velocity, Flat, and Parkour inherit this term. Set `prolonged_swing=None` in `LowerRewardsCfg`.

## Stationary Air-Foot Penalty

Reuse the existing `feet_air_without_cmd` reward. It counts selected feet whose recent contact force remains below the contact threshold and applies the count only when the complete velocity-command norm is below the command threshold. It also uses `_upright_gate(env)`.

Register the term in shared `RewardsCfg` with:

- `weight=-2.0`
- `command_name="base_velocity"`
- `command_threshold=0.1`
- `contact_threshold=1.0`
- Foot contact-sensor body selection.

The existing `LowerRewardsCfg` definition already supplies equivalent behavior and remains unchanged.

## Existing Reward Terms

Do not change:

- `feet_air_time`: weight `0.1`, threshold `0.5` seconds.
- `feet_air_time_variance`: weight `-1.0`.
- `feet_contact_without_cmd`: weight `0.1`.
- Body-frame foot-height terms or gait terms.

The intended moving swing shaping is therefore:

- Below `0.5` seconds: a negative completed-air-time contribution at touchdown.
- From `0.5` to `0.60` seconds: positive completed-air-time reward without prolonged-swing penalty.
- Above `0.60` seconds: positive completed-air-time reward at touchdown competes with an increasing squared penalty while the foot remains airborne.

## Verification

Add focused tests that verify:

- `prolonged_swing` uses `current_air_time`, a squared excess above the configured threshold, planar-command gating, and `_upright_gate`.
- Shared `RewardsCfg` registers `prolonged_swing` with weight `-1.5` and threshold `0.60`.
- Shared `RewardsCfg` registers `feet_air_without_cmd` with weight `-2.0` and explicit thresholds.
- Existing `feet_air_time` and variance configuration remains unchanged.
- `LowerRewardsCfg` disables `prolonged_swing` while retaining its stationary-air-foot term.

Run focused reward tests, Python compilation, and diff checks. A simulator rollout is not required for this reward-configuration change.

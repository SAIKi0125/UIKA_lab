# UIKA PACE Actuator Randomization Design

## Goal

Add PACE-derived actuator domain randomization to the UIKA velocity task without changing the existing action-delay behavior. The parameter source is:

`/home/esd/project/dr002-sim/Walking_Eagle-Play/logs/pace/uika_sim/uika_200hz_full_pace_report_20260705.md`

The implementation randomizes armature, viscous damping, and joint friction. Encoder bias is intentionally excluded because the policy observation/default-position convention and the real encoder zero-point convention are not being migrated together.

## Grouping And Sampling

The 12 joints are divided into three actuator groups:

- `hip`: `FL_hip_joint`, `FR_hip_joint`, `RL_hip_joint`, `RR_hip_joint`
- `thigh`: `FL_thigh_joint`, `FR_thigh_joint`, `RL_thigh_joint`, `RR_thigh_joint`
- `calf`: `FL_calf_joint`, `FR_calf_joint`, `RL_calf_joint`, `RR_calf_joint`

At each environment reset, each group samples one armature, one viscous damping, and one friction value from uniform distributions. The four joints in that group share the sampled values. Different groups and different environments sample independently.

The ranges are the minimum and maximum identified values within each group:

| Group | Armature | Viscous damping | Friction |
| --- | --- | --- | --- |
| hip | `[0.012499551, 0.014251016]` | `[0.000871807, 0.002837807]` | `[0.005748361, 0.018038273]` |
| thigh | `[0.012442659, 0.014217399]` | `[0.001674354, 0.003362268]` | `[0.009873092, 0.022234440]` |
| calf | `[0.022163186, 0.024222745]` | `[0.001198500, 0.002290815]` | `[0.007793665, 0.015262470]` |

## Architecture

Create a focused reset event in the UIKA locomotion MDP layer. It will:

1. Resolve the articulation joints belonging to each named actuator group.
2. Sample tensors shaped `(num_reset_envs, 1)` for the three parameters.
3. Broadcast each sample across the four group joints.
4. Write armature, static joint friction, and viscous joint friction through the Isaac Lab articulation runtime APIs.

The UIKA asset configuration will expose separate `hip`, `thigh`, and `calf` actuator groups so group ownership and parameter mapping are explicit. Nominal asset values will remain valid deterministic defaults; reset randomization will overwrite only the selected environments.

The existing delay path is unchanged. No additional action buffer or torque buffer will be introduced, preventing double application of latency.

## Validation

Static tests will verify:

- the report-derived ranges and joint groups;
- the event is registered in reset mode;
- one sampled scalar is broadcast to all four joints in a group;
- different groups receive separate samples;
- encoder bias is absent;
- action-delay configuration is unchanged.

An Isaac Sim 5 smoke test will instantiate a small UIKA environment, reset selected environments, and inspect articulation buffers/runtime values when the environment is available.

## Compatibility

The actor and critic observation shapes, action ordering, reward configuration, terrain configuration, and checkpoint tensor shapes remain unchanged. Existing policies remain load-compatible, although rollout dynamics become randomized during training.

# UIKA PACE Actuator Randomization Design

## Goal

Add a complete PACE-derived actuator model to the UIKA velocity task. The model uses Isaac Lab's command-delay semantics while retaining the UIKA DC motor torque-speed limits, and randomizes the identified physical parameters. The parameter source is:

`/home/esd/project/dr002-sim/Walking_Eagle-Play/logs/pace/uika_sim/uika_200hz_full_pace_report_20260705.md`

The implementation randomizes command delay, armature, viscous damping, and joint friction. Encoder bias is intentionally excluded because the policy observation/default-position convention and the real encoder zero-point convention are not being migrated together. The nominal damping gain is 1.5, and the existing actuator-gain randomization is retained.

## Grouping And Sampling

The 12 joints use three parameter-range families:

- `hip`: `FL_hip_joint`, `FR_hip_joint`, `RL_hip_joint`, `RR_hip_joint`
- `thigh`: `FL_thigh_joint`, `FR_thigh_joint`, `RL_thigh_joint`, `RR_thigh_joint`
- `calf`: `FL_calf_joint`, `FR_calf_joint`, `RL_calf_joint`, `RR_calf_joint`

At each environment reset, every joint independently samples delay, armature, viscous damping, and friction. Joints of the same type use the same range but do not share samples. Different joints and different environments sample independently. Independent sampling does not require all sampled values to be unique; in particular, the discrete delay has only the values two and three.

The ranges are the minimum and maximum identified values within each group:

| Group | Armature | Viscous damping | Friction |
| --- | --- | --- | --- |
| hip | `[0.012499551, 0.014251016]` | `[0.000871807, 0.002837807]` | `[0.005748361, 0.018038273]` |
| thigh | `[0.012442659, 0.014217399]` | `[0.001674354, 0.003362268]` | `[0.009873092, 0.022234440]` |
| calf | `[0.022163186, 0.024222745]` | `[0.001198500, 0.002290815]` | `[0.007793665, 0.015262470]` |

PACE identified a 10-14 ms delay. The UIKA task runs physics at `sim.dt = 0.005`, so the delay range is converted by physical time to 2-3 physics steps. The old 5-7 step configuration from another branch would represent 25-35 ms in this task and is not reused.

## Architecture

Create a focused reset event in the UIKA locomotion MDP layer. It will:

1. Resolve the articulation joints belonging to each parameter-range family.
2. Sample tensors shaped `(num_reset_envs, num_group_joints)` for the three parameters.
3. Keep every joint column independently sampled without broadcasting.
4. Write armature, equal-valued static/dynamic joint friction, and viscous joint friction through the Isaac Lab articulation runtime APIs.

The UIKA asset configuration will expose 12 single-joint actuators. Each actuator owns its own delay buffers, so FL/FR/RL/RR joints sample delay independently while still using the hip/thigh/calf motor limits. Nominal asset values remain valid deterministic defaults; reset randomization overwrites only the selected environments.

Add a `DelayedDCMotor` that follows Isaac Lab's official `DelayedPDActuator` data flow: delay the position, velocity, and effort commands first, then call `DCMotor.compute()`. This preserves the DC motor torque-speed saturation and keeps `computed_effort` and `applied_effort` consistent with the torque sent to PhysX. No torque-after-PD buffer is introduced.

The training configuration uses a random 2-3 step delay. Play configurations disable the PACE physical-parameter reset events and use a fixed two-step delay. Existing non-PACE event behavior, including actuator-gain randomization, is unchanged. The actor and critic interfaces do not change.

## Validation

Static tests will verify:

- the report-derived ranges and joint groups;
- the event is registered in reset mode;
- each of the four joints in a range family receives an independently sampled physical parameter value;
- all 12 single-joint actuators own separate delay buffers and independently sample delay;
- encoder bias is absent;
- delay is applied to commands before PD and DC motor saturation;
- delay is 2-3 physics steps during training and fixed at two steps in Play;
- subset resets only change the selected environments.

An Isaac Sim 5 smoke test will instantiate a small UIKA environment, reset selected environments, and inspect articulation buffers/runtime values when the environment is available.

## Compatibility

The actor and critic observation shapes, action ordering, reward configuration, terrain configuration, and checkpoint tensor shapes remain unchanged. The nominal damping gain is explicitly fixed at the requested value of 1.5. Existing policies remain load-compatible, although rollout dynamics become randomized during training.

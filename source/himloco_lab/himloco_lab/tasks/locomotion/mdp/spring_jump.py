from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _init_sj_state(env: ManagerBasedRLEnv) -> None:
    if hasattr(env, "_sj_was_in_flight"):
        return
    env._sj_was_in_flight = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._sj_has_jumped = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._sj_landing_xy = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device)
    env._sj_init_xy = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device)
    env._sj_last_contact_xy = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device)
    env._sj_takeoff_xy = torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device)
    env._sj_takeoff_xy_recorded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._sj_max_height = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    env._sj_last_step_idx = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    env._sj_push_applied = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._sj_successful_jump_rewarded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _jump_flag(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)[:, 0]


def spring_jump_update(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    push_vel_z_range: tuple[float, float] = (1.5, 2.2),
    push_initial_prob: float = 0.8,
    push_decay_steps: int = 1200,
) -> None:
    """Update My_unitree-style jump state and optional upward training impulse."""
    _init_sj_state(env)
    cur_step = env.episode_length_buf
    fresh = cur_step != env._sj_last_step_idx
    if not fresh.any():
        return

    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, 2]
    foot_in_contact = torch.max(torch.abs(forces_z), dim=1)[0] > float(contact_threshold)
    all_air = ~foot_in_contact.any(dim=1)
    any_contact = foot_in_contact.any(dim=1)

    jump_flag = _jump_flag(env, command_name)
    jump_active = jump_flag == 1.0

    push_candidates = jump_active & (~env._sj_push_applied) & (~env._sj_has_jumped) & fresh
    if push_candidates.any() and push_initial_prob > 0.0:
        step_count = int(getattr(env, "common_step_counter", 0))
        current_prob = max(8 - int(step_count / float(push_decay_steps)), 0) / 10.0
        current_prob = min(current_prob, float(push_initial_prob))
        if current_prob > 0.0:
            rand_mask = torch.rand(env.num_envs, device=env.device) < current_prob
            push_ids = torch.where(push_candidates & rand_mask)[0]
            if push_ids.numel() > 0:
                root_vel = torch.cat(
                    [asset.data.root_lin_vel_w[push_ids], asset.data.root_ang_vel_w[push_ids]], dim=-1
                ).clone()
                vel_lo, vel_hi = float(push_vel_z_range[0]), float(push_vel_z_range[1])
                root_vel[:, 2] += torch.empty(push_ids.numel(), device=env.device).uniform_(vel_lo, vel_hi)
                asset.write_root_velocity_to_sim(root_vel, env_ids=push_ids)
        env._sj_push_applied[push_candidates] = True

    contact_before_takeoff = any_contact & fresh & (~env._sj_takeoff_xy_recorded)
    env._sj_last_contact_xy[contact_before_takeoff] = asset.data.root_pos_w[contact_before_takeoff, :2]

    first_takeoff = all_air & fresh & jump_active & (~env._sj_takeoff_xy_recorded)
    env._sj_takeoff_xy[first_takeoff] = env._sj_last_contact_xy[first_takeoff]
    env._sj_takeoff_xy_recorded[first_takeoff] = True
    env._sj_was_in_flight = env._sj_was_in_flight | (all_air & fresh & jump_active)

    just_landed = env._sj_was_in_flight & any_contact & ~env._sj_has_jumped & fresh
    env._sj_landing_xy[just_landed] = asset.data.root_pos_w[just_landed, :2]
    env._sj_has_jumped = env._sj_has_jumped | just_landed

    base_z = asset.data.root_pos_w[:, 2]
    env._sj_max_height = torch.where(fresh & (base_z > env._sj_max_height), base_z, env._sj_max_height)
    env._sj_last_step_idx = torch.where(fresh, cur_step, env._sj_last_step_idx)


def spring_jump_update_event(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    push_vel_z_range: tuple[float, float] = (1.5, 2.2),
    push_initial_prob: float = 0.8,
    push_decay_steps: int = 1200,
) -> None:
    """Event-manager adapter for spring-jump state updates."""
    del env_ids
    spring_jump_update(
        env,
        command_name=command_name,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        contact_threshold=contact_threshold,
        push_vel_z_range=push_vel_z_range,
        push_initial_prob=push_initial_prob,
        push_decay_steps=push_decay_steps,
    )


def reset_to_joint_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg,
    joint_angles: dict[str, float],
) -> None:
    """Reset selected joints to a named pose, leaving unspecified joints at default reset values."""
    asset: Articulation = env.scene[asset_cfg.name]
    if isinstance(env_ids, list):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    name_to_idx = {name: idx for idx, name in enumerate(asset.joint_names)}
    target_idx: list[int] = []
    target_val: list[float] = []
    for name, value in joint_angles.items():
        if name in name_to_idx:
            target_idx.append(name_to_idx[name])
            target_val.append(float(value))
    if not target_idx:
        return

    idx = torch.tensor(target_idx, device=asset.device, dtype=torch.long)
    val = torch.tensor(target_val, device=asset.device, dtype=torch.float)
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(asset.data.default_joint_vel[env_ids])
    joint_pos[:, idx] = val.unsqueeze(0)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def spring_jump_state_reset(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Clear spring-jump episode state and snapshot the episode initial xy."""
    _init_sj_state(env)
    asset: RigidObject = env.scene[asset_cfg.name]
    if isinstance(env_ids, list):
        env_ids = torch.tensor(env_ids, device=asset.device, dtype=torch.long)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=asset.device, dtype=torch.long)
    if len(env_ids) == 0:
        return

    env._sj_was_in_flight[env_ids] = False
    env._sj_has_jumped[env_ids] = False
    env._sj_landing_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]
    env._sj_init_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]
    env._sj_last_contact_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]
    env._sj_takeoff_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]
    env._sj_takeoff_xy_recorded[env_ids] = False
    env._sj_max_height[env_ids] = asset.data.root_pos_w[env_ids, 2]
    env._sj_last_step_idx[env_ids] = -1
    env._sj_push_applied[env_ids] = False
    env._sj_successful_jump_rewarded[env_ids] = False


def is_too_low(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < float(threshold)


def landing_xy_from_start(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the landing xy offset from episode start, using current xy before landing."""
    _init_sj_state(env)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2]
    landing_xy = torch.where(env._sj_has_jumped.unsqueeze(1), env._sj_landing_xy, current_xy)
    return landing_xy - env._sj_init_xy


def landing_xy_from_takeoff(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the landing/current xy offset from the actual takeoff position."""
    _init_sj_state(env)
    asset: RigidObject = env.scene[asset_cfg.name]
    current_xy = asset.data.root_pos_w[:, :2]
    landing_xy = torch.where(env._sj_has_jumped.unsqueeze(1), env._sj_landing_xy, current_xy)
    return landing_xy - env._sj_takeoff_xy


def has_jumped_obs(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    return env._sj_has_jumped.float().unsqueeze(1)


def line_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_flag = _jump_flag(env, command_name)
    return torch.clamp(asset.data.root_lin_vel_w[:, 2], min=0.0) * (~env._sj_has_jumped).float() * (
        jump_flag == 1.0
    ).float()


def before_setting(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
    leg_asset_cfg: SceneEntityCfg | None = None,
    target_joint_angles: dict[str, float] | None = None,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    if target_joint_angles is not None:
        if not hasattr(env, "_sj_before_setting_target"):
            target = asset.data.default_joint_pos[0].clone()
            name_to_idx = {name: idx for idx, name in enumerate(asset.joint_names)}
            for name, value in target_joint_angles.items():
                if name in name_to_idx:
                    target[name_to_idx[name]] = float(value)
            env._sj_before_setting_target = target
        target_pos = env._sj_before_setting_target.unsqueeze(0)
    else:
        target_pos = asset.data.default_joint_pos

    ids = leg_asset_cfg.joint_ids if leg_asset_cfg is not None else slice(None)
    dev = torch.sum(torch.abs(asset.data.joint_pos[:, ids] - target_pos[:, ids]), dim=1)
    jump_flag = _jump_flag(env, command_name)
    return torch.exp(-dev / 2.0) * (jump_flag == 0.0).float()


def flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    return env._sj_was_in_flight.float()


def base_height_flight(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_height: float = 0.47,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    rew = torch.exp(-torch.abs(asset.data.root_pos_w[:, 2] - float(target_height)) * 5.0)
    return rew * env._sj_was_in_flight.float() * (~env._sj_has_jumped).float() * 6.0


def base_height_stance_sj(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    stance_target: float = 0.30,
    setting_target: float = 0.25,
    setting_height_coef: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_flag = _jump_flag(env, command_name)
    h = asset.data.root_pos_w[:, 2]
    rew = torch.abs(h - float(stance_target)) * env._sj_has_jumped.float()
    rew += float(setting_height_coef) * torch.abs(h - float(setting_target)) * (jump_flag == 0.0).float()
    return rew


def land_pos(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    min_height: float = 0.42,
    target_xy: tuple[float, float] = (1.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_offset = torch.tensor(target_xy, dtype=env._sj_init_xy.dtype, device=env.device).unsqueeze(0)
    target_xy = env._sj_init_xy + target_offset
    land_err = torch.sum(torch.abs(target_xy - env._sj_landing_xy), dim=1)
    upright = torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1) < 0.6
    height_ok = env._sj_max_height > float(min_height)
    return torch.exp(-land_err) * env._sj_has_jumped.float() * upright.float() * height_ok.float()


def successful_jump_sj(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    success_height: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    upright = torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1) < 0.6
    first_success = (
        env._sj_has_jumped
        & (env._sj_max_height >= float(success_height))
        & upright
        & (~env._sj_successful_jump_rewarded)
    )
    env._sj_successful_jump_rewarded[first_success] = True
    return first_success.float()


def tracking_lin_vel_jump(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    vel_scale: float = 1.6,
    target_forward_velocity: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    target_velocity = float(vel_scale) if target_forward_velocity is None else float(target_forward_velocity)
    err = torch.square(target_velocity - asset.data.root_lin_vel_b[:, 0])
    return torch.exp(-err) * env._sj_was_in_flight.float() * (~env._sj_has_jumped).float() * 5.0


def line_vel_x_setting(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_flag = _jump_flag(env, command_name)
    return torch.square(asset.data.root_lin_vel_b[:, 0]) * (jump_flag == 0.0).float()


def line_vel_y_setting(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    jump_flag = _jump_flag(env, command_name)
    return torch.square(asset.data.root_lin_vel_b[:, 1]) * (jump_flag == 0.0).float()


def line_vel_stance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.root_lin_vel_b[:, :2]), dim=1) * env._sj_has_jumped.float()


def foot_clearance_jump(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_z_body: float = -0.20,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_rel = foot_pos_w - asset.data.root_pos_w[:, :3].unsqueeze(1)
    foot_body = torch.zeros_like(foot_rel)
    for foot_idx in range(foot_rel.shape[1]):
        foot_body[:, foot_idx, :] = quat_apply_inverse(asset.data.root_quat_w, foot_rel[:, foot_idx, :])
    height_error = torch.abs(foot_body[:, :, 2] - float(target_z_body))
    return torch.sum(height_error, dim=1) * env._sj_was_in_flight.float() * (~env._sj_has_jumped).float() * 6.0


def dof_pos_penalty_prepare_sj(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    target_joint_angles: dict[str, float] | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    if target_joint_angles is not None and not hasattr(env, "_sj_dof_pos_prepare_target"):
        target = asset.data.default_joint_pos[0].clone()
        name_to_idx = {name: idx for idx, name in enumerate(asset.joint_names)}
        for name, value in target_joint_angles.items():
            if name in name_to_idx:
                target[name_to_idx[name]] = float(value)
        env._sj_dof_pos_prepare_target = target
    jump_flag = _jump_flag(env, command_name)
    if target_joint_angles is None:
        target_pos = asset.data.default_joint_pos
    else:
        target_pos = env._sj_dof_pos_prepare_target.unsqueeze(0)
    err = torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pos[:, asset_cfg.joint_ids]).sum(dim=1)
    return err * (jump_flag == 0.0).float()


def dof_pos_penalty_sj(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str | None = None,
    sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    if command_name is not None and sensor_cfg is not None:
        spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    _init_sj_state(env)
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ).sum(dim=1)
    return err


def dof_hip_pos_penalty_sj(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    ).sum(dim=1)
    return err


def spring_jump_ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.root_ang_vel_b), dim=1)


def spring_jump_torques(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def spring_jump_dof_vel_limits(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    joint_vel_limits = asset.data.joint_vel_limits[:, asset_cfg.joint_ids]
    return torch.sum((torch.abs(joint_vel) - joint_vel_limits).clip(min=0.0), dim=1)


def spring_jump_dof_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def spring_jump_collision(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.1,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > float(threshold)
    return torch.sum(contact.float(), dim=1)


def spring_jump_feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 150.0,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_norm = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    return torch.sum((force_norm - float(max_contact_force)).clip(min=0.0), dim=1)


def lin_vel_z_stance(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2]) * env._sj_has_jumped.float()


def stand_still_crouch_setting(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    target_joint_angles: dict[str, float] | None = None,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    jump_flag = _jump_flag(env, command_name)
    if target_joint_angles is not None:
        if not hasattr(env, "_sj_stand_still_crouch_target"):
            target = asset.data.default_joint_pos[0].clone()
            name_to_idx = {name: idx for idx, name in enumerate(asset.joint_names)}
            for name, value in target_joint_angles.items():
                if name in name_to_idx:
                    target[name_to_idx[name]] = float(value)
            env._sj_stand_still_crouch_target = target
        target_pos = env._sj_stand_still_crouch_target.unsqueeze(0)
    else:
        target_pos = asset.data.default_joint_pos
    err = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - target_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return err * (jump_flag == 0.0).float()


def stand_still_setting(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    asset: Articulation = env.scene[asset_cfg.name]
    err = torch.sum(
        torch.abs(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )
    return err * env._sj_has_jumped.float()


def action_rate_l2_sj(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    jump_action_rate_scale: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    spring_jump_update(env, command_name, sensor_cfg, asset_cfg, contact_threshold)
    raw = torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=1)
    jump_flag = _jump_flag(env, command_name)
    in_jump = (jump_flag == 1.0) & (~env._sj_has_jumped)
    scale = torch.where(
        in_jump,
        torch.full((env.num_envs,), float(jump_action_rate_scale), device=env.device),
        torch.ones(env.num_envs, device=env.device),
    )
    return raw * scale


def orientation_jump(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.exp(-torch.sum(torch.abs(asset.data.projected_gravity_b[:, :2]), dim=1))

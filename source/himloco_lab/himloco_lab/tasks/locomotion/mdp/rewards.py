from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.managers import ManagerTermBase
from isaaclab.managers import RewardTermCfg as RewTerm


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from himloco_lab.envs import HimlocoManagerBasedRLEnv

"""
Joint penalties.
"""


def _upright_gate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7


def _target_joint_pos(
    asset: Articulation, joint_ids: list[int] | slice, target_joint_pos: dict[str, float] | None
) -> torch.Tensor:
    target_pos = asset.data.default_joint_pos[:, joint_ids]
    if target_joint_pos is None:
        return target_pos

    if isinstance(joint_ids, slice):
        selected_joint_ids = list(range(len(asset.joint_names)))[joint_ids]
    else:
        selected_joint_ids = joint_ids
    joint_names = [asset.joint_names[int(joint_id)] for joint_id in selected_joint_ids]
    target_values = torch.tensor(
        [target_joint_pos[joint_name] for joint_name in joint_names],
        dtype=asset.data.joint_pos.dtype,
        device=asset.data.joint_pos.device,
    )
    return target_values.unsqueeze(0).expand_as(target_pos)


def energy(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the energy used by the robot's joints."""
    asset: Articulation = env.scene[asset_cfg.name]

    qvel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    qfrc = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(qvel) * torch.abs(qfrc), dim=-1)


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power."""
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    target_pos = _target_joint_pos(asset, joint_ids, target_joint_pos)
    reward = torch.sum(torch.abs(asset.data.joint_pos[:, joint_ids] - target_pos), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    reward *= cmd_norm < command_threshold
    reward *= _upright_gate(env)
    return reward


"""
Robot.
"""


def orientation_l2(
    env: ManagerBasedRLEnv, desired_gravity: list[float], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward the agent for aligning its gravity with the desired gravity vector using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    desired_gravity = torch.tensor(desired_gravity, device=env.device)
    cos_dist = torch.sum(asset.data.projected_gravity_b * desired_gravity, dim=-1)  # cosine distance
    normalized = 0.5 * cos_dist + 0.5  # map from [-1, 1] to [0, 1]
    return torch.square(normalized)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward keeping the base upright via projected gravity (matches robot_lab).

    g_z = projected_gravity_b[:, 2]: -1 upright, 0 sideways, +1 inverted.
    Returns (1 - g_z)^2, which peaks at 4 upright, 1 sideways, 0 inverted.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= _upright_gate(env)
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= _upright_gate(env)
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= _upright_gate(env)
    return reward


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= _upright_gate(env)
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= _upright_gate(env)
    return reward


def not_moving_when_commanded(
    env: ManagerBasedRLEnv,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize near-zero planar body speed when a planar velocity command is active."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_cmd_norm = torch.linalg.norm(env.command_manager.get_command(command_name)[:, :2], dim=1)
    body_vel_norm = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    reward = torch.clamp(velocity_threshold - body_vel_norm, min=0.0) / velocity_threshold
    reward *= lin_cmd_norm > command_threshold
    reward *= _upright_gate(env)
    return reward


def joint_position_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_name: str = "base_velocity",
    command_threshold: float = 0.1,
    target_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize joint position error from a target posture on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    target_pos = _target_joint_pos(asset, joint_ids, target_joint_pos)
    reward = torch.linalg.norm((asset.data.joint_pos[:, joint_ids] - target_pos), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        reward,
        stand_still_scale * reward,
    )
    reward *= _upright_gate(env)
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    target_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    """Penalize joint position error from a target posture on the articulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    target_pos = _target_joint_pos(asset, joint_ids, target_joint_pos)
    running_reward = torch.linalg.norm((asset.data.joint_pos[:, joint_ids] - target_pos), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= _upright_gate(env)
    return reward

def smoothness(env: HimlocoManagerBasedRLEnv) -> torch.Tensor:
    """Penalize the rate of change of the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action*2 + env.pre_pre_action), dim=1)

"""
Feet rewards.
"""


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= _upright_gate(env)
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env)
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Penalize swing foot height error, weighted by lateral foot speed."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env)
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using first-contact air time."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env)
    return reward


def feet_too_near(
    env: ManagerBasedRLEnv, threshold: float = 0.2, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_contact_without_cmd(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    Reward for feet contact when the command is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < 0.1
    reward *= _upright_gate(env)
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize mismatch between expected and observed foot contact count while moving."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    reward *= torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    reward *= _upright_gate(env)
    return reward


def air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= _upright_gate(env)
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= _upright_gate(env)
    return reward
    
def base_height(
    env: ManagerBasedRLEnv,
    target_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if target_height is None:
        return asset.data.root_pos_w[:, 2:3]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        
        # Replace invalid values (NaN, Inf, too large) with NaN for masked mean
        valid_mask = ~torch.isnan(ray_hits) & ~torch.isinf(ray_hits) & (torch.abs(ray_hits) < 1e6)
        ray_hits_masked = torch.where(valid_mask, ray_hits, torch.tensor(float('nan'), device=ray_hits.device))
        
        # Compute mean ignoring NaN (i.e., ignoring invalid points)
        # nanmean computes mean per environment, automatically ignoring NaN values
        adjusted_heights = torch.nanmean(ray_hits_masked, dim=1)
        
        # For environments where ALL points are invalid, nanmean returns NaN
        # Replace these with current robot height
        all_invalid_mask = torch.isnan(adjusted_heights)
        if all_invalid_mask.any():
            invalid_env_count = all_invalid_mask.sum().item()
            total_invalid_points = (~valid_mask).sum().item()
            print(f"[WARNING] base_height - {invalid_env_count} envs with all invalid points, total invalid: {total_invalid_points}/{ray_hits.numel()} ({total_invalid_points/ray_hits.numel():.2%})")
            adjusted_heights[all_invalid_mask] = asset.data.root_link_pos_w[all_invalid_mask, 2] - target_height
        
        adjusted_target_height = target_height + adjusted_heights
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel."""
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        adjusted_target_height = target_height
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= _upright_gate(env)
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    reward = torch.sum(is_contact, dim=1).float()
    reward *= _upright_gate(env)
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding in the body frame while the foot is in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_lateral_vel * contacts, dim=1)
    reward *= _upright_gate(env)
    return reward


"""
Feet Gait rewards.
"""


def feet_gait(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name=None,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    phases = []
    for offset_ in offset:
        phase = (global_phase + offset_) % 1.0
        phases.append(phase)
    leg_phase = torch.cat(phases, dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for i in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, i] < threshold
        reward += ~(is_stance ^ is_contact[:, i])

    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
        reward *= cmd_norm > 0.1
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = torch.linalg.norm(env.command_manager.get_command(self.command_name), dim=1)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward
    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

"""
Other rewards.
"""


def joint_mirror(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
    use_default_offset: bool = False,
    target_joint_pos: dict[str, float] | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    joint_pos = asset.data.joint_pos
    if target_joint_pos is not None:
        joint_pos = joint_pos - _target_joint_pos(asset, slice(None), target_joint_pos)
    elif use_default_offset:
        joint_pos = joint_pos - asset.data.default_joint_pos
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        reward += torch.sum(
            torch.square(joint_pos[:, joint_pair[0][0]] - joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


"""
Rough-terrain rewards ported from rl-trained-robot-dog (agent_ppo).

These follow the "stair-aware" design of that project's standard_locomotion config:
barrier-style orientation/height penalties (Kim et al., 2025), world-frame progress
drive, a touchdown-event foot clearance, and a set of anti-exploit penalties. Joint
indices are resolved by name via SceneEntityCfg (never hard-coded) so the port is
robust to Isaac Lab's internal joint ordering.
"""


def _reset_mask(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the current reset mask (terminated | timed-out), safe before init."""
    term_mgr = getattr(env, "termination_manager", None)
    if term_mgr is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return term_mgr.terminated | term_mgr.time_outs


def _world_heading_vectors(asset: RigidObject) -> tuple[torch.Tensor, torch.Tensor]:
    """Return body forward and left unit vectors in the world XY frame."""
    heading = asset.data.heading_w
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    forward = torch.stack([cos_h, sin_h], dim=1)
    left = torch.stack([-sin_h, cos_h], dim=1)
    return forward, left


def stair_orientation(
    env: ManagerBasedRLEnv,
    pitch_threshold: float = 0.28,
    barrier_delta: float = 0.3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Barrier-function tilt penalty (Kim et al., 2025).

    Safe zone (tilt <= pitch_threshold): no penalty, giving the policy freedom to
    lean forward for stair climbing. Beyond threshold: logarithmic barrier with a
    clamped ratio (max 0.9) to avoid extreme values.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    tilt = torch.sqrt(torch.sum(torch.square(gravity_xy), dim=1))
    excess = torch.clamp(tilt - pitch_threshold, min=0.0)
    ratio = torch.clamp(excess / max(barrier_delta, 1e-6), max=0.9)
    safe = excess <= 0.0
    result = -torch.log(1.0 - ratio)
    return result.masked_fill(safe, 0.0)


def stair_base_height(
    env: ManagerBasedRLEnv,
    flat_height: float = 0.32,
    stair_height: float = 0.50,
    low_barrier_delta: float = 0.15,
    high_barrier_delta: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Asymmetric barrier-function base-height penalty.

    Low side (ground crawling): logarithmic barrier with a narrow delta -> sharp
    gradient that strongly discourages being too low. High side (stair climbing):
    logarithmic barrier with a wider delta and 0.3x weight -> gentle gradient that
    tolerates height gain. Follows the reference: uses the raw world-frame base
    height (no terrain compensation).
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]

    low_excess = torch.clamp(flat_height - h, min=0.0)
    low_ratio = torch.clamp(low_excess / max(low_barrier_delta, 1e-6), max=0.9)
    low_penalty = (-torch.log(1.0 - low_ratio)).masked_fill(low_excess <= 0.0, 0.0)

    high_excess = torch.clamp(h - stair_height, min=0.0)
    high_ratio = torch.clamp(high_excess / max(high_barrier_delta, 1e-6), max=0.9)
    high_penalty = (-torch.log(1.0 - high_ratio)).masked_fill(high_excess <= 0.0, 0.0)

    return low_penalty + 0.3 * high_penalty


def distance_progress(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward world-frame radial progress speed away from the spawn point.

    Spinning in place yields zero displacement -> zero reward, closing the circling
    exploit that body-frame velocity rewards cannot prevent. Returns delta / step_dt
    (the reward manager applies dt scaling). Uses two-step reset masking: Isaac Lab
    auto-resets AFTER reward computation, so the post-reset spawn appears at t+1 with
    reset_mask=False; without the previous-step mask the first frame of a new episode
    would emit a huge spurious negative delta.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2]
    spawn = env.scene.env_origins[:, :2]
    dist = torch.norm(pos - spawn, dim=1)

    if getattr(env, "_distance_progress_prev", None) is None:
        env._distance_progress_prev = dist.clone()
        env._distance_progress_prev_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return torch.zeros_like(dist)

    delta = dist - env._distance_progress_prev
    reset_mask = _reset_mask(env)
    prev_reset = getattr(env, "_distance_progress_prev_reset", None)
    if prev_reset is not None:
        delta[prev_reset] = 0.0
    delta[reset_mask] = 0.0
    env._distance_progress_prev_reset = reset_mask.clone()
    env._distance_progress_prev = dist.clone()
    step_dt = max(float(getattr(env, "step_dt", 1.0)), 1e-6)
    return delta / step_dt


def x_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward world-frame displacement projected onto the commanded direction.

    Complements distance_progress: radial progress prevents spinning in place,
    while command-direction progress reduces the incentive to cut diagonally
    through stair/tile corners. Returns clamp(delta / step_dt, -1.0, 1.5) with the
    same two-step reset masking as distance_progress.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w[:, :2]

    if getattr(env, "_x_progress_prev", None) is None:
        env._x_progress_prev = pos.clone()
        env._x_progress_prev_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        return torch.zeros(env.num_envs, device=env.device)

    delta_pos = pos - env._x_progress_prev
    cmd_xy = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
    body_forward, body_left = _world_heading_vectors(asset)
    cmd_dir_w = cmd_xy[:, 0:1] * body_forward + cmd_xy[:, 1:2] * body_left
    cmd_dir_w = cmd_dir_w / torch.norm(cmd_dir_w, dim=1, keepdim=True).clamp(min=1e-6)
    delta = torch.sum(delta_pos * cmd_dir_w, dim=1)
    delta = torch.where(cmd_norm.squeeze(1) > 0.05, delta, torch.zeros_like(delta))

    reset_mask = _reset_mask(env)
    prev_reset = getattr(env, "_x_progress_prev_reset", None)
    if prev_reset is not None:
        delta[prev_reset] = 0.0
    delta[reset_mask] = 0.0
    env._x_progress_prev_reset = reset_mask.clone()
    env._x_progress_prev = pos.clone()
    step_dt = max(float(getattr(env, "step_dt", 1.0)), 1e-6)
    return torch.clamp(delta / step_dt, min=-1.0, max=1.5)


def forward_progress(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    tilt_gate_threshold: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward body-frame forward velocity aligned with the command.

    Overlap mitigation: on flat ground (tilt < threshold) this reward is gated down
    because track_lin_vel_xy already covers the same signal. On tilted terrain
    (stairs/slopes) the gate opens to provide the body-frame driving force that
    world-frame tracking cannot give.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    gravity_xy = asset.data.projected_gravity_b[:, :2]
    tilt = torch.sqrt(torch.sum(torch.square(gravity_xy), dim=1))
    gate = torch.clamp(tilt / max(tilt_gate_threshold, 1e-6), max=1.0)

    v_xy = asset.data.root_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1, keepdim=True).clamp(min=0.1)
    cmd_dir = cmd / cmd_norm
    forward_vel = torch.sum(v_xy * cmd_dir, dim=1)
    return torch.clamp(forward_vel, min=0.0) * gate


def stuck_penalty(
    env: ManagerBasedRLEnv,
    speed_threshold: float = 0.08,
    command_threshold: float = 0.15,
    grace_time: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize staying nearly still while commanded to move (after a grace period)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command("base_velocity")
    commanded = torch.norm(cmd[:, :2], dim=1) > command_threshold
    speed = torch.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    stuck_now = commanded & (speed < speed_threshold)

    if getattr(env, "_stuck_time", None) is None:
        env._stuck_time = torch.zeros(env.num_envs, device=env.device)

    dt = max(float(getattr(env, "step_dt", 1.0)), 1e-6)
    env._stuck_time = torch.where(stuck_now, env._stuck_time + dt, torch.zeros_like(env._stuck_time))
    env._stuck_time[_reset_mask(env)] = 0.0
    return torch.clamp(env._stuck_time - grace_time, min=0.0)


def prolonged_swing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_swing_time: float = 0.65,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize any foot that stays airborne longer than max_swing_time while moving.

    Matches the reference line-for-line. The unbounded squared excess is safe because
    the base_contact fall termination resets a downed robot before its swing feet can
    accumulate seconds of air time.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    excess = torch.clamp(air_time - max_swing_time, min=0.0)
    penalty = torch.sum(torch.square(excess), dim=1)
    is_moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return penalty * is_moving


def leg_activity(
    env: ManagerBasedRLEnv,
    sigma: float = 0.5,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize a nearly dead leg during commanded motion.

    The joint velocities selected by ``asset_cfg`` are reshaped into (num_legs, 3);
    pass the 12 leg joints in per-leg [hip, thigh, calf] order via the joint-name cfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    n_legs = joint_vel.shape[1] // 3
    joint_vel = joint_vel.view(joint_vel.shape[0], n_legs, 3)
    leg_speed = torch.norm(joint_vel, dim=2)
    inactivity = torch.exp(-leg_speed / max(sigma, 1e-6))
    max_inactivity = inactivity.max(dim=1)[0]
    is_moving = (torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1).float()
    return max_inactivity * is_moving


def lateral_drift(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    track_std: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize lateral velocity (perpendicular to the command) while tracking forward.

    A velocity-tracking Gaussian gate keeps the penalty inactive when standing still
    (where noise dominates), targeting the side-sliding exploit.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    v_xy = asset.data.root_lin_vel_b[:, :2]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    cmd_norm = torch.norm(cmd, dim=1, keepdim=True).clamp(min=1e-6)
    cmd_dir = cmd / cmd_norm
    forward_vel = torch.sum(v_xy * cmd_dir, dim=1, keepdim=True)
    lateral_vec = v_xy - forward_vel * cmd_dir
    lateral_penalty = torch.sum(torch.square(lateral_vec), dim=1)
    track_err = torch.sum(torch.square(v_xy - cmd), dim=1)
    rvx = torch.exp(-track_err / max(track_std, 1e-6))
    return lateral_penalty * rvx


def hip_abduction(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize excessive hip abduction plus left-right asymmetric hip offsets.

    ``asset_cfg`` must select the four hip joints in the order
    [front-left, front-right, rear-left, rear-right]. Beyond the per-joint L2
    penalty, a left/right asymmetry term targets the "front-legs-left,
    rear-legs-right" drift pattern that a symmetric-only penalty misses.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    hip_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    hip_default = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    delta = hip_pos - hip_default
    base_penalty = torch.sum(torch.square(delta), dim=1)
    front_asym = torch.square(delta[:, 0] - delta[:, 1])
    rear_asym = torch.square(delta[:, 2] - delta[:, 3])
    return base_penalty + 0.5 * (front_asym + rear_asym)


def foot_clearance_touchdown(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    target_height: float = 0.08,
    min_air_time: float = 0.07,
    max_air_time: float = 0.30,
    min_command: float = 0.10,
    progress_speed: float = 0.25,
    min_step_length: float = 0.05,
) -> torch.Tensor:
    """Reward a completed swing step that clears a height and advances forward.

    This is a touchdown-event reward, not an air-time reward: a foot gets credit only
    when it lands after a swing that both clears ``target_height`` and advances at
    least ``min_step_length`` along the command direction. That blocks the
    tripod + one-foot-tapping exploit. Per-foot swing state is tracked on ``env``
    with contact/reset resetting.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )

    reset_mask = _reset_mask(env)
    if getattr(env, "_foot_clearance_contact_z", None) is None:
        env._foot_clearance_contact_z = foot_z.clone()
        env._foot_clearance_liftoff_xy = foot_xy.clone()
        env._foot_clearance_max_lift = torch.zeros_like(foot_z)
        env._foot_clearance_max_step = torch.zeros_like(foot_z)

    contact_z = env._foot_clearance_contact_z.clone()
    liftoff_xy = env._foot_clearance_liftoff_xy.clone()
    max_lift = env._foot_clearance_max_lift.clone()
    max_step = env._foot_clearance_max_step.clone()

    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    current_contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    first_contact = (current_contact_time > 0.0) & (current_contact_time <= env.step_dt + 1e-8)

    command = env.command_manager.get_command(command_name)
    cmd_xy = command[:, :2]
    cmd_norm = torch.norm(cmd_xy, dim=1, keepdim=True)
    body_forward, body_left = _world_heading_vectors(asset)
    cmd_dir_w = cmd_xy[:, 0:1] * body_forward + cmd_xy[:, 1:2] * body_left
    cmd_dir_w = cmd_dir_w / torch.norm(cmd_dir_w, dim=1, keepdim=True).clamp(min=1e-6)

    lift = torch.clamp(foot_z - contact_z, min=0.0)
    step = torch.clamp(torch.sum((foot_xy - liftoff_xy) * cmd_dir_w[:, None, :], dim=2), min=0.0)
    swing = ~contacts
    max_lift = torch.where(swing, torch.maximum(max_lift, lift), max_lift)
    max_step = torch.where(swing, torch.maximum(max_step, step), max_step)

    v_xy = asset.data.root_lin_vel_b[:, :2]
    cmd_dir_b = cmd_xy / cmd_norm.clamp(min=1e-6)
    forward_speed = torch.clamp(torch.sum(v_xy * cmd_dir_b, dim=1), min=0.0)
    moving_gate = (cmd_norm.squeeze(1) > min_command).float()
    progress_gate = torch.clamp(forward_speed / max(progress_speed, 1e-6), max=1.0)

    valid_swing = (last_air_time >= min_air_time) & (last_air_time <= max_air_time)
    valid_step = max_step >= min_step_length
    clearance_score = torch.clamp(max_lift / max(target_height, 1e-6), max=1.0)
    touchdown_reward = clearance_score * first_contact.float() * valid_swing.float() * valid_step.float()
    reward = torch.sum(touchdown_reward, dim=1) * moving_gate * progress_gate

    contact_or_reset = contacts | reset_mask[:, None]
    contact_z = torch.where(contact_or_reset, foot_z, contact_z)
    liftoff_xy = torch.where(contact_or_reset[:, :, None], foot_xy, liftoff_xy)
    max_lift = torch.where(contact_or_reset, torch.zeros_like(max_lift), max_lift)
    max_step = torch.where(contact_or_reset, torch.zeros_like(max_step), max_step)

    env._foot_clearance_contact_z = contact_z.clone()
    env._foot_clearance_liftoff_xy = liftoff_xy.clone()
    env._foot_clearance_max_lift = max_lift.clone()
    env._foot_clearance_max_step = max_step.clone()

    reward[reset_mask] = 0.0
    return reward


"""
Rough-terrain variants of shared base rewards WITHOUT the himloco `_upright_gate`
multiplier, to match the reference project line-for-line. The shared (flat) functions
above keep the gate and are left untouched; these are used only by RoughRewardsCfg.
"""


def track_lin_vel_xy_rough(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track xy linear velocity (exp kernel), no upright gate (reference-faithful)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_rough(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track yaw angular velocity (exp kernel), no upright gate (reference-faithful)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2_rough(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z linear velocity (L2), no upright gate (reference-faithful)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2_rough(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy angular velocity (L2), no upright gate (reference-faithful)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def flat_orientation_l2_rough(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize non-flat base orientation (L2), no upright gate (reference-faithful)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def undesired_contacts_rough(
    env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize undesired contacts above a threshold, no upright gate (reference-faithful)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1).float()


def feet_slide_rough(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize foot sliding while in contact using WORLD-frame foot speed.

    Reference-faithful: uses ``body_lin_vel_w[..., :2].norm()`` directly (no body-frame
    transform) and no upright gate.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_stumble_rough(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces (forces_xy > 5*forces_z), no gate.

    Reference-faithful: threshold is 5x (himloco's shared feet_stumble uses 4x).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > 5 * forces_z, dim=1).float()


def joint_pos_penalty_rough(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Penalize joint deviation from default, scaled up when standing still, no gate.

    Reference-faithful: the "moving" test is ``cmd > 0.0`` (himloco's shared version
    uses ``cmd > command_threshold``) and there is no upright gate.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    return torch.where(
        torch.logical_or(cmd > 0.0, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )


def joint_power_rough(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Absolute joint mechanical power sum with NaN guard (reference energy_biomechanical)."""
    asset: Articulation = env.scene[asset_cfg.name]
    power = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return torch.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)

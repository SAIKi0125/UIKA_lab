from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import RayCaster


def base_external_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """observe external force applied on the base"""
    asset: Articulation = env.scene[asset_cfg.name]
    # shape: (num_envs, 3)
    return asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :].squeeze(1).clone()


def base_height(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root height in world frame."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2:3]


def base_euler_xyz(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root roll, pitch, yaw in XYZ convention."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    return torch.stack(euler_xyz_from_quat(asset.data.root_quat_w), dim=-1)


def contact_mask(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 5.0) -> torch.Tensor:
    """Binary contact mask for selected bodies from world-frame z contact force."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]
    return (forces_z > float(threshold)).float()


def joint_pos_rel_to_target(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_joint_pos: dict[str, float],
) -> torch.Tensor:
    """Joint positions relative to a named target posture instead of asset default."""
    asset: Articulation = env.scene[asset_cfg.name]
    target_values = torch.tensor(
        [target_joint_pos[asset.joint_names[int(joint_id)]] for joint_id in asset_cfg.joint_ids],
        dtype=asset.data.joint_pos.dtype,
        device=asset.data.joint_pos.device,
    )
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - target_values.unsqueeze(0)


def height_scan_clip(
    env: ManagerBasedRLEnv, 
    sensor_cfg: SceneEntityCfg,
    clip: tuple[float, float] = (-1.0, 1.0), 
    offset: float = 0.5) -> torch.Tensor:
    """Height scan from the given sensor w.r.t. the sensor's frame.

    The provided offset (Defaults to 0.5) is subtracted from the returned values.
    """
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height scan: height = sensor_height - hit_point_z - offset
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clip(height, clip[0], clip[1])

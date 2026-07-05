from __future__ import annotations

import torch
from typing import TYPE_CHECKING

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


def waypoint_height_delta(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Current waypoint surface height relative to the robot base height."""
    asset: Articulation = env.scene[asset_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    marker_routes = getattr(command_term, "_marker_routes", None)
    if marker_routes is None:
        return torch.zeros(
            asset.data.root_pos_w.shape[0],
            1,
            dtype=asset.data.root_pos_w.dtype,
            device=asset.data.root_pos_w.device,
        )

    route_ids = command_term._terrain_route_ids().to(device=marker_routes.device, dtype=torch.long)
    goal_idx = command_term.goal_idx.to(device=marker_routes.device, dtype=torch.long)
    waypoint_z = marker_routes[route_ids, goal_idx, 2].to(
        device=asset.data.root_pos_w.device, dtype=asset.data.root_pos_w.dtype
    )
    return (waypoint_z - asset.data.root_pos_w[:, 2]).unsqueeze(1)


def foot_contact_state(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 2.0) -> torch.Tensor:
    """Filtered foot contact state as centered binary features."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    previous_contact_forces = contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]
    contacts = torch.linalg.norm(contact_forces, dim=-1) > threshold
    previous_contacts = torch.linalg.norm(previous_contact_forces, dim=-1) > threshold
    return torch.logical_or(contacts, previous_contacts).to(torch.float) - 0.5


def terrain_route_one_hot(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """One-hot encoded waypoint route/terrain id for the current environment."""
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "_terrain_route_ids") and hasattr(command_term, "_route_names"):
        route_ids = command_term._terrain_route_ids().to(dtype=torch.long)
        num_routes = len(command_term._route_names)
        return torch.nn.functional.one_hot(route_ids, num_classes=num_routes).to(torch.float)

    terrain = getattr(env.scene, "terrain", None)
    terrain_types = getattr(terrain, "terrain_types", None)
    terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
    sub_terrains = getattr(terrain_cfg, "sub_terrains", None)
    if terrain_types is None or sub_terrains is None:
        return torch.zeros(env.num_envs, 1, dtype=torch.float, device=env.device)

    num_terrains = len(sub_terrains)
    terrain_ids = torch.clamp(terrain_types.to(device=env.device, dtype=torch.long), min=0, max=num_terrains - 1)
    return torch.nn.functional.one_hot(terrain_ids, num_classes=num_terrains).to(torch.float)


def base_mass_com(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Base mass and center-of-mass position privileged observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else [0]
    com_pos_b = asset.data.com_pos_b
    masses = asset.root_physx_view.get_masses()[:, body_ids].to(device=com_pos_b.device, dtype=com_pos_b.dtype)
    com_pos = com_pos_b[:, body_ids, :].reshape(masses.shape[0], -1)
    return torch.cat((masses, com_pos), dim=-1)


def body_friction(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """First material static-friction privileged observation."""
    asset: Articulation = env.scene[asset_cfg.name]
    material_properties = asset.root_physx_view.get_material_properties()
    device = getattr(env, "device", material_properties.device)
    return material_properties[:, 0, 0].to(device=device).unsqueeze(1)


def joint_stiffness_damping_scale(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint stiffness and damping scale offsets from randomized actuator gains."""
    asset: Articulation = env.scene[asset_cfg.name]
    stiffness_scale = asset.data.joint_stiffness / asset.data.default_joint_stiffness - 1.0
    damping_scale = asset.data.joint_damping / asset.data.default_joint_damping - 1.0
    return torch.cat((stiffness_scale, damping_scale), dim=-1)

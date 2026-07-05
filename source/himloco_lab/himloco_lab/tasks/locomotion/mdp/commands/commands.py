from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils

from isaaclab.markers import VisualizationMarkers
from isaaclab.envs.mdp import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import (
        UniformLevelVelocityCommandCfg,
        UniformThresholdVelocityCommandCfg,
        WaypointVelocityCommandCfg,
    )


def _is_robot_on_terrain(env: ManagerBasedEnv, terrain_name: str, asset_name: str = "robot") -> torch.Tensor:
    terrain = getattr(env.scene, "terrain", None)
    if terrain is None or not hasattr(terrain, "terrain_types"):
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_type != "generator" or terrain.cfg.terrain_generator is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain.cfg.terrain_generator.sub_terrains is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if terrain_name not in terrain.cfg.terrain_generator.sub_terrains:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    terrain_cfg = terrain.cfg.terrain_generator
    sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
    proportions = torch.tensor([sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()], device=env.device)
    proportions = proportions / proportions.sum()
    cumsum_props = torch.cumsum(proportions, dim=0)

    terrain_idx = sub_terrain_names.index(terrain_name)
    col_start = round((0.0 if terrain_idx == 0 else cumsum_props[terrain_idx - 1].item()) * terrain_cfg.num_cols)
    col_end = round(cumsum_props[terrain_idx].item() * terrain_cfg.num_cols)

    asset = env.scene[asset_name]
    robot_pos_w = asset.data.root_pos_w[:, :2]
    terrain_origins_2d = terrain.terrain_origins[:, :, :2].reshape(-1, 2)
    closest_flat_idx = torch.argmin(torch.cdist(robot_pos_w, terrain_origins_2d), dim=1)
    col_idx = closest_flat_idx % terrain.terrain_origins.shape[1]
    return (col_idx >= col_start) & (col_idx < col_end)


class UniformThresholdVelocityCommand(UniformVelocityCommand):
    """RobotLab-style uniform velocity command with small planar commands thresholded to zero."""

    cfg: UniformThresholdVelocityCommandCfg

    def __init__(self, cfg: UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        super()._update_command()

        on_pits = _is_robot_on_terrain(self._env, "pits")
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            self._resample_command(torch.where(left_pit_mask)[0])

        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0
            self.vel_command_b[pit_env_ids, 2] = 0.0
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        self.was_on_pit = on_pits


class WaypointVelocityCommand(UniformVelocityCommand):
    """Velocity command that points the robot toward terrain-local waypoints."""

    cfg: WaypointVelocityCommandCfg

    def __init__(self, cfg: WaypointVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.route_speed = torch.full((self.num_envs,), cfg.speed_range[0], dtype=torch.float, device=self.device)
        self._tile_half = torch.tensor(cfg.tile_size, dtype=torch.float, device=self.device)[:2] * 0.5
        self._route_names = tuple(cfg.terrain_names)
        self._route_lengths = torch.tensor(
            [len(cfg.routes[name]) for name in self._route_names], dtype=torch.long, device=self.device
        )
        max_goals = int(self._route_lengths.max().item())
        self._routes = torch.zeros(len(self._route_names), max_goals, 2, dtype=torch.float, device=self.device)
        for route_id, name in enumerate(self._route_names):
            self._routes[route_id, : len(cfg.routes[name])] = torch.tensor(
                cfg.routes[name], dtype=torch.float, device=self.device
            )
        self._marker_routes = None
        if cfg.marker_routes is not None:
            self._marker_routes = torch.zeros(
                len(self._route_names), max_goals, 3, dtype=torch.float, device=self.device
            )
            for route_id, name in enumerate(self._route_names):
                marker_route = torch.tensor(cfg.marker_routes[name], dtype=torch.float, device=self.device)
                if marker_route.shape != (len(cfg.routes[name]), 3):
                    raise ValueError(f"Marker route for terrain '{name}' must have one xyz marker per waypoint.")
                self._marker_routes[route_id, : marker_route.shape[0]] = marker_route
        self.metrics["distance_to_waypoint"] = torch.zeros(self.num_envs, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return
        first_sample = self.command_counter[env_ids] == 0
        if torch.any(first_sample):
            self.goal_idx[env_ids[first_sample]] = 0
        r = torch.empty(env_ids.numel(), device=self.device)
        self.route_speed[env_ids] = r.uniform_(*self.cfg.speed_range)

    def _update_command(self):
        route_ids = self._terrain_route_ids()
        env_origins = self._env.scene.env_origins[:, :2]
        target_local = self._routes[route_ids, self.goal_idx]
        target_w = env_origins + target_local - self._tile_half

        target_vec_w = target_w - self.robot.data.root_pos_w[:, :2]
        distance = torch.norm(target_vec_w, dim=1)
        route_last_goal = self._route_lengths[route_ids] - 1
        reached = (distance < self.cfg.waypoint_threshold) & (self.goal_idx < route_last_goal)
        if torch.any(reached):
            self.goal_idx[reached] += 1
            target_local = self._routes[route_ids, self.goal_idx]
            target_w = env_origins + target_local - self._tile_half
            target_vec_w = target_w - self.robot.data.root_pos_w[:, :2]
            distance = torch.norm(target_vec_w, dim=1)

        target_dir_w = target_vec_w / (distance.unsqueeze(1) + 1e-5)
        vel_w = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        vel_w[:, :2] = target_dir_w * self.route_speed.unsqueeze(1)
        vel_b = math_utils.quat_apply_inverse(math_utils.yaw_quat(self.robot.data.root_quat_w), vel_w)
        self.vel_command_b[:, :2] = vel_b[:, :2]

        target_yaw = torch.atan2(target_dir_w[:, 1], target_dir_w[:, 0])
        heading_error = math_utils.wrap_to_pi(target_yaw - self.robot.data.heading_w)
        self.vel_command_b[:, 2] = torch.clip(
            self.cfg.heading_control_stiffness * heading_error,
            min=self.cfg.ranges.ang_vel_z[0],
            max=self.cfg.ranges.ang_vel_z[1],
        )

    def _update_metrics(self):
        route_ids = self._terrain_route_ids()
        env_origins = self._env.scene.env_origins[:, :2]
        target_local = self._routes[route_ids, self.goal_idx]
        target_w = env_origins + target_local - self._tile_half
        self.metrics["distance_to_waypoint"] = torch.norm(target_w - self.robot.data.root_pos_w[:, :2], dim=1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer = VisualizationMarkers(self.cfg.waypoint_visualizer_cfg)
            self.waypoint_visualizer.set_visibility(True)
        else:
            if hasattr(self, "waypoint_visualizer"):
                self.waypoint_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return
        translations, marker_indices = self._debug_waypoint_markers()
        self.waypoint_visualizer.visualize(translations=translations, marker_indices=marker_indices)

    def _terrain_route_ids(self) -> torch.Tensor:
        terrain = getattr(self._env.scene, "terrain", None)
        if terrain is None:
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        route_ids = self._route_ids_from_env_origins(terrain)
        if route_ids is not None:
            return route_ids

        if not hasattr(terrain, "terrain_types"):
            return torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        return torch.clamp(terrain.terrain_types, min=0, max=len(self._route_names) - 1).to(torch.long)

    def _route_ids_from_env_origins(self, terrain) -> torch.Tensor | None:
        cached_route_ids = getattr(self, "_cached_route_ids", None)
        if cached_route_ids is not None:
            return cached_route_ids
        if not hasattr(terrain, "terrain_origins") or not hasattr(self._env.scene, "env_origins"):
            return None

        terrain_origins = terrain.terrain_origins
        if terrain_origins.ndim != 3 or terrain_origins.shape[1] == 0:
            return None

        terrain_origins_2d = terrain_origins[:, :, :2].reshape(-1, 2).to(self.device)
        env_origins = self._env.scene.env_origins[:, :2].to(self.device)
        closest_flat_idx = torch.argmin(torch.cdist(env_origins, terrain_origins_2d), dim=1)
        col_idx = closest_flat_idx % terrain_origins.shape[1]
        route_ids = self._route_ids_from_columns(terrain, col_idx)
        self._cached_route_ids = route_ids
        return route_ids

    def _route_ids_from_columns(self, terrain, col_idx: torch.Tensor) -> torch.Tensor:
        terrain_cfg = getattr(getattr(terrain, "cfg", None), "terrain_generator", None)
        num_cols = int(getattr(terrain_cfg, "num_cols", int(col_idx.max().item()) + 1))
        if terrain_cfg is None or terrain_cfg.sub_terrains is None:
            return torch.clamp(col_idx, min=0, max=len(self._route_names) - 1).to(torch.long)

        sub_terrain_names = list(terrain_cfg.sub_terrains.keys())
        route_name_to_id = {name: route_id for route_id, name in enumerate(self._route_names)}
        column_to_route = torch.zeros(num_cols, dtype=torch.long, device=self.device)

        proportions = torch.tensor(
            [sub_cfg.proportion for sub_cfg in terrain_cfg.sub_terrains.values()],
            dtype=torch.float,
            device=self.device,
        )
        proportions = proportions / torch.clamp(proportions.sum(), min=1e-6)
        cumsum_props = torch.cumsum(proportions, dim=0)

        col_start = 0
        for terrain_id, terrain_name in enumerate(sub_terrain_names):
            col_end = round(cumsum_props[terrain_id].item() * num_cols)
            col_end = max(col_start, min(num_cols, col_end))
            route_id = route_name_to_id.get(terrain_name, 0)
            column_to_route[col_start:col_end] = route_id
            col_start = col_end

        return column_to_route[torch.clamp(col_idx, min=0, max=num_cols - 1)]

    def _debug_waypoint_markers(self) -> tuple[torch.Tensor, torch.Tensor]:
        route_ids = self._terrain_route_ids()
        env_origins = self._env.scene.env_origins.to(self.device)
        route_lengths = self._route_lengths[route_ids]
        max_goals = self._routes.shape[1]

        goal_ids = torch.arange(max_goals, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        valid_goal_mask = goal_ids < route_lengths.unsqueeze(1)

        marker_routes = getattr(self, "_marker_routes", None)
        target_local = self._routes[route_ids] if marker_routes is None else marker_routes[route_ids, :, :2]
        translations = torch.zeros(self.num_envs, max_goals, 3, dtype=torch.float, device=self.device)
        translations[:, :, :2] = env_origins[:, None, :2] + target_local - self._tile_half
        if marker_routes is None:
            translations[:, :, 2] = env_origins[:, None, 2] + self.cfg.waypoint_marker_height
        else:
            translations[:, :, 2] = marker_routes[route_ids, :, 2] + getattr(
                self.cfg, "waypoint_marker_surface_offset", 0.0
            )

        marker_indices = torch.zeros(self.num_envs, max_goals, dtype=torch.long, device=self.device)
        marker_indices[torch.arange(self.num_envs, device=self.device), self.goal_idx] = 1

        return translations[valid_goal_mask], marker_indices[valid_goal_mask]


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from a normal distribution.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The command is sampled from a normal distribution with mean and standard deviation specified in
    the configuration. With equal probability, the sign of the individual components is flipped.
    """

    cfg: UniformLevelVelocityCommandCfg
    """The command generator configuration."""

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env: ManagerBasedEnv):
        """Initializes the command generator.

        Args:
            cfg: The command generator configuration.
            env: The environment.
        """
        super().__init__(cfg, env)

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "UniformVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tHeading command: {self.cfg.heading_command}\n"
        if self.cfg.heading_command:
            msg += f"\tHeading probability: {self.cfg.rel_heading_envs}\n"
        return msg

    def _resample_command(self, env_ids: Sequence[int]):
        # sample velocity commands
        r = torch.empty(len(env_ids), device=self.device)
        # -- linear velocity - x direction
        lin_vel_x_range = self.cfg.low_vel_env_lin_x_ranges or self.cfg.ranges.lin_vel_x
        self.vel_command_b[env_ids, 0] = r.uniform_(*lin_vel_x_range)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        if self.cfg.rel_high_vel_envs is not None:
            high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
            high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
            r_high = torch.empty(len(high_vel_env_ids), device=self.device)
            self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)
            if self.cfg.low_vel_env_lin_x_ranges is not None:
                # set y commands of high vel envs to zero
                low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
                low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
                in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & \
                                    (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
                self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range
        
        # set small commands to zero
        if self.cfg.min_command_norm is not None:
            self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > \
                                                self.cfg.min_command_norm).unsqueeze(1)
        
    def _update_command(self):
        """Post-processes the velocity command.

        This function sets velocity command to zero for standing environments and computes angular
        velocity from heading direction if the heading_command flag is set.
        """
        # Compute angular velocity from heading direction
        if self.cfg.heading_command:
            # resolve indices of heading envs
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            # compute angular velocity
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - self.robot.data.heading_w[env_ids])
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

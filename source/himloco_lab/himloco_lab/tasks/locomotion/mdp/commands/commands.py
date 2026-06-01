from __future__ import annotations

from isaaclab.utils import configclass
from dataclasses import MISSING
from typing import TYPE_CHECKING
import torch
from collections.abc import Sequence
import isaaclab.utils.math as math_utils

from isaaclab.envs.mdp import UniformVelocityCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformLevelVelocityCommandCfg, UniformThresholdVelocityCommandCfg


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
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.low_vel_env_lin_x_ranges)
        # -- linear velocity - y direction
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        # -- ang vel yaw - rotation around z
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)
        # heading target
        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            # update heading envs
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        
        high_vel_env_ids = env_ids <= (self.num_envs * self.cfg.rel_high_vel_envs)
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        r_high = torch.empty(len(high_vel_env_ids), device=self.device)
        self.vel_command_b[high_vel_env_ids, 0] = r_high.uniform_(*self.cfg.ranges.lin_vel_x)
        # set y commands of high vel envs to zero
        low_vel_x_min = self.cfg.low_vel_env_lin_x_ranges[0]
        low_vel_x_max = self.cfg.low_vel_env_lin_x_ranges[1]
        in_low_vel_range = (self.vel_command_b[high_vel_env_ids, 0:1] >= low_vel_x_min) & \
                            (self.vel_command_b[high_vel_env_ids, 0:1] <= low_vel_x_max)
        self.vel_command_b[high_vel_env_ids, 1:2] *= in_low_vel_range
        
        # set small commands to zero
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

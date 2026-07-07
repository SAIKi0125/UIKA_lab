from dataclasses import MISSING

from isaaclab.utils import configclass

from .commands import SpringJumpCommand, UniformLevelVelocityCommand, UniformThresholdVelocityCommand

from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.managers import CommandTermCfg


@configclass
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):

    class_type: type = UniformThresholdVelocityCommand


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):

    class_type: type = UniformLevelVelocityCommand

    curriculums_limit_ranges: tuple[float, float] = MISSING
    
    low_vel_env_lin_x_ranges: tuple[float, float] = MISSING
    
    rel_high_vel_envs: float = MISSING
    
    min_command_norm: float = MISSING


@configclass
class SpringJumpCommandCfg(CommandTermCfg):
    """Configuration for the three-channel spring-jump command."""

    class_type: type = SpringJumpCommand

    @configclass
    class Ranges:
        target_x: tuple[float, float] = (0.8, 1.2)
        target_y: tuple[float, float] = (0.0, 0.0)

    asset_name: str = "robot"
    ranges: Ranges = Ranges()
    setting_frame_range: tuple[int, int] = (50, 60)

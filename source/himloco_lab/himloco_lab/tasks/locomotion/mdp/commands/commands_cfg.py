from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.utils import configclass

from .commands import UniformLevelVelocityCommand, UniformThresholdVelocityCommand, WaypointVelocityCommand

from isaaclab.envs.mdp import UniformVelocityCommandCfg

WAYPOINT_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Command/waypoints",
    markers={
        "waypoint": sim_utils.SphereCfg(
            radius=0.055,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.75, 0.05)),
        ),
        "current_waypoint": sim_utils.SphereCfg(
            radius=0.085,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.15)),
        ),
    },
)


@configclass
class UniformThresholdVelocityCommandCfg(UniformVelocityCommandCfg):

    class_type: type = UniformThresholdVelocityCommand


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):

    class_type: type = UniformLevelVelocityCommand

    curriculums_limit_ranges: tuple[float, float] | None = None
    
    low_vel_env_lin_x_ranges: tuple[float, float] | None = None
    
    rel_high_vel_envs: float | None = None
    
    min_command_norm: float | None = None


@configclass
class WaypointVelocityCommandCfg(UniformVelocityCommandCfg):

    class_type: type = WaypointVelocityCommand

    terrain_names: tuple[str, ...] = MISSING

    routes: dict[str, tuple[tuple[float, float], ...]] = MISSING

    marker_routes: dict[str, tuple[tuple[float, float, float], ...]] | None = None

    tile_size: tuple[float, float] = MISSING

    speed_range: tuple[float, float] = MISSING

    waypoint_threshold: float = 0.35

    waypoint_marker_height: float = 0.35

    waypoint_marker_surface_offset: float = 0.0

    waypoint_visualizer_cfg: VisualizationMarkersCfg = WAYPOINT_MARKER_CFG

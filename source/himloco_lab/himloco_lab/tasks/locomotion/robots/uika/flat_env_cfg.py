from isaaclab import sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg, RobotSceneCfg


@configclass
class FlatRobotSceneCfg(RobotSceneCfg):
    """Normal-height UIKA scene on an infinite flat plane."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )


@configclass
class FlatRobotEnvCfg(RobotEnvCfg):
    """Normal-height UIKA velocity training environment on flat ground."""

    scene: FlatRobotSceneCfg = FlatRobotSceneCfg(num_envs=4096, env_spacing=2.5)


@configclass
class FlatRobotPlayEnvCfg(RobotPlayEnvCfg):
    """Flat-ground play environment with the standard fixed forward command."""

    scene: FlatRobotSceneCfg = FlatRobotSceneCfg(num_envs=64, env_spacing=2.5)

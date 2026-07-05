import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import himloco_lab.terrains as him_terrains
from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.robots.uika.velocity_env_cfg import (
    RobotEnvCfg,
    RobotSceneCfg,
    RewardsCfg,
    UIKA_JOINT_NAMES,
)
from himloco_lab.terrains.extreme_parkour import (
    build_extreme_parkour_marker_routes,
    build_extreme_parkour_routes,
    build_flat_parkour_marker_route,
    build_flat_parkour_route,
)


PARKOUR_TILE_SIZE = (8.0, 8.0)
PARKOUR_HORIZONTAL_SCALE = 0.05
EXTREME_PARKOUR_HEIGHTMAP_TERRAIN_NAMES = ("T_step_stl", "Slope", "BridgeA", "BridgeB")
FLAT_PARKOUR_TERRAIN_NAME = "Flat"
EXTREME_PARKOUR_TERRAIN_NAMES = EXTREME_PARKOUR_HEIGHTMAP_TERRAIN_NAMES + (FLAT_PARKOUR_TERRAIN_NAME,)
EXTREME_PARKOUR_ROUTES = build_extreme_parkour_routes(
    tile_size=PARKOUR_TILE_SIZE,
    horizontal_scale=PARKOUR_HORIZONTAL_SCALE,
    vertical_scale=0.005,
)
EXTREME_PARKOUR_ROUTES[FLAT_PARKOUR_TERRAIN_NAME] = build_flat_parkour_route(tile_size=PARKOUR_TILE_SIZE)
EXTREME_PARKOUR_MARKER_ROUTES = build_extreme_parkour_marker_routes(
    tile_size=PARKOUR_TILE_SIZE,
    horizontal_scale=PARKOUR_HORIZONTAL_SCALE,
    vertical_scale=0.005,
)
EXTREME_PARKOUR_MARKER_ROUTES[FLAT_PARKOUR_TERRAIN_NAME] = build_flat_parkour_marker_route(
    EXTREME_PARKOUR_ROUTES[FLAT_PARKOUR_TERRAIN_NAME]
)

EXTREME_PARKOUR_CFG = terrain_gen.TerrainGeneratorCfg(
    size=PARKOUR_TILE_SIZE,
    border_width=25.0,
    num_rows=10,
    num_cols=18,
    horizontal_scale=PARKOUR_HORIZONTAL_SCALE,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        name: him_terrains.HfExtremeParkourHeightmapTerrainCfg(
            proportion=1.0,
            border_width=0.0,
            terrain_name=name,
        )
        for name in EXTREME_PARKOUR_HEIGHTMAP_TERRAIN_NAMES
    }
    | {
        FLAT_PARKOUR_TERRAIN_NAME: terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.5,
        )
    },
)


@configclass
class RobotParkourSceneCfg(RobotSceneCfg):
    """Scene configuration using imported Extreme Parkour terrains."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=EXTREME_PARKOUR_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )


@configclass
class ParkourCommandsCfg:
    """Waypoint route command specifications for Extreme Parkour terrain."""

    base_velocity = mdp.WaypointVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(20.0, 20.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=1.2,
        debug_vis=False,
        terrain_names=EXTREME_PARKOUR_TERRAIN_NAMES,
        routes=EXTREME_PARKOUR_ROUTES,
        marker_routes=EXTREME_PARKOUR_MARKER_ROUTES,
        tile_size=PARKOUR_TILE_SIZE,
        speed_range=(0.35, 0.65),
        waypoint_threshold=0.35,
        ranges=mdp.WaypointVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0), heading=None
        ),
    )


@configclass
class ParkourObservationsCfg:
    """Observation specifications for the imported Extreme Parkour task."""

    @configclass
    class PolicyCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))

        def __post_init__(self):
            self.enable_corruption = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100), noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100), noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
            clip=(-100, 100),
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        base_external_force = ObsTerm(
            func=mdp.base_external_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = True

    critic: CriticCfg = CriticCfg()


@configclass
class ParkourRewardsCfg(RewardsCfg):
    """Reward terms used by the imported Extreme Parkour terrain task."""

    # Disable baseline velocity-task rewards that over-constrain parkour motions.
    is_terminated = None
    base_linear_velocity = None
    base_angular_velocity = None
    joint_acc = None
    energy = None
    action_rate = None
    smoothness = None
    lin_vel_z_l2 = None
    ang_vel_xy_l2 = None
    flat_orientation_l2 = None
    base_height_l2 = None
    body_lin_acc_l2 = None
    upward = None
    joint_torques_l2 = None
    joint_power = None
    joint_vel_l2 = None
    joint_acc_l2 = None
    joint_pos_limits = None
    joint_vel_limits = None
    stand_still = None
    joint_pos_penalty = None
    joint_mirror = None
    action_rate_l2 = None
    undesired_contacts = None
    contact_forces = None
    feet_air_time = None
    feet_air_time_variance = None
    feet_contact = None
    feet_contact_without_cmd = None
    feet_stumble = None
    feet_slide = None
    feet_height = None
    feet_height_body = None
    feet_gait = None

    reward_collision = RewTerm(
        func=mdp.collision_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base", ".*_calf", ".*_thigh"]),
            "threshold": 0.1,
        },
    )

    reward_feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "terrain_names": EXTREME_PARKOUR_TERRAIN_NAMES,
            "tile_size": PARKOUR_TILE_SIZE,
            "horizontal_scale": PARKOUR_HORIZONTAL_SCALE,
            "vertical_scale": 0.005,
            "height_threshold": 0.05,
            "edge_width": 0.05,
            "contact_threshold": 2.0,
            "terrain_level_threshold": None,
        },
    )

    reward_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    reward_dof_error = RewTerm(
        func=mdp.joint_dof_error_l2,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_pos = RewTerm(
        func=mdp.hip_pos_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_ang_vel_xy = RewTerm(func=mdp.source_ang_vel_xy_l2, weight=-0.05)
    reward_action_rate = RewTerm(
        func=mdp.ActionRateNorm,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_term_name": "JointPositionAction",
        },
    )
    reward_dof_acc = RewTerm(
        func=mdp.JointDofAccL2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=mdp.source_lin_vel_z_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_names": EXTREME_PARKOUR_TERRAIN_NAMES,
            "flat_terrain_names": (FLAT_PARKOUR_TERRAIN_NAME,),
            "nonflat_scale": 0.5,
        },
    )
    reward_orientation = RewTerm(
        func=mdp.source_flat_orientation_l2,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "terrain_names": EXTREME_PARKOUR_TERRAIN_NAMES,
            "flat_terrain_names": (FLAT_PARKOUR_TERRAIN_NAME,),
        },
    )
    reward_feet_stumble = RewTerm(
        func=mdp.source_feet_stumble,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    reward_delta_torques = RewTerm(
        func=mdp.DeltaTorquesL2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RobotParkourEnvCfg(RobotEnvCfg):
    """UIKA environment on imported Extreme Parkour terrains with waypoint commands."""

    scene: RobotParkourSceneCfg = RobotParkourSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: ParkourCommandsCfg = ParkourCommandsCfg()
    observations: ParkourObservationsCfg = ParkourObservationsCfg()
    rewards: ParkourRewardsCfg = ParkourRewardsCfg()
    reward_clip_min: float | None = 0.0

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.curriculum.lin_vel_cmd_levels = None
        self.curriculum.command_levels_ang_vel = None

        if self.scene.terrain.terrain_generator is not None:
            # Keep Isaac Lab's column-based terrain generation so fixed route columns remain stable.
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 0


@configclass
class RobotParkourPlayEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64

        self.commands.base_velocity.speed_range = (0.45, 0.45)
        self.commands.base_velocity.debug_vis = True

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp
import himloco_lab.terrains as him_terrains
from himloco_lab.terrains.extreme_parkour import (
    build_extreme_parkour_marker_routes,
    build_extreme_parkour_routes,
    build_flat_parkour_marker_route,
    build_flat_parkour_route,
)

UIKA_JOINT_NAMES = list(ROBOT_CFG.joint_sdk_names)

UIKA_PACE_ACTUATOR_RANGES = {
    "hip": {
        "armature": (0.012499551, 0.014251016),
        "viscous_friction": (0.000871807, 0.002837807),
        "friction": (0.005748361, 0.018038273),
    },
    "thigh": {
        "armature": (0.012442659, 0.014217399),
        "viscous_friction": (0.001674354, 0.003362268),
        "friction": (0.009873092, 0.022234440),
    },
    "calf": {
        "armature": (0.022163186, 0.024222745),
        "viscous_friction": (0.001198500, 0.002290815),
        "friction": (0.007793665, 0.015262470),
    },
}

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
        ),
        "hf_slope_with_noise": him_terrains.HfPyramidSlopeWithNoiseCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
            noise_amplitude_range=(0.01, 0.08),
            noise_step=0.005,
            downsampled_scale=0.2,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.0,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.15),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.0,
        ),
        "discrete_obstacles": him_terrains.HfDiscreteObstaclesTerrainCfg(
            proportion=0.1,
            max_height_range=(0.05, 0.15),
            obstacle_size_range=(1.0, 2.0),
            num_obstacles=20,
            platform_width=3.0,
        ),
    },
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
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with UIKA robot."""

    # --- 平地模式（取消注释以启用） ---
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #     ),
    #     debug_vis=False,
    # )
    # --- 地形模式 ---
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
        max_init_terrain_level=0,
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
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    base_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=[0.3, 0.4]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
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
class EventCfg:
    """Configuration for events."""

    # startup randomization: keep the HimLoco baseline outside reset-time sampling.
    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.25),
            "dynamic_friction_range": (0.2, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset randomization.
    # Go2 does not apply reset-time base force randomization.
    randomize_apply_external_force_torque = None

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_pace_hip = EventTerm(
        func=mdp.randomize_actuator_group_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            **UIKA_PACE_ACTUATOR_RANGES["hip"],
        },
    )

    randomize_pace_thigh = EventTerm(
        func=mdp.randomize_actuator_group_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
            **UIKA_PACE_ACTUATOR_RANGES["thigh"],
        },
    )

    randomize_pace_calf = EventTerm(
        func=mdp.randomize_actuator_group_parameters,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
            **UIKA_PACE_ACTUATOR_RANGES["calf"],
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-1, 1),
                "pitch": (-1, 1),
                "yaw": (-1, 1),
                # "x": (-3.5, -3.5),
                # "y": (-1.0, 1.0),
                # "z": (0.0, 0.0),
                # "roll": (-0.0, 0.0),
                # "pitch": (-0.0, 0.0),
                # "yaw": (-0.0, 0.0),
            },
            "velocity_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (-0.5, 0.5),
                # "roll": (-0.5, 0.5),
                # "pitch": (-0.5, 0.5),
                # "yaw": (-0.5, 0.5),
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )

    # interval randomization: keep the HimLoco disturbance schedule.
    external_force = EventTerm(
        func=mdp.apply_periodic_external_force_torque,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "period_step": 8,
            "force_range": (-30.0, 30.0),
            "torque_range": (-0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )

    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(16.0, 16.0),
        params={
            "velocity_range": {"x": (-1, 1), "y": (-1, 1)},
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0), heading=(-math.pi, math.pi)
        ),
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
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=UIKA_JOINT_NAMES,
        preserve_order=True,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

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
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100), noise=Unoise(n_min=-0.1, n_max=0.1))
        base_external_force = ObsTerm(
            func=mdp.base_external_force,
            params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
            clip=(-100, 100),
        )
        height_scanner = ObsTerm(
            func=mdp.height_scan_clip,
            scale=5.0,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-100, 100),
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )

        def __post_init__(self):
            self.enable_corruption = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # ---------------------------------------------------------------------
    # Episode state
    # ---------------------------------------------------------------------
    # Episode-level bookkeeping rewards. Termination is registered but disabled;
    # falls and illegal contacts are handled by termination terms and contact penalties.
    is_terminated = RewTerm(func=mdp.is_terminated, weight=0)

    # ---------------------------------------------------------------------
    # Base/root regularization
    # ---------------------------------------------------------------------
    # Shape body motion and posture while leaving planar/yaw tracking to command rewards.
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-0.0,
        params={
            "target_height": 0.33,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )

    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    upward = RewTerm(func=mdp.upward, weight=0.5)

    # ---------------------------------------------------------------------
    # Joint regularization
    # ---------------------------------------------------------------------
    # Keep joint motion conservative: limit torque/power, acceleration, soft-limit hits,
    # deviation from default posture, and diagonal leg asymmetry.
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.5e-5)
    joint_power = RewTerm(func=mdp.joint_power, weight=-2e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=0, params={"soft_ratio": 1.0})
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.3,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 16.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
                ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
            ],
            "use_default_offset": True,
        },
    )

    # ---------------------------------------------------------------------
    # Action smoothness
    # ---------------------------------------------------------------------
    # Penalize large step-to-step action changes.
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # ---------------------------------------------------------------------
    # Contact penalties
    # ---------------------------------------------------------------------
    # Penalize non-foot contacts and excessive foot impact forces.
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot).*"),
            "threshold": 1.0,
        },
    )
    contact_forces = RewTerm(
        func=mdp.contact_forces,
        # weight=-1.5e-4,
        weight=-1.5e-4,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "threshold": 100.0},
    )

    # ---------------------------------------------------------------------
    # Command tracking
    # ---------------------------------------------------------------------
    # Main task rewards: track commanded planar velocity and yaw rate.
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=3.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # ---------------------------------------------------------------------
    # Foot timing and gait
    # ---------------------------------------------------------------------
    # Shape foot timing, stance behavior, sliding, clearance, and diagonal trot rhythm.
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        # weight=0.1,
        weight=0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )

    feet_air_time_variance = RewTerm(
        func=mdp.feet_air_time_variance_penalty,
        weight=-1.0,
        # weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    prolonged_swing = RewTerm(
        func=mdp.prolonged_swing,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_swing_time": 0.60,
            "command_name": "base_velocity",
        },
    )

    feet_contact = RewTerm(
        func=mdp.feet_contact,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "expect_contact_num": 2,
        },
    )

    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.1,
        # weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
        },
    )

    feet_air_without_cmd = RewTerm(
        func=mdp.feet_air_without_cmd,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "contact_threshold": 1.0,
        },
    )

    feet_stumble = RewTerm(
        func=mdp.feet_stumble,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        # weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    feet_height = RewTerm(
        func=mdp.feet_height,
        weight=-0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "tanh_mult": 2.0,
            "target_height": 0.05,
            "command_name": "base_velocity",
        },
    )

    feet_height_body = RewTerm(
        func=mdp.feet_height_body,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.23,
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
        },
    )

    feet_lift_body = RewTerm(
        func=mdp.feet_lift_body,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "minimum_height": -0.30,
            "target_height": -0.10,
            "tanh_mult": 2.0,
            "command_name": "base_velocity",
        },
    )

    feet_gait = RewTerm(
        func=mdp.GaitReward,
        # weight=0.0,
        weight=0.5,
        params={
            "std": math.sqrt(0.5),
            "command_name": "base_velocity",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class ParkourRewardsCfg(RewardsCfg):
    """Reward terms used by the imported Extreme Parkour terrain task."""

    feet_edge = RewTerm(
        func=mdp.feet_edge,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "terrain_names": EXTREME_PARKOUR_TERRAIN_NAMES,
            "tile_size": PARKOUR_TILE_SIZE,
            "horizontal_scale": PARKOUR_HORIZONTAL_SCALE,
            "vertical_scale": 0.005,
            "height_threshold": 0.05,
            "edge_width": 0.05,
            "contact_threshold": 1.0,
            "terrain_level_threshold": None,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    command_levels_lin_vel = None
    command_levels_ang_vel = None


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for UIKA locomotion velocity-tracking environment."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def _configure_pace_actuators_for_play(self):
        self.events.randomize_pace_hip = None
        self.events.randomize_pace_thigh = None
        self.events.randomize_pace_calf = None
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 2
            actuator.max_delay = 2

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 20.0

        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.5

        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23

        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.base_height_scanner.update_period = self.sim.dt * self.decimation

        if self.scene.terrain.terrain_generator is None:
            self.curriculum.terrain_levels = None
            self.terminations.terrain_out_of_bounds = None
        elif getattr(self.curriculum, "terrain_levels", None) is not None:
            self.scene.terrain.terrain_generator.curriculum = True
        else:
            self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotParkourEnvCfg(RobotEnvCfg):
    """UIKA environment on imported Extreme Parkour terrains with waypoint commands."""

    scene: RobotParkourSceneCfg = RobotParkourSceneCfg(num_envs=4096, env_spacing=2.5)
    commands: ParkourCommandsCfg = ParkourCommandsCfg()
    rewards: ParkourRewardsCfg = ParkourRewardsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.scene.terrain.terrain_generator is not None:
            # Keep Isaac Lab's column-based terrain generation so the four columns stay fixed
            # to T_step, Slope, BridgeA, and BridgeB, but disable terrain-level promotion.
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 0


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_pace_actuators_for_play()
        self.scene.num_envs = 64

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_cols = 10
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 10
        else:
            self.curriculum.terrain_levels = None
            self.terminations.terrain_out_of_bounds = None

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges = mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0, 0),
        )


@configclass
class RobotParkourPlayEnvCfg(RobotParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_pace_actuators_for_play()
        self.scene.num_envs = 64

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.commands.base_velocity.speed_range = (0.45, 0.45)
        self.commands.base_velocity.debug_vis = True

import math

import isaaclab.sim as sim_utils
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
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp

UIKA_JOINT_NAMES = list(ROBOT_CFG.joint_sdk_names)
UIKA_LOWER_JOINT_POS_TARGET = {
    "FL_hip_joint": -0.40,
    "FL_thigh_joint": 0.40,
    "FL_calf_joint": 0.20,
    "FR_hip_joint": 0.40,
    "FR_thigh_joint": 0.40,
    "FR_calf_joint": 0.20,
    "RL_hip_joint": -0.40,
    "RL_thigh_joint": 0.40,
    "RL_calf_joint": 0.20,
    "RR_hip_joint": 0.40,
    "RR_thigh_joint": 0.40,
    "RR_calf_joint": 0.20,
}


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with UIKA robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
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

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                # "x": (-0.5, 0.5),
                # "y": (-0.5, 0.5),
                # "z": (0.0, 0.2),
                # "roll": (-3.14, 3.14),
                # "pitch": (-3.14, 3.14),
                # "yaw": (-3.14, 3.14),
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
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
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0)
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
        weight=-5.0,
        params={
            # "target_height": 0.3357,
            "target_height": 0.10,
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
        # weight=-2.0,
        weight=-0.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET,
        },
    )

    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.3,
        # weight=-0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 16.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET,
        },
    )

    joint_mirror = RewTerm(
        func=mdp.joint_mirror,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
                ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
            ],
            "target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET,
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
        weight=0.0,
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
    not_moving_when_commanded = RewTerm(
        func=mdp.not_moving_when_commanded,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "velocity_threshold": 0.15,
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
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
        # weight=-5.0,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": -0.23,
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
    # command_levels_lin_vel = CurrTerm(
    #     func=mdp.command_levels_lin_vel,
    #     params={
    #         "reward_term_name": "track_lin_vel_xy",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )
    # command_levels_ang_vel = CurrTerm(
    #     func=mdp.command_levels_ang_vel,
    #     params={
    #         "reward_term_name": "track_ang_vel_z",
    #         "range_multiplier": (0.1, 1.0),
    #     },
    # )


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
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
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

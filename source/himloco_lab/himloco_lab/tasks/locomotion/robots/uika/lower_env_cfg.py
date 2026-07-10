import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.robots.uika.velocity_env_cfg import (
    EventCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
    TerminationsCfg,
    UIKA_JOINT_NAMES,
)


UIKA_LOWER_JOINT_POS_TARGET = {
    "FL_hip_joint": -0.7,
    "FL_thigh_joint": 0.30,
    "FL_calf_joint": 0.20,
    "FR_hip_joint": 0.7,
    "FR_thigh_joint": 0.30,
    "FR_calf_joint": 0.20,
    "RL_hip_joint": -0.7,
    "RL_thigh_joint": 0.30,
    "RL_calf_joint": 0.20,
    "RR_hip_joint": 0.7,
    "RR_thigh_joint": 0.30,
    "RR_calf_joint": 0.20,
}


@configclass
class LowerRobotSceneCfg(RobotSceneCfg):
    """Flat-ground UIKA scene initialized in the lower crouch posture."""

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
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.25),
            joint_pos=UIKA_LOWER_JOINT_POS_TARGET,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class LowerEventCfg(EventCfg):
    """PACE randomization with a stable crouch reset pose."""

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )


@configclass
class LowerCommandsCfg:
    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class LowerActionsCfg:
    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=UIKA_JOINT_NAMES,
        preserve_order=True,
        scale={".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25},
        offset=UIKA_LOWER_JOINT_POS_TARGET,
        use_default_offset=False,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class LowerObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_to_target,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True),
                "target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET,
            },
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel_to_target,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True),
                "target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET,
            },
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

    critic: CriticCfg = CriticCfg()


@configclass
class LowerRewardsCfg(RewardsCfg):
    is_terminated = None
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    upward = RewTerm(func=mdp.upward, weight=0.0)

    joint_vel_l2 = None
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_vel_limits = None
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-2.0,
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
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 16.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET,
        },
    )
    joint_mirror = None

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot).*"),
            "threshold": 1.0,
        },
    )
    contact_forces = None

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    feet_air_time = None
    feet_air_time_variance = None
    feet_contact = None
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd,
        weight=0.0,
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
        },
    )
    single_foot_air_time = RewTerm(
        func=mdp.single_foot_air_time,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.25,
        },
    )
    feet_stumble = None
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_height = None
    feet_height_body = None
    feet_gait = None


@configclass
class LowerTerminationsCfg(TerminationsCfg):
    too_low = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"asset_cfg": SceneEntityCfg("robot"), "minimum_height": 0.15},
    )


@configclass
class LowerRobotEnvCfg(RobotEnvCfg):
    scene: LowerRobotSceneCfg = LowerRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: LowerObservationsCfg = LowerObservationsCfg()
    actions: LowerActionsCfg = LowerActionsCfg()
    commands: LowerCommandsCfg = LowerCommandsCfg()
    rewards: LowerRewardsCfg = LowerRewardsCfg()
    terminations: LowerTerminationsCfg = LowerTerminationsCfg()
    events: LowerEventCfg = LowerEventCfg()


@configclass
class LowerRobotPlayEnvCfg(LowerRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self._configure_pace_actuators_for_play()
        self.scene.num_envs = 64
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

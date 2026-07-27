"""Normal-standing UIKA rough-terrain task using the master lower reward recipe."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp
from himloco_lab.tasks.locomotion.robots.uika.velocity_env_cfg import (
    RewardsCfg,
    RobotEnvCfg,
    RobotPlayEnvCfg,
)


UIKA_NORMAL_JOINT_POS_TARGET = dict(ROBOT_CFG.init_state.joint_pos)


@configclass
class RoughNormalGaitRewardsCfg(RewardsCfg):
    """Master lower reward recipe adapted to the normal UIKA standing pose."""

    is_terminated = None
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.0,
        params={
            "target_height": 0.33,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "sensor_cfg": SceneEntityCfg("base_height_scanner"),
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )
    upward = RewTerm(func=mdp.upward, weight=0.25)

    joint_vel_l2 = None
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    joint_vel_limits = None
    stand_still = RewTerm(
        func=mdp.stand_still,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "target_joint_pos": UIKA_NORMAL_JOINT_POS_TARGET,
        },
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty,
        weight=-0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 16.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "target_joint_pos": UIKA_NORMAL_JOINT_POS_TARGET,
        },
    )
    joint_mirror = None

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
        weight=-1e-2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 100.0,
        },
    )

    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.2,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.6,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    feet_air_time = None
    feet_air_time_variance = None
    prolonged_swing = None
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
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_height = None
    feet_height_body = None
    feet_lift_body = None
    feet_gait = None


@configclass
class RoughNormalGaitEnvCfg(RobotEnvCfg):
    """Training configuration for the isolated normal-gait reward experiment."""

    rewards: RoughNormalGaitRewardsCfg = RoughNormalGaitRewardsCfg()


@configclass
class RoughNormalGaitPlayEnvCfg(RobotPlayEnvCfg):
    """Playback configuration for the isolated normal-gait reward experiment."""

    rewards: RoughNormalGaitRewardsCfg = RoughNormalGaitRewardsCfg()

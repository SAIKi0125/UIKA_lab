"""Configuration for UIKA robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from himloco_lab.assets.delayed_motor import DelayedDCMotorCfg

UIKA_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "uika")

_UIKA_MOTOR_LIMITS = {
    "hip": (17.0, 28.80),
    "thigh": (17.0, 28.80),
    "calf": (31.7, 15.43),
}


def _make_uika_actuator_cfg(joint_name: str, joint_type: str) -> DelayedDCMotorCfg:
    effort_limit, velocity_limit = _UIKA_MOTOR_LIMITS[joint_type]
    return DelayedDCMotorCfg(
        joint_names_expr=[joint_name],
        effort_limit=effort_limit,
        saturation_effort=effort_limit,
        velocity_limit=velocity_limit,
        stiffness=30.0,
        damping=1.5,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
        armature=0.0042,
        min_delay=2,
        max_delay=3,
    )


UIKA_ACTUATORS = {
    f"{leg}_{joint_type}": _make_uika_actuator_cfg(f"{leg}_{joint_type}_joint", joint_type)
    for joint_type in ("hip", "thigh", "calf")
    for leg in ("FL", "FR", "RL", "RR")
}


@configclass
class UIKAArticulationCfg(ArticulationCfg):
    """Configuration for UIKA articulation with SDK joint order."""
    joint_sdk_names: list[str] = None


@configclass
class UIKAUrdfFileCfg(sim_utils.UrdfFileCfg):
    fix_base: bool = False
    merge_fixed_joints: bool = False
    activate_contact_sensors: bool = True
    replace_cylinders_with_capsules = True
    joint_drive = sim_utils.UrdfConverterCfg.JointDriveCfg(
        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
    )
    articulation_props = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=True,
        solver_position_iteration_count=8,
        solver_velocity_iteration_count=4,
    )
    rigid_props = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        retain_accelerations=False,
        linear_damping=0.0,
        angular_damping=0.0,
        max_linear_velocity=1000.0,
        max_angular_velocity=1000.0,
        max_depenetration_velocity=1.0,
    )


UIKA_CFG = UIKAArticulationCfg(
    spawn=UIKAUrdfFileCfg(
        asset_path=f"{UIKA_ASSETS_DIR}/urdf/uika.urdf",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3357),
        joint_pos={
            "FL_hip_joint": -0.78,
            "FL_thigh_joint": 0.05,
            "FL_calf_joint": 0.70,
            "FR_hip_joint": 0.78,
            "FR_thigh_joint": 0.05,
            "FR_calf_joint": 0.70,
            "RL_hip_joint": -0.78,
            "RL_thigh_joint": 0.05,
            "RL_calf_joint": 0.70,
            "RR_hip_joint": 0.78,
            "RR_thigh_joint": 0.05,
            "RR_calf_joint": 0.70,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=UIKA_ACTUATORS,
    # fmt: off
    joint_sdk_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    # fmt: on
)

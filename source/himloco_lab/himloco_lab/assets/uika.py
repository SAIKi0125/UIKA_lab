"""Configuration for UIKA robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from himloco_lab.assets.pace_actuator_cfg import PaceDCMotorCfg

UIKA_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "uika")
UIKA_MOTOR_DELAY_STEPS = 6
UIKA_MOTOR_DELAY_SCALE_RANGE = (0.8, 1.2)


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
    actuators={
        "hip_thigh": PaceDCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
            effort_limit=17.0,
            saturation_effort=17.0,
            velocity_limit=28.80,
            stiffness=30.0,
            damping=1.0,
            armature=0.0042,
            friction=0.0,
            max_delay=UIKA_MOTOR_DELAY_STEPS,
            delay_scale_range=UIKA_MOTOR_DELAY_SCALE_RANGE,
        ),
        "calf": PaceDCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=31.7,
            saturation_effort=31.7,
            velocity_limit=15.43,
            stiffness=30.0,
            damping=1.0,
            armature=0.0042,
            friction=0.0,
            max_delay=UIKA_MOTOR_DELAY_STEPS,
            delay_scale_range=UIKA_MOTOR_DELAY_SCALE_RANGE,
        ),
    },
    # fmt: off
    joint_sdk_names=[
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ],
    # fmt: on
)

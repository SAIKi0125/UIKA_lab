"""Configuration for UIKA robot."""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from himloco_lab.assets.pace_actuator_cfg import PaceDCMotorCfg

UIKA_ASSETS_DIR = os.path.join(os.path.dirname(__file__), "uika")
UIKA_MOTOR_DELAY_STEPS = 6
UIKA_MOTOR_DELAY_SCALE_RANGE = (0.8, 1.2)
UIKA_HIP_THIGH_JOINT_NAMES = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
)
UIKA_CALF_JOINT_NAMES = (
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
)

UIKA_JOINT_ARMATURE = {
    "FL_hip_joint": 0.014251016,
    "FR_hip_joint": 0.014131420,
    "RL_hip_joint": 0.013336750,
    "RR_hip_joint": 0.012499551,
    "FL_thigh_joint": 0.014217399,
    "FR_thigh_joint": 0.013819096,
    "RL_thigh_joint": 0.012442659,
    "RR_thigh_joint": 0.013771501,
    "FL_calf_joint": 0.023925945,
    "FR_calf_joint": 0.024222745,
    "RL_calf_joint": 0.022859331,
    "RR_calf_joint": 0.022163186,
}

UIKA_JOINT_VISCOUS_DAMPING = {
    "FL_hip_joint": 0.002837807,
    "FR_hip_joint": 0.002691776,
    "RL_hip_joint": 0.000891626,
    "RR_hip_joint": 0.000871807,
    "FL_thigh_joint": 0.002336711,
    "FR_thigh_joint": 0.001674354,
    "RL_thigh_joint": 0.001915306,
    "RR_thigh_joint": 0.003362268,
    "FL_calf_joint": 0.001588821,
    "FR_calf_joint": 0.002290815,
    "RL_calf_joint": 0.001198500,
    "RR_calf_joint": 0.001509964,
}

UIKA_JOINT_FRICTION = {
    "FL_hip_joint": 0.018038273,
    "FR_hip_joint": 0.017919287,
    "RL_hip_joint": 0.005748361,
    "RR_hip_joint": 0.006325230,
    "FL_thigh_joint": 0.013646141,
    "FR_thigh_joint": 0.009873092,
    "RL_thigh_joint": 0.011157677,
    "RR_thigh_joint": 0.022234440,
    "FL_calf_joint": 0.010355398,
    "FR_calf_joint": 0.015262470,
    "RL_calf_joint": 0.007793665,
    "RR_calf_joint": 0.010796309,
}

UIKA_ENCODER_BIAS = {
    "FL_hip_joint": -0.093794197,
    "FR_hip_joint": 0.094102569,
    "RL_hip_joint": -0.002604164,
    "RR_hip_joint": 0.022023439,
    "FL_thigh_joint": -0.092052884,
    "FR_thigh_joint": -0.093451768,
    "RL_thigh_joint": -0.093989372,
    "RR_thigh_joint": -0.094484143,
    "FL_calf_joint": 0.035290889,
    "FR_calf_joint": -0.007711932,
    "RL_calf_joint": 0.048811503,
    "RR_calf_joint": 0.049209438,
}


def _select_joint_params(params: dict[str, float], joint_names: tuple[str, ...]) -> dict[str, float]:
    return {joint_name: params[joint_name] for joint_name in joint_names}


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
        asset_path=f"{UIKA_ASSETS_DIR}/urdf/uika_simple_collision.urdf",
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
            friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_HIP_THIGH_JOINT_NAMES),
            dynamic_friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_HIP_THIGH_JOINT_NAMES),
            viscous_friction=_select_joint_params(UIKA_JOINT_VISCOUS_DAMPING, UIKA_HIP_THIGH_JOINT_NAMES),
            armature=_select_joint_params(UIKA_JOINT_ARMATURE, UIKA_HIP_THIGH_JOINT_NAMES),
            encoder_bias=_select_joint_params(UIKA_ENCODER_BIAS, UIKA_HIP_THIGH_JOINT_NAMES),
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
            friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_CALF_JOINT_NAMES),
            dynamic_friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_CALF_JOINT_NAMES),
            viscous_friction=_select_joint_params(UIKA_JOINT_VISCOUS_DAMPING, UIKA_CALF_JOINT_NAMES),
            armature=_select_joint_params(UIKA_JOINT_ARMATURE, UIKA_CALF_JOINT_NAMES),
            encoder_bias=_select_joint_params(UIKA_ENCODER_BIAS, UIKA_CALF_JOINT_NAMES),
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

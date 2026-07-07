import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG
from himloco_lab.tasks.locomotion import mdp

UIKA_JOINT_NAMES = list(ROBOT_CFG.joint_sdk_names)
UIKA_HIP_JOINT_NAMES = [name for name in UIKA_JOINT_NAMES if "_hip_joint" in name]

TAKEOFF_CROUCH_JOINT_ANGLES = {
    "FL_hip_joint": -0.78,
    "FL_thigh_joint": 0.40,
    "FL_calf_joint": 0.20,
    "FR_hip_joint": 0.78,
    "FR_thigh_joint": 0.40,
    "FR_calf_joint": 0.20,
    "RL_hip_joint": -0.78,
    "RL_thigh_joint": -0.05,
    "RL_calf_joint": 0.20,
    "RR_hip_joint": 0.78,
    "RR_thigh_joint": -0.05,
    "RR_calf_joint": 0.20,
}


@configclass
class TakeoffSceneCfg(InteractiveSceneCfg):
    """Flat scene for UIKA fixed-point spring-jump training."""

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
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=2, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Reset, randomization, and spring-jump bookkeeping."""

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

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "yaw": (0.0, 0.0)},
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

    reset_to_crouch_pose = EventTerm(
        func=mdp.reset_to_joint_pose,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True),
            "joint_angles": TAKEOFF_CROUCH_JOINT_ANGLES,
        },
    )

    reset_spring_jump_state = EventTerm(
        func=mdp.spring_jump_state_reset,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot")},
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

    spring_jump_update = EventTerm(
        func=mdp.spring_jump_update_event,
        mode="interval",
        interval_range_s=(0.02, 0.02),
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
            "push_vel_z_range": (1.5, 2.2),
            "push_initial_prob": 0.8,
            "push_decay_steps": 1200,
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the fixed-point target jump."""

    base_velocity = mdp.SpringJumpCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        ranges=mdp.SpringJumpCommandCfg.Ranges(target_x=(0.8, 1.2), target_y=(0.0, 0.0)),
        setting_frame_range=(50, 60),
    )


@configclass
class ActionsCfg:
    """Action specifications for UIKA joints."""

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
    """Observation groups for adaptation-policy training."""

    @configclass
    class PolicyCfg(ObsGroup):
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
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

    @configclass
    class ActorHistoryCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.history_length = 10
            self.flatten_history_dim = True

    actor_history: ActorHistoryCfg = ActorHistoryCfg()

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, clip=(-100, 100))
        base_height = ObsTerm(func=mdp.base_height, params={"asset_cfg": SceneEntityCfg("robot")}, clip=(-100, 100))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100, 100))
        joint_pos_rel = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
            clip=(-100, 100),
        )
        joint_vel_rel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
            clip=(-100, 100),
        )
        last_action = ObsTerm(func=mdp.last_action, clip=(-100, 100))
        velocity_commands = ObsTerm(
            func=mdp.generated_commands, clip=(-100, 100), params={"command_name": "base_velocity"}
        )
        landing_xy_from_start = ObsTerm(
            func=mdp.landing_xy_from_start,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100, 100),
        )
        has_jumped = ObsTerm(
            func=mdp.has_jumped_obs,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            },
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
            self.history_length = 3
            self.flatten_history_dim = True

    critic: CriticCfg = CriticCfg()

    @configclass
    class PrivilegedTargetCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100, 100))
        landing_xy_from_start = ObsTerm(
            func=mdp.landing_xy_from_start,
            params={"asset_cfg": SceneEntityCfg("robot")},
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    privileged_target: PrivilegedTargetCfg = PrivilegedTargetCfg()


@configclass
class RewardsCfg:
    """Spring-jump reward terms ported to UIKA."""

    before_setting = RewTerm(
        func=mdp.before_setting,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_joint_angles": TAKEOFF_CROUCH_JOINT_ANGLES,
        },
    )
    line_z = RewTerm(
        func=mdp.line_z,
        weight=16.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    flight = RewTerm(
        func=mdp.flight,
        weight=2.0,
        params={"command_name": "base_velocity", "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    base_height_flight = RewTerm(
        func=mdp.base_height_flight,
        weight=3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_height": 0.50,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_height_stance_sj = RewTerm(
        func=mdp.base_height_stance_sj,
        weight=-10.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "stance_target": 0.3357,
            "setting_target": 0.24,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    orientation_jump = RewTerm(func=mdp.orientation_jump, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")})
    land_pos = RewTerm(
        func=mdp.land_pos,
        weight=25.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    tracking_lin_vel_jump = RewTerm(
        func=mdp.tracking_lin_vel_jump,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    line_vel_stance = RewTerm(
        func=mdp.line_vel_stance,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    foot_clearance_jump = RewTerm(
        func=mdp.foot_clearance_jump,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_z_body": -0.20,
        },
    )
    dof_pos_penalty_prepare_sj = RewTerm(
        func=mdp.dof_pos_penalty_prepare_sj,
        weight=-3.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True),
            "target_joint_angles": TAKEOFF_CROUCH_JOINT_ANGLES,
        },
    )
    dof_pos_penalty_sj = RewTerm(
        func=mdp.dof_pos_penalty_sj,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True),
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    dof_hip_pos_penalty_sj = RewTerm(
        func=mdp.dof_hip_pos_penalty_sj,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_HIP_JOINT_NAMES), "command_name": "base_velocity"},
    )
    ang_vel_xy = RewTerm(
        func=mdp.spring_jump_ang_vel_xy,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    torques = RewTerm(
        func=mdp.spring_jump_torques,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
    )
    dof_vel_limits = RewTerm(
        func=mdp.spring_jump_dof_vel_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
    )
    dof_vel = RewTerm(
        func=mdp.spring_jump_dof_vel,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True)},
    )
    collision = RewTerm(
        func=mdp.spring_jump_collision,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]), "threshold": 0.1},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    feet_contact_forces = RewTerm(
        func=mdp.spring_jump_feet_contact_forces,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), "max_contact_force": 150.0},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the fixed-point jump."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    too_low = DoneTerm(func=mdp.is_too_low, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.12})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class CurriculumCfg:
    """No terrain curriculum for the flat spring-jump task."""

    terrain_levels = None


@configclass
class TakeoffEnvCfg(ManagerBasedRLEnvCfg):
    """UIKA fixed-point takeoff environment."""

    scene: TakeoffSceneCfg = TakeoffSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5.0

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


@configclass
class TakeoffPlayEnvCfg(TakeoffEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.commands.base_velocity.ranges = mdp.SpringJumpCommandCfg.Ranges(target_x=(1.0, 1.0), target_y=(0.0, 0.0))
        self.commands.base_velocity.setting_frame_range = (50, 50)
        self.events.randomize_rigid_body_material = None
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_com_positions = None
        self.events.randomize_actuator_gains = None

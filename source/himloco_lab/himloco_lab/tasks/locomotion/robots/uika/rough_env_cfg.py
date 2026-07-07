import copy

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import himloco_lab.terrains as him_terrains
from himloco_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    UIKA_JOINT_NAMES,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
    TerminationsCfg,
)

ROUGH_PLAY_TERRAIN_LEVEL = 2

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
            step_height_range=(0.05, 0.23),
            step_width=0.30,
            platform_width=3.0,
            border_width=0.0,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
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


@configclass
class RoughRobotSceneCfg(RobotSceneCfg):
    """Configuration for UIKA rough terrain training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=COBBLESTONE_ROAD_CFG,
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
class RoughRewardsCfg(RewardsCfg):
    """Rough-terrain reward set, extending the shared UIKA rewards.

    Adds the stair-aware terms ported from rl-trained-robot-dog (agent_ppo):
    barrier-style orientation/height penalties, world-frame progress drive, a
    touchdown-event foot clearance, and anti-exploit penalties. All are registered
    with weight 0 here; effective rough weights are assigned in
    RoughRobotEnvCfg._configure_rewards_for_rough. These live only on the rough
    config, so the flat/base RewardsCfg is left untouched.
    """

    stair_orientation = RewTerm(
        func=mdp.stair_orientation,
        weight=0.0,
        params={"pitch_threshold": 0.28, "barrier_delta": 0.3},
    )
    stair_base_height = RewTerm(
        func=mdp.stair_base_height,
        weight=0.0,
        params={
            "flat_height": 0.3357,
            "stair_height": 0.50,
            "low_barrier_delta": 0.10,
            "high_barrier_delta": 0.25,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    distance_progress = RewTerm(func=mdp.distance_progress, weight=0.0)
    x_progress = RewTerm(
        func=mdp.x_progress,
        weight=0.0,
        params={"command_name": "base_velocity"},
    )
    forward_progress = RewTerm(
        func=mdp.forward_progress,
        weight=0.0,
        params={"command_name": "base_velocity", "tilt_gate_threshold": 0.5},
    )
    action_smoothness = RewTerm(func=mdp.smoothness, weight=0.0)
    stuck_penalty = RewTerm(
        func=mdp.stuck_penalty,
        weight=0.0,
        params={
            "speed_threshold": 0.08,
            "command_threshold": 0.15,
            "grace_time": 1.0,
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        },
    )
    prolonged_swing = RewTerm(
        func=mdp.prolonged_swing,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "max_swing_time": 0.65,
            "command_name": "base_velocity",
        },
    )
    leg_activity = RewTerm(
        func=mdp.leg_activity,
        weight=0.0,
        params={
            "sigma": 0.5,
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=UIKA_JOINT_NAMES, preserve_order=True),
        },
    )
    lateral_drift = RewTerm(
        func=mdp.lateral_drift,
        weight=0.0,
        params={"command_name": "base_velocity", "track_std": 0.35},
    )
    hip_abduction = RewTerm(
        func=mdp.hip_abduction,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=["FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"],
                preserve_order=True,
            ),
        },
    )
    foot_clearance_touchdown = RewTerm(
        func=mdp.foot_clearance_touchdown,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "command_name": "base_velocity",
            "target_height": 0.08,
            "min_air_time": 0.07,
            "max_air_time": 0.30,
            "min_command": 0.10,
            "progress_speed": 0.25,
            "min_step_length": 0.05,
        },
    )

    # --- Override shared base terms with reference-faithful (no upright_gate)
    # variants. The shared/flat RewardsCfg terms keep the gate; only rough uses these. ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_rough,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.3},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_rough,
        weight=0.4,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2_rough, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2_rough, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2_rough, weight=0.0)
    joint_power = RewTerm(func=mdp.joint_power_rough, weight=0.0)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts_rough,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="^(?!.*_foot).*"),
            "threshold": 1.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide_rough,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble_rough,
        weight=0.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    )
    joint_pos_penalty = RewTerm(
        func=mdp.joint_pos_penalty_rough,
        weight=0.0,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stand_still_scale": 2.0,
            "velocity_threshold": 0.1,
        },
    )


@configclass
class RoughRewardsTerminationsCfg(TerminationsCfg):
    """Rough terminations: add a fall termination on illegal base contact.

    The shared config only has time_out / terrain_out_of_bounds, so a fallen robot
    keeps lying until the 20 s episode ends. That lets a swing foot's air_time grow
    unbounded and made prolonged_swing explode (~-260). Terminating on base contact
    resets fallen envs promptly and pairs with the is_terminated=-2.0 penalty.
    """

    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
            "threshold": 1.0,
        },
    )


@configclass
class RoughRobotEnvCfg(RobotEnvCfg):
    """Configuration for UIKA rough terrain velocity-tracking environment."""

    scene: RoughRobotSceneCfg = RoughRobotSceneCfg(num_envs=4096, env_spacing=2.5)
    rewards: RoughRewardsCfg = RoughRewardsCfg()
    terminations: RoughRewardsTerminationsCfg = RoughRewardsTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges.heading = None
        self._configure_rewards_for_rough()

    def _configure_rewards_for_rough(self):
        # Weights ported from rl-trained-robot-dog / agent_ppo
        # (train_env_conf_standard_locomotion.toml). The design softens the
        # stability penalties that fight stair/rough dynamics, replaces symmetric
        # quadratic penalties with barrier functions, adds world-frame progress
        # drive, and uses a touchdown-event foot clearance plus anti-exploit terms.

        # --- Episode state ---
        # High penalty to prevent "early-death exploitation" (ref termination=-2.0).
        self.rewards.is_terminated.weight = -2.0

        # --- Base/root regularization (softened for stair/rough dynamics) ---
        # lin_vel_z softened -2.0 -> -0.4: stair climbing produces upward velocity.
        self.rewards.lin_vel_z_l2.weight = -0.4
        # ang_vel_xy softened to -0.08: preserve natural pitch/roll needed on stairs.
        self.rewards.ang_vel_xy_l2.weight = -0.08
        # Continuous posture penalty (no deadzone), complements stair_orientation barrier.
        self.rewards.flat_orientation_l2.weight = -0.2

        # Barrier-style tilt penalty (Kim et al., 2025): safe zone allows forward lean.
        self.rewards.stair_orientation.weight = -0.5
        # Asymmetric barrier base height: hard on crawling, soft on rising.
        # flat_height uses UIKA's physical base height 0.3357 (robot-specific;
        # the reference's 0.30 was its own robot's height). low_barrier_delta=0.10 per ref.
        self.rewards.stair_base_height.weight = -0.15
        self.rewards.stair_base_height.params["flat_height"] = 0.3357
        self.rewards.stair_base_height.params["low_barrier_delta"] = 0.10

        # --- Joint regularization ---
        self.rewards.joint_torques_l2.weight = -5e-5
        # energy_biomechanical analog: absolute joint mechanical power (ref -1.5e-3).
        self.rewards.joint_power.weight = -1.5e-3
        # dof_pos_limits: loosened to allow large thigh/calf angles for clearing edges.
        self.rewards.joint_pos_limits.weight = -3.0
        # joint_pos_penalty reduced -0.3 -> -0.01: large deviations are needed on stairs.
        # (func/params set in RoughRewardsCfg; rough variant has no command_threshold.)
        self.rewards.joint_pos_penalty.weight = -0.01

        # Hip abduction + L/R asymmetry penalty (ref -1.0): targets splay/drift gaits.
        self.rewards.hip_abduction.weight = -1.0

        # --- Action smoothness ---
        self.rewards.action_rate_l2.weight = -0.015
        # 2nd-order action difference / jerk penalty (ref action_smoothness=-0.005).
        self.rewards.action_smoothness.weight = -0.005

        # --- Contact penalties ---
        self.rewards.undesired_contacts.weight = -0.5
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = "^(?!.*_foot).*"

        # --- Command tracking + progress drive ---
        self.rewards.track_lin_vel_xy.weight = 1.0
        self.rewards.track_lin_vel_xy.params["std"] = 0.3
        self.rewards.track_ang_vel_z.weight = 0.4
        self.rewards.track_ang_vel_z.params["std"] = 0.25
        # World-frame radial progress (anti-spin), command-direction progress
        # (anti-corner-cut), and body-frame forward drive on tilt.
        self.rewards.distance_progress.weight = 0.6
        self.rewards.x_progress.weight = 0.6
        self.rewards.forward_progress.weight = 0.6
        self.rewards.stuck_penalty.weight = -0.8
        # Penalize a nearly dead leg during commanded motion.
        self.rewards.leg_activity.weight = -0.25
        # Penalize lateral drift (perpendicular to command) while tracking forward.
        self.rewards.lateral_drift.weight = -1.0

        # --- Foot timing and gait ---
        # feet_stumble: penalize feet hitting stair edges / vertical faces (ref -0.4).
        self.rewards.feet_stumble.weight = -0.4
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = ".*_foot"

        self.rewards.feet_slide.weight = -0.08
        self.rewards.feet_slide.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.feet_slide.params["asset_cfg"].body_names = ".*_foot"

        # Touchdown-event foot clearance (ref foot_clearance=1.5): anti-tripod.
        self.rewards.foot_clearance_touchdown.weight = 1.5
        # Penalize feet airborne beyond max_swing_time while moving (ref -1.5).
        self.rewards.prolonged_swing.weight = -1.5

        # --- Disable all terms not used on rough (remove from the manager entirely
        # so they are not computed each step). These are either inherited base/flat
        # terms superseded by the ported set, or terms absent from the reference. ---
        self.rewards.base_height_l2 = None  # superseded by stair_base_height
        self.rewards.body_lin_acc_l2 = None
        self.rewards.upward = None
        self.rewards.joint_vel_l2 = None
        self.rewards.joint_acc_l2 = None  # ref joint_acc=0 (raw magnitude dominates)
        self.rewards.joint_vel_limits = None
        self.rewards.stand_still = None  # not in reference; handled by stuck_penalty
        self.rewards.joint_mirror = None
        self.rewards.contact_forces = None
        self.rewards.not_moving_when_commanded = None  # replaced by stuck_penalty
        self.rewards.feet_air_time = None  # pure air-time reward encourages tripod hacks
        self.rewards.feet_air_time_variance = None
        self.rewards.feet_contact = None
        self.rewards.feet_contact_without_cmd = None  # not in reference
        self.rewards.feet_height = None  # superseded by foot_clearance_touchdown
        self.rewards.feet_height_body = None  # superseded by foot_clearance_touchdown
        self.rewards.feet_gait = None  # superseded by foot_clearance_touchdown


@configclass
class RoughRobotPlayEnvCfg(RoughRobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64

        play_terrain_cfg = copy.deepcopy(COBBLESTONE_ROAD_CFG)
        play_terrain_cfg.num_rows = 1
        play_terrain_cfg.num_cols = 10
        play_terrain_cfg.curriculum = True
        play_terrain_cfg.difficulty_range = (0.2, 0.3)
        self.scene.terrain.terrain_generator = play_terrain_cfg
        self.scene.terrain.max_init_terrain_level = 0

        self.curriculum.terrain_levels = None
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.ranges = mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-0, 0),
        )

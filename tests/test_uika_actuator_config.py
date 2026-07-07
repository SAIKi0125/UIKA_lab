import xml.etree.ElementTree as ET
from pathlib import Path


def test_uika_actuators_use_delayed_dc_motor_with_may31_constants():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "uika.py"
    ).read_text()

    assert "UIKA_JOINT_ARMATURE" not in source
    assert "UIKA_JOINT_VISCOUS_DAMPING" not in source
    assert "UIKA_JOINT_FRICTION" not in source
    assert "UIKA_ENCODER_BIAS" not in source
    assert "PaceDCMotorCfg" not in source
    assert "DelayedDCMotorCfg" in source
    assert "min_delay=5" in source
    assert "max_delay=7" in source
    assert "delay_scale_range" not in source
    assert "damping=1.5" in source
    assert "armature=0.0042" in source
    assert "friction=0.0" in source


def test_delayed_dc_motor_keeps_dc_motor_semantics_and_delays_commands():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "delayed_motor.py"
    ).read_text()
    assert "class DelayedDCMotor(DCMotor)" in source
    assert "class DelayedDCMotorCfg(DCMotorCfg)" in source
    assert "class_type: type = DelayedDCMotor" in source
    assert "torques_delay_buffer" not in source
    assert "positions_delay_buffer.compute(control_action.joint_positions)" in source
    assert "velocities_delay_buffer.compute(control_action.joint_velocities)" in source
    assert "efforts_delay_buffer.compute(control_action.joint_efforts)" in source


def test_uika_training_command_samples_x_velocity_and_yaw_only():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "velocity_env_cfg.py"
    ).read_text()

    assert "lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.0, 0.0), ang_vel_z=(-1.0, 1.0)" in source
    assert "heading_command=False" in source
    assert "rel_heading_envs=0.0" in source
    assert "heading=(-math.pi, math.pi)" not in source


def test_uika_flat_and_rough_tasks_are_registered_separately():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "__init__.py"
    ).read_text()

    assert 'id="UIKA-Velocity"' in source
    assert '"env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg"' in source
    assert 'id="UIKA-Velocity-Rough"' in source
    assert '"env_cfg_entry_point": f"{__name__}.rough_env_cfg:RoughRobotEnvCfg"' in source
    assert "UIKA-Velocity-Rough-Reward" not in source


def test_uika_flat_training_uses_plane_without_terrain_curriculum():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "velocity_env_cfg.py"
    ).read_text()

    assert "class RobotSceneCfg" in source
    assert 'terrain_type="plane"' in source
    assert "class RobotEnvCfg" in source
    assert "self.curriculum.terrain_levels = None" in source


def test_uika_rough_training_uses_generated_terrain_curriculum_and_disables_heading():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "rough_env_cfg.py"
    ).read_text()

    assert "class RoughRobotSceneCfg" in source
    assert "class RoughRobotEnvCfg" in source
    assert 'terrain_type="generator"' in source
    assert "terrain_generator=COBBLESTONE_ROAD_CFG" in source
    assert "max_init_terrain_level=5" in source
    assert "class RoughRobotEnvCfg(RobotEnvCfg)" in source
    assert "super().__post_init__()" in source
    assert "self.commands.base_velocity.heading_command = False" in source
    assert "self.commands.base_velocity.rel_heading_envs = 0.0" in source
    assert "self.commands.base_velocity.ranges.heading = None" in source


def test_uika_velocity_cfg_keeps_rough_config_in_separate_file():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "velocity_env_cfg.py"
    ).read_text()

    assert "class RoughRobotSceneCfg" not in source
    assert "class RoughRobotEnvCfg" not in source
    assert "class RoughRobotPlayEnvCfg" not in source


def test_uika_rough_rewards_are_configured_in_rough_cfg():
    velocity_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "velocity_env_cfg.py"
    ).read_text()
    rough_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "rough_env_cfg.py"
    ).read_text()

    assert "flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0)" in velocity_source
    assert "upward = RewTerm(func=mdp.upward, weight=0.25)" in velocity_source
    assert "self.rewards.flat_orientation_l2.weight = -0.2" in rough_source
    assert "self.rewards.base_height_l2.weight = -1.0" in rough_source
    assert 'self.rewards.base_height_l2.params["target_height"] = 0.3357' in rough_source
    assert "self.rewards.feet_height_body.weight = -0.01" in rough_source
    assert "class RoughRewardRobotEnvCfg" not in rough_source


def test_uika_rough_cfg_exposes_all_reward_weights_for_tuning():
    velocity_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "velocity_env_cfg.py"
    ).read_text()
    rough_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "rough_env_cfg.py"
    ).read_text()

    reward_names = [
        "is_terminated",
        "lin_vel_z_l2",
        "ang_vel_xy_l2",
        "flat_orientation_l2",
        "base_height_l2",
        "body_lin_acc_l2",
        "upward",
        "joint_torques_l2",
        "joint_power",
        "joint_vel_l2",
        "joint_acc_l2",
        "joint_pos_limits",
        "joint_vel_limits",
        "stand_still",
        "joint_pos_penalty",
        "joint_mirror",
        "action_rate_l2",
        "undesired_contacts",
        "contact_forces",
        "track_lin_vel_xy",
        "track_ang_vel_z",
        "not_moving_when_commanded",
        "feet_air_time",
        "feet_air_time_variance",
        "feet_contact",
        "feet_contact_without_cmd",
        "feet_stumble",
        "feet_slide",
        "feet_height",
        "feet_height_body",
        "feet_gait",
    ]

    assert "def _configure_rewards_for_rough(self):" in rough_source
    for reward_name in reward_names:
        assert f"self.rewards.{reward_name}.weight =" in rough_source
    assert "not_moving_when_commanded = RewTerm(" in velocity_source
    assert "func=mdp.not_moving_when_commanded" in velocity_source
    assert 'self.rewards.not_moving_when_commanded.params["velocity_threshold"] = 0.15' in rough_source


def test_uika_rough_play_uses_level3_terrain_without_changing_training():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "rough_env_cfg.py"
    ).read_text()

    assert "max_init_terrain_level=5" in source
    assert "ROUGH_PLAY_TERRAIN_LEVEL = 2" in source
    assert "play_terrain_cfg.num_rows = 1" in source
    assert "play_terrain_cfg.difficulty_range = (0.2, 0.3)" in source
    assert "self.scene.terrain.max_init_terrain_level = 0" in source
    assert "self.curriculum.terrain_levels = None" in source


def test_uika_velocity_default_pose_rewards_target_lower_pose():
    velocity_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "uika"
        / "velocity_env_cfg.py"
    ).read_text()
    rewards_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "rewards.py"
    ).read_text()

    expected_targets = {
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

    assert "UIKA_LOWER_JOINT_POS_TARGET = {" in velocity_source
    for joint_name, target in expected_targets.items():
        assert f'"{joint_name}": {target}' in velocity_source
    assert velocity_source.count('"target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET') == 3
    assert "target_joint_pos: dict[str, float] | None = None" in rewards_source
    assert "asset.data.default_joint_pos" in rewards_source
    assert "target_joint_pos is not None" in rewards_source
    assert "reward = mdp.joint_deviation_l1(env, asset_cfg)" not in rewards_source


def test_uika_uses_original_collision_urdf():
    repo_root = Path(__file__).resolve().parents[1]
    uika_source = (repo_root / "source" / "himloco_lab" / "himloco_lab" / "assets" / "uika.py").read_text()
    urdf_path = (
        repo_root
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "uika"
        / "urdf"
        / "uika.urdf"
    )

    assert "uika.urdf" in uika_source
    assert "uika_simple_collision.urdf" not in uika_source

    root = ET.parse(urdf_path).getroot()
    collision_geometries = [collision.find("geometry") for collision in root.findall(".//collision")]
    assert collision_geometries
    assert all(geometry.find("mesh") is not None for geometry in collision_geometries)

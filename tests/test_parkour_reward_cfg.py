import ast
from pathlib import Path
import xml.etree.ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_uika_uses_simple_primitive_collision_urdf():
    asset_cfg = (
        REPO_ROOT
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "uika.py"
    ).read_text()
    assert 'asset_path=f"{UIKA_ASSETS_DIR}/urdf/uika_simple_collision.urdf"' in asset_cfg

    urdf_path = (
        REPO_ROOT
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "uika"
        / "urdf"
        / "uika_simple_collision.urdf"
    )
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    collisions = root.findall(".//collision")
    assert len(collisions) == 17

    expected_primitives = {
        "base": "box",
        "FL_hip": "box",
        "FR_hip": "box",
        "RL_hip": "box",
        "RR_hip": "box",
        "FL_thigh": "box",
        "FR_thigh": "box",
        "RL_thigh": "box",
        "RR_thigh": "box",
        "FL_calf": "box",
        "FR_calf": "box",
        "RL_calf": "box",
        "RR_calf": "box",
        "FL_foot": "sphere",
        "FR_foot": "sphere",
        "RL_foot": "sphere",
        "RR_foot": "sphere",
    }

    for link in root.findall(".//link"):
        collision = link.find("collision")
        assert collision is not None
        geometry = collision.find("geometry")
        assert geometry is not None
        assert geometry.find("mesh") is None
        primitive_tags = [child.tag for child in geometry]
        assert len(primitive_tags) == 1
        assert primitive_tags[0] == expected_primitives[link.attrib["name"]]

        if link.attrib["name"].endswith("_thigh"):
            size = [float(value) for value in geometry.find("box").attrib["size"].split()]
            assert max(size[1], size[2]) <= 0.11

        if link.attrib["name"].endswith(("_thigh", "_calf")):
            rpy = [float(value) for value in collision.find("origin").attrib["rpy"].split()]
            assert any(abs(value) > 1.0e-3 for value in rpy)


def test_uika_base_command_uses_robotlab_style_velocity_curriculum_without_legacy_fields():
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
    commands_cfg = source.split("class CommandsCfg", maxsplit=1)[1].split(
        "@configclass\nclass ActionsCfg", maxsplit=1
    )[0]
    curriculum_cfg = source.split("class CurriculumCfg", maxsplit=1)[1].split(
        "@configclass\nclass RobotEnvCfg", maxsplit=1
    )[0]

    expected_command_snippets = [
        "base_velocity = mdp.UniformLevelVelocityCommandCfg(",
        "ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(",
        "lin_vel_x=(-2.0, 2.0)",
        "lin_vel_y=(-1.0, 1.0)",
        "ang_vel_z=(-1.0, 1.0)",
        "heading=None",
    ]

    for snippet in expected_command_snippets:
        assert snippet in commands_cfg

    legacy_command_fields = [
        "curriculums_limit_ranges",
        "low_vel_env_lin_x_ranges",
        "rel_high_vel_envs",
        "min_command_norm",
    ]
    for snippet in legacy_command_fields:
        assert snippet not in commands_cfg

    assert "UniformThresholdVelocityCommandCfg" not in commands_cfg
    assert "lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)" not in curriculum_cfg
    assert "command_levels_lin_vel = CurrTerm(" in curriculum_cfg
    assert "func=mdp.command_levels_lin_vel" in curriculum_cfg
    assert '"reward_term_name": "track_lin_vel_xy"' in curriculum_cfg
    assert '"range_multiplier": (0.1, 1.0)' in curriculum_cfg
    assert "command_levels_ang_vel = CurrTerm(" in curriculum_cfg
    assert "func=mdp.command_levels_ang_vel" in curriculum_cfg
    assert '"reward_term_name": "track_ang_vel_z"' in curriculum_cfg
    assert '"range_multiplier": (0.1, 1.0)' in curriculum_cfg
    assert "command_levels_lin_vel = None" not in curriculum_cfg
    assert "command_levels_ang_vel = None" not in curriculum_cfg


def test_uniform_level_velocity_command_legacy_fields_are_optional_for_uika():
    cfg_source = (
        REPO_ROOT
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "commands"
        / "commands_cfg.py"
    ).read_text()
    command_source = (
        REPO_ROOT
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "commands"
        / "commands.py"
    ).read_text()

    assert "curriculums_limit_ranges: tuple[float, float] | None = None" in cfg_source
    assert "low_vel_env_lin_x_ranges: tuple[float, float] | None = None" in cfg_source
    assert "rel_high_vel_envs: float | None = None" in cfg_source
    assert "min_command_norm: float | None = None" in cfg_source
    assert "lin_vel_x_range = self.cfg.low_vel_env_lin_x_ranges or self.cfg.ranges.lin_vel_x" in command_source
    assert "if self.cfg.rel_high_vel_envs is not None:" in command_source
    assert "if self.cfg.min_command_norm is not None:" in command_source


def test_uika_robotlab_linear_velocity_curriculum_updates_lateral_range():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "curriculums.py"
    ).read_text()
    lin_vel_curriculum = source.split("def command_levels_lin_vel", maxsplit=1)[1].split(
        "\ndef command_levels_ang_vel", maxsplit=1
    )[0]

    assert "base_velocity_ranges.lin_vel_y = new_vel_y.tolist()" in lin_vel_curriculum
    assert "base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()" in lin_vel_curriculum
    assert "lateral curriculum disabled" not in lin_vel_curriculum


def test_uika_play_runs_on_plane_without_command_curriculum():
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
    play_cfg = source.split("class RobotPlayEnvCfg", maxsplit=1)[1]
    pre_super_cfg = play_cfg.split("super().__post_init__()", maxsplit=1)[0]

    assert 'self.scene.terrain.terrain_type = "plane"' in pre_super_cfg
    assert "self.scene.terrain.terrain_generator = None" in pre_super_cfg
    assert "self.curriculum.terrain_levels = None" in play_cfg
    assert "self.curriculum.command_levels_lin_vel = None" in play_cfg
    assert "self.curriculum.command_levels_ang_vel = None" in play_cfg
    assert "lin_vel_x=(0.3, 0.3)" in play_cfg


def test_uika_uses_identified_pace_motor_parameters_with_unified_delay():
    source = (REPO_ROOT / "source" / "himloco_lab" / "himloco_lab" / "assets" / "uika.py").read_text()
    module = ast.parse(source)
    assignments = {}
    for node in module.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        try:
            assignments[node.targets[0].id] = ast.literal_eval(node.value)
        except ValueError:
            continue

    expected_snippets = [
        "from himloco_lab.assets.pace_actuator_cfg import PaceDCMotorCfg",
        "UIKA_MOTOR_DELAY_STEPS = 6",
        "UIKA_MOTOR_DELAY_SCALE_RANGE = (0.8, 1.2)",
        "UIKA_HIP_THIGH_JOINT_NAMES = (",
        "UIKA_CALF_JOINT_NAMES = (",
        "hip_thigh\": PaceDCMotorCfg(",
        "calf\": PaceDCMotorCfg(",
        "max_delay=UIKA_MOTOR_DELAY_STEPS",
        "delay_scale_range=UIKA_MOTOR_DELAY_SCALE_RANGE",
    ]

    for snippet in expected_snippets:
        assert snippet in source

    assert assignments["UIKA_JOINT_ARMATURE"] == {
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
    assert assignments["UIKA_JOINT_VISCOUS_DAMPING"] == {
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
    assert assignments["UIKA_JOINT_FRICTION"] == {
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
    assert assignments["UIKA_ENCODER_BIAS"] == {
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
    assert "from isaaclab.actuators import DCMotorCfg" not in source
    assert '"hip_thigh": DCMotorCfg(' not in source
    assert '"calf": DCMotorCfg(' not in source
    assert "friction=UIKA_JOINT_FRICTION" not in source
    assert "dynamic_friction=UIKA_JOINT_FRICTION" not in source
    assert "viscous_friction=UIKA_JOINT_VISCOUS_DAMPING" not in source
    assert "armature=UIKA_JOINT_ARMATURE" not in source
    assert "encoder_bias=UIKA_ENCODER_BIAS" not in source
    assert "friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_HIP_THIGH_JOINT_NAMES)" in source
    assert "friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_CALF_JOINT_NAMES)" in source
    assert "dynamic_friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_HIP_THIGH_JOINT_NAMES)" in source
    assert "dynamic_friction=_select_joint_params(UIKA_JOINT_FRICTION, UIKA_CALF_JOINT_NAMES)" in source
    assert (
        "viscous_friction=_select_joint_params(UIKA_JOINT_VISCOUS_DAMPING, UIKA_HIP_THIGH_JOINT_NAMES)"
        in source
    )
    assert "viscous_friction=_select_joint_params(UIKA_JOINT_VISCOUS_DAMPING, UIKA_CALF_JOINT_NAMES)" in source
    assert "armature=_select_joint_params(UIKA_JOINT_ARMATURE, UIKA_HIP_THIGH_JOINT_NAMES)" in source
    assert "armature=_select_joint_params(UIKA_JOINT_ARMATURE, UIKA_CALF_JOINT_NAMES)" in source
    assert "encoder_bias=_select_joint_params(UIKA_ENCODER_BIAS, UIKA_HIP_THIGH_JOINT_NAMES)" in source
    assert "encoder_bias=_select_joint_params(UIKA_ENCODER_BIAS, UIKA_CALF_JOINT_NAMES)" in source
    assert "UIKA_CALF_MOTOR_DELAY_STEPS" not in source
    assert "UIKA_HIP_THIGH_MOTOR_DELAY_STEPS" not in source


def test_pace_actuator_randomizes_delay_by_scale_range_on_reset():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "pace_actuator.py"
    ).read_text()
    cfg_source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "pace_actuator_cfg.py"
    ).read_text()

    assert "delay_scale_range: tuple[float, float] = (1.0, 1.0)" in cfg_source
    assert "self.delay_scale_range = cfg.delay_scale_range" in source
    assert "self.delay_buffer_max = int(math.ceil(self.nominal_delay * self.delay_scale_range[1]))" in source
    assert "self.torques_delay_buffer = DelayBuffer(self.delay_buffer_max + 1" in source
    assert "self._sample_time_lags(torch.arange(self._num_envs, device=self._device))" in source
    assert "self._sample_time_lags(env_ids)" in source
    assert "torch.empty(len(env_ids), device=self._device).uniform_(" in source
    assert "torch.round(self.nominal_delay * scales).to(dtype=torch.int)" in source


def test_uika_training_and_play_run_at_200hz():
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
    post_init = source.split("class RobotEnvCfg", maxsplit=1)[1].split(
        "@configclass\nclass RobotPlayEnvCfg", maxsplit=1
    )[0]

    assert "self.decimation = 2" in post_init
    assert "self.sim.dt = 0.0025" in post_init
    assert "self.sim.render_interval = self.decimation" in post_init


def test_uika_runner_uses_requested_short_rollout():
    source = (REPO_ROOT / "source" / "himloco_lab" / "himloco_lab" / "tasks" / "locomotion" / "agents" / "himloco_rsl_rl_cfg.py").read_text()
    module = ast.parse(source)
    uika_runner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "UIKAPPORunnerCfg")
    rollout = next(
        ast.literal_eval(node.value)
        for node in uika_runner.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "num_steps_per_env"
    )

    assert rollout == 24


def test_deploy_export_includes_robotlab_velocity_scales():
    source = (REPO_ROOT / "source" / "himloco_lab" / "himloco_lab" / "utils" / "export_deploy_cfg.py").read_text()

    assert 'cfg["lin_vel_scale"] = _obs_term_scale(env, "critic", "base_lin_vel", 2.0)' in source
    assert 'cfg["ang_vel_scale"] = _obs_term_scale(env, "policy", "base_ang_vel", 0.25)' in source


def test_uika_base_rewards_match_go2_reward_set():
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
    rewards_cfg = source.split("class RewardsCfg", maxsplit=1)[1].split(
        "@configclass\nclass TerminationsCfg", maxsplit=1
    )[0]

    expected_snippets = [
        "track_lin_vel_xy = RewTerm(",
        "func=mdp.track_lin_vel_xy_exp",
        "weight=1.0",
        "track_ang_vel_z = RewTerm(",
        "func=mdp.track_ang_vel_z_exp",
        "weight=0.5",
        "base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)",
        "base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)",
        "flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)",
        "joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)",
        "energy = RewTerm(func=mdp.energy, weight=-2e-5)",
        "base_height_l2 = RewTerm(",
        "func=mdp.base_height",
        "weight=-1.0",
        '"target_height": 0.3',
        "feet_height_body = RewTerm(",
        "func=mdp.feet_height_body",
        "weight=-0.01",
        '"target_height": -0.2',
        '"tanh_mult": 2.0',
        "action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)",
        "smoothness = RewTerm(func=mdp.smoothness, weight=-0.01)",
    ]
    removed_uika_specific_terms = [
        "is_terminated = RewTerm(",
        "upward = RewTerm(",
        "joint_torques_l2 = RewTerm(",
        "joint_power = RewTerm(",
        "joint_pos_limits = RewTerm(",
        "stand_still = RewTerm(",
        "joint_pos_penalty = RewTerm(",
        "undesired_contacts = RewTerm(",
        "feet_slide = RewTerm(",
        "feet_contact_without_cmd = RewTerm(",
        "feet_gait = RewTerm(",
    ]

    for snippet in expected_snippets:
        assert snippet in rewards_cfg
    for snippet in removed_uika_specific_terms:
        assert snippet not in rewards_cfg


def test_parkour_rewards_cfg_uses_source_parkour_regularizers_with_velocity_tracking():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "parkour"
        / "velocity_env_cfg.py"
    ).read_text()
    parkour_cfg = source.split("class ParkourRewardsCfg", maxsplit=1)[1].split(
        "@configclass\nclass TerminationsCfg", maxsplit=1
    )[0]

    expected_snippets = [
        "base_linear_velocity = None",
        "base_angular_velocity = None",
        "joint_acc = None",
        "energy = None",
        "action_rate = None",
        "smoothness = None",
        "reward_collision = RewTerm(",
        "weight=-10.0",
        '"threshold": 0.1',
        'body_names=["base", ".*_calf", ".*_thigh"]',
        "feet_stumble = None",
        "reward_feet_edge = RewTerm(",
        "contact_threshold\": 2.0",
        "terrain_level_threshold\": None",
        "reward_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)",
        "reward_dof_error = RewTerm(",
        "func=mdp.joint_dof_error_l2",
        "weight=-0.04",
        "reward_hip_pos = RewTerm(",
        "func=mdp.hip_pos_l2",
        "weight=-0.5",
        "reward_action_rate = RewTerm(",
        "func=mdp.ActionRateNorm",
        "weight=-0.1",
        '"action_term_name": "JointPositionAction"',
        "reward_dof_acc = RewTerm(",
        "func=mdp.JointDofAccL2",
        "weight=-2.5e-7",
        "reward_lin_vel_z = RewTerm(",
        "func=mdp.source_lin_vel_z_l2",
        "weight=-1.0",
        "reward_orientation = RewTerm(",
        "func=mdp.source_flat_orientation_l2",
        "track_lin_vel_xy = RewTerm(",
        "func=mdp.track_lin_vel_xy_exp",
        "weight=1.5",
        "track_ang_vel_z = RewTerm(",
        "func=mdp.track_ang_vel_z_exp",
        "weight=0.5",
        "reward_delta_torques = RewTerm(",
        "func=mdp.DeltaTorquesL2",
        "weight=-1.0e-7",
        "reward_feet_stumble = RewTerm(",
        "weight=-1.0",
    ]

    for snippet in expected_snippets:
        assert snippet in parkour_cfg

    assert "reward_tracking_goal_vel" not in parkour_cfg
    assert "track_goal_vel_from_command" not in parkour_cfg


def test_parkour_env_disables_inherited_himloco_velocity_curriculum():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "robots"
        / "parkour"
        / "velocity_env_cfg.py"
    ).read_text()
    parkour_env_cfg = source.split("class RobotParkourEnvCfg", maxsplit=1)[1].split(
        "@configclass\nclass RobotPlayEnvCfg", maxsplit=1
    )[0]

    assert "self.curriculum.terrain_levels = None" in parkour_env_cfg
    assert "self.curriculum.lin_vel_cmd_levels = None" in parkour_env_cfg
    assert "self.curriculum.command_levels_ang_vel = None" in parkour_env_cfg


def test_uika_task_file_does_not_define_or_register_parkour_tasks():
    source_root = Path(__file__).resolve().parents[1] / "source" / "himloco_lab" / "himloco_lab"
    uika_cfg = (
        source_root / "tasks" / "locomotion" / "robots" / "uika" / "velocity_env_cfg.py"
    ).read_text(encoding="utf-8")
    uika_init = (source_root / "tasks" / "locomotion" / "robots" / "uika" / "__init__.py").read_text(
        encoding="utf-8"
    )

    forbidden_snippets = [
        "ParkourCommandsCfg",
        "ParkourRewardsCfg",
        "RobotParkourEnvCfg",
        "RobotParkourPlayEnvCfg",
        "EXTREME_PARKOUR",
        "UIKA-Parkour",
        "WaypointVelocityCommandCfg",
    ]

    for snippet in forbidden_snippets:
        assert snippet not in uika_cfg
        assert snippet not in uika_init


def test_parkour_task_folder_registers_parkour_tasks():
    source_root = Path(__file__).resolve().parents[1] / "source" / "himloco_lab" / "himloco_lab"
    parkour_init = (
        source_root / "tasks" / "locomotion" / "robots" / "parkour" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert 'id="UIKA-Parkour-Velocity"' in parkour_init
    assert 'id="UIKA-Parkour-Velocity-Play"' in parkour_init
    assert ".velocity_env_cfg:RobotParkourEnvCfg" in parkour_init
    assert ".velocity_env_cfg:RobotParkourPlayEnvCfg" in parkour_init

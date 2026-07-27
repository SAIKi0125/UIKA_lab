from __future__ import annotations

import ast
import importlib.util
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
UIKA_REGISTRY_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/__init__.py"
LOWER_CFG_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/lower_env_cfg.py"
OBSERVATIONS_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/mdp/observations.py"
REWARDS_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/mdp/rewards.py"
TRAIN_PATH = ROOT / "scripts/himloco_rsl_rl/train.py"
PLAY_PATH = ROOT / "scripts/himloco_rsl_rl/play.py"
AGENT_CFG_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_rsl_rl_cfg.py"
)

EXPECTED_LOWER_TARGET = {
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


def _load_observations_module():
    spec = importlib.util.spec_from_file_location("lower_observations_under_test", OBSERVATIONS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lower_tasks_are_registered_without_replacing_velocity_tasks():
    source = UIKA_REGISTRY_PATH.read_text(encoding="utf-8")

    assert 'id="UIKA-Velocity"' in source
    assert 'id="UIKA-Lower-Velocity"' in source
    assert '"env_cfg_entry_point": f"{__name__}.lower_env_cfg:LowerRobotEnvCfg"' in source
    assert 'id="UIKA-Lower-Velocity-Play"' in source
    assert '"env_cfg_entry_point": f"{__name__}.lower_env_cfg:LowerRobotPlayEnvCfg"' in source
    assert source.count("himloco_rsl_rl_cfg:UIKALowerPPORunnerCfg") == 2


def test_lower_runner_uses_isolated_experiment_directory():
    source = AGENT_CFG_PATH.read_text(encoding="utf-8")

    assert "class UIKALowerPPORunnerCfg(UIKAPPORunnerCfg):" in source
    assert 'experiment_name = "uika_lower"' in source


def test_lower_config_uses_requested_pose_commands_and_stable_reset():
    source = LOWER_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "UIKA_LOWER_JOINT_POS_TARGET"
    }

    assert assignments["UIKA_LOWER_JOINT_POS_TARGET"] == EXPECTED_LOWER_TARGET
    assert "terrain = TerrainImporterCfg(" in source
    assert 'terrain_type="plane"' in source
    assert "terrain_generator=None" in source
    assert "pos=(0.0, 0.0, 0.25)" in source
    assert "joint_pos=UIKA_LOWER_JOINT_POS_TARGET" in source
    assert "lin_vel_x=(-1.0, 1.0)" in source
    assert "lin_vel_y=(-0.5, 0.5)" in source
    assert "ang_vel_z=(-0.5, 0.5)" in source
    assert "heading_command=False" in source
    assert '"target_height": 0.25' in source
    assert "func=mdp.root_height_below_minimum" in source
    assert '"minimum_height": 0.15' in source
    assert '"roll": (0.0, 0.0)' in source
    assert '"pitch": (0.0, 0.0)' in source
    assert '"yaw": (0.0, 0.0)' in source
    assert "randomize_reset_joints = EventTerm(" in source
    assert "func=mdp.reset_joints_by_offset" in source


def test_lower_config_uses_crouch_reference_for_actions_observations_and_rewards():
    source = LOWER_CFG_PATH.read_text(encoding="utf-8")

    assert "offset=UIKA_LOWER_JOINT_POS_TARGET" in source
    assert "use_default_offset=False" in source
    assert "func=mdp.joint_pos_rel_to_target" in source
    assert source.count('"target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET') >= 4
    assert "class LowerRewardsCfg(RewardsCfg)" in source
    assert "flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.2)" in source
    assert "feet_air_without_cmd = RewTerm(" in source
    assert "single_foot_air_time = RewTerm(" in source


def test_lower_config_reuses_current_pace_asset_and_events():
    source = LOWER_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    assert "class LowerEventCfg(EventCfg)" in source
    assert "robot: ArticulationCfg = ROBOT_CFG.replace(" in source
    assert "actuators=" not in source
    assert "UIKA_PACE_ACTUATOR_RANGES" not in source
    assert "self._configure_pace_actuators_for_play()" in source
    assert ".ranges =" not in classes["LowerRobotPlayEnvCfg"]


def test_lower_inherits_velocity_observation_and_action_dimensions():
    source = LOWER_CFG_PATH.read_text(encoding="utf-8")

    assert "class LowerObservationsCfg(ObservationsCfg)" in source
    assert "class PolicyCfg(ObservationsCfg.PolicyCfg)" in source
    assert "class CriticCfg(ObservationsCfg.CriticCfg)" in source
    assert "joint_names=UIKA_JOINT_NAMES" in source
    assert "preserve_order=True" in source


def test_joint_position_observation_is_relative_to_named_lower_target():
    observations = _load_observations_module()
    asset = types.SimpleNamespace(
        joint_names=["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"],
        data=types.SimpleNamespace(joint_pos=torch.tensor([[0.10, 0.40, 0.30], [0.20, 0.50, 0.40]])),
    )
    env = types.SimpleNamespace(scene={"robot": asset})
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[2, 0])
    target = {"FL_hip_joint": -0.7, "FL_thigh_joint": 0.3, "FL_calf_joint": 0.2}

    result = observations.joint_pos_rel_to_target(env, asset_cfg, target)

    expected = torch.tensor([[0.10, 0.80], [0.20, 0.90]])
    assert torch.allclose(result, expected)


def test_reward_helpers_accept_named_lower_target():
    source = REWARDS_PATH.read_text(encoding="utf-8")

    assert "def _target_joint_pos(" in source
    stand_still_source = source.split("def stand_still(", 1)[1].split('"""\nRobot.', 1)[0]
    joint_penalty_source = source.split("def joint_pos_penalty(", 1)[1].split("def smoothness(", 1)[0]
    assert "target_joint_pos: dict[str, float] | None = None" in stand_still_source
    assert "target_pos = _target_joint_pos(asset, joint_ids, target_joint_pos)" in stand_still_source
    assert "target_joint_pos: dict[str, float] | None = None" in joint_penalty_source
    assert "target_pos = _target_joint_pos(asset, joint_ids, target_joint_pos)" in joint_penalty_source
    assert "def feet_air_without_cmd(" in source
    assert "def single_foot_air_time(" in source


def test_train_and_play_configure_sim5_urdf_importer_before_app_launcher():
    for script_path in (TRAIN_PATH, PLAY_PATH):
        source = script_path.read_text(encoding="utf-8")
        compat_import = source.index("from isaacsim_compat import configure_isaacsim_urdf_importer")
        compat_call = source.index("configure_isaacsim_urdf_importer()")
        app_launcher_import = source.index("from isaaclab.app import AppLauncher")

        assert compat_import < compat_call < app_launcher_import
        assert "enable_extension" not in source

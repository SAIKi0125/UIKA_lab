from __future__ import annotations

import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UIKA_REGISTRY_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/__init__.py"
)
ROUGH_NORMAL_GAIT_CFG_PATH = (
    ROOT
    / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/rough_normal_gait_env_cfg.py"
)
AGENT_CFG_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_rsl_rl_cfg.py"
)
VELOCITY_CFG_PATH = (
    ROOT
    / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py"
)

# SHA-256 prefixes of normalized AST values from uika/master@c1af265.  Keeping the
# manifest here makes reward parity independent of the mutable local lower config.
MASTER_LOWER_OVERRIDE_FINGERPRINTS = {
    "base_height_l2": "2283b78a6ae3317c",
    "body_lin_acc_l2": "1712affbedb98a15",
    "contact_forces": "8f2cda988d9ca730",
    "feet_air_time": "8f2cda988d9ca730",
    "feet_air_time_variance": "8f2cda988d9ca730",
    "feet_air_without_cmd": "91b8fb4154acdff3",
    "feet_contact": "8f2cda988d9ca730",
    "feet_contact_without_cmd": "b536de55bf8a0bb1",
    "feet_gait": "8f2cda988d9ca730",
    "feet_height": "8f2cda988d9ca730",
    "feet_height_body": "8f2cda988d9ca730",
    "feet_lift_body": "8f2cda988d9ca730",
    "feet_slide": "27183340d267c6f3",
    "feet_stumble": "8f2cda988d9ca730",
    "flat_orientation_l2": "8f00f1944d8cb67c",
    "is_terminated": "8f2cda988d9ca730",
    "joint_mirror": "8f2cda988d9ca730",
    "joint_pos_limits": "bcb12393fc715a90",
    "joint_pos_penalty": "1c8975d4ad313c62",
    "joint_vel_l2": "8f2cda988d9ca730",
    "joint_vel_limits": "8f2cda988d9ca730",
    "prolonged_swing": "8f2cda988d9ca730",
    "single_foot_air_time": "123efd4e4efe2ba2",
    "stand_still": "b42703bb3f34c8c6",
    "track_ang_vel_z": "6760dedb0637f3ca",
    "track_lin_vel_xy": "72050f1d82cdd222",
    "undesired_contacts": "1d7cebe8337c74fe",
    "upward": "8d2b9ae6ffbfeb8d",
}

# The complete allowlist of intentional differences from the master lower recipe.
ROUGH_NORMAL_GAIT_DIFFERENCE_FINGERPRINTS = {
    "base_height_l2": "8904e8b5330db3d3",
    "contact_forces": "e22e459364136305",
    "feet_slide": "942a6a271877b830",
    "joint_pos_penalty": "e445d4f438a641e2",
    "stand_still": "54ac6adc4a750d13",
    "track_ang_vel_z": "0a5d8377b8bbe59a",
    "track_lin_vel_xy": "62f2fd1a6c602bca",
    "undesired_contacts": "b79fa2205b5b351a",
    "upward": "a850769e40d15f93",
}

# Lower inherits these six current normal velocity rewards without overriding them.
MASTER_LOWER_INHERITED_FINGERPRINTS = {
    "action_rate_l2": "af7d416ff02afe4b",
    "ang_vel_xy_l2": "f5ff363641100539",
    "joint_acc_l2": "2b1d23bb4654128c",
    "joint_power": "e2126c3c03541f0d",
    "joint_torques_l2": "76430b30bfc472b9",
    "lin_vel_z_l2": "0342b799612bd0b8",
}


def _class_source(path: Path, class_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    class_source = ast.get_source_segment(source, node)
    assert class_source is not None
    return class_source


def _assignment_fingerprints(path: Path, class_name: str) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fingerprints = {}
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.targets[0], ast.Name):
            continue
        normalized = ast.dump(node.value, include_attributes=False).encode("utf-8")
        fingerprints[node.targets[0].id] = hashlib.sha256(normalized).hexdigest()[:16]
    return fingerprints


def test_rough_normal_gait_tasks_and_logs_are_isolated():
    registry_source = UIKA_REGISTRY_PATH.read_text(encoding="utf-8")
    agent_source = AGENT_CFG_PATH.read_text(encoding="utf-8")

    assert 'id="UIKA-Rough-Normal-Gait"' in registry_source
    assert (
        '"env_cfg_entry_point": f"{__name__}.rough_normal_gait_env_cfg:RoughNormalGaitEnvCfg"'
        in registry_source
    )
    assert 'id="UIKA-Rough-Normal-Gait-Play"' in registry_source
    assert (
        '"env_cfg_entry_point": f"{__name__}.rough_normal_gait_env_cfg:RoughNormalGaitPlayEnvCfg"'
        in registry_source
    )
    assert (
        registry_source.count("himloco_rsl_rl_cfg:UIKARoughNormalGaitPPORunnerCfg")
        == 2
    )
    assert "class UIKARoughNormalGaitPPORunnerCfg(UIKAPPORunnerCfg):" in agent_source
    assert 'experiment_name = "uika_rough_normal_gait"' in agent_source


def test_rough_normal_gait_only_replaces_rewards_on_normal_rough_envs():
    source = ROUGH_NORMAL_GAIT_CFG_PATH.read_text(encoding="utf-8")
    train_source = _class_source(ROUGH_NORMAL_GAIT_CFG_PATH, "RoughNormalGaitEnvCfg")
    play_source = _class_source(ROUGH_NORMAL_GAIT_CFG_PATH, "RoughNormalGaitPlayEnvCfg")

    assert "class RoughNormalGaitEnvCfg(RobotEnvCfg):" in train_source
    assert "class RoughNormalGaitPlayEnvCfg(RobotPlayEnvCfg):" in play_source
    expected_rewards = "rewards: RoughNormalGaitRewardsCfg = RoughNormalGaitRewardsCfg()"
    assert expected_rewards in train_source
    assert expected_rewards in play_source
    forbidden_overrides = (
        "scene:",
        "observations:",
        "actions:",
        "commands:",
        "events:",
        "terminations:",
    )
    for forbidden_override in forbidden_overrides:
        assert forbidden_override not in train_source
        assert forbidden_override not in play_source
    assert "LowerRobotEnvCfg" not in source
    assert "LowerRobotPlayEnvCfg" not in source


def test_rough_normal_gait_uses_normal_pose_and_height_references():
    source = ROUGH_NORMAL_GAIT_CFG_PATH.read_text(encoding="utf-8")
    rewards_source = _class_source(ROUGH_NORMAL_GAIT_CFG_PATH, "RoughNormalGaitRewardsCfg")

    assert "UIKA_NORMAL_JOINT_POS_TARGET = dict(ROBOT_CFG.init_state.joint_pos)" in source
    assert "UIKA_LOWER_JOINT_POS_TARGET" not in source
    assert '"target_height": 0.33' in rewards_source
    assert rewards_source.count('"target_joint_pos": UIKA_NORMAL_JOINT_POS_TARGET') == 2


def test_rough_normal_gait_copies_master_lower_reward_recipe():
    actual = _assignment_fingerprints(
        ROUGH_NORMAL_GAIT_CFG_PATH, "RoughNormalGaitRewardsCfg"
    )
    assert actual.keys() == MASTER_LOWER_OVERRIDE_FINGERPRINTS.keys()

    observed_differences = {
        name
        for name, fingerprint in actual.items()
        if fingerprint != MASTER_LOWER_OVERRIDE_FINGERPRINTS[name]
    }
    assert observed_differences == ROUGH_NORMAL_GAIT_DIFFERENCE_FINGERPRINTS.keys()

    for name, expected in MASTER_LOWER_OVERRIDE_FINGERPRINTS.items():
        if name not in ROUGH_NORMAL_GAIT_DIFFERENCE_FINGERPRINTS:
            assert actual[name] == expected
    for name, expected in ROUGH_NORMAL_GAIT_DIFFERENCE_FINGERPRINTS.items():
        assert actual[name] == expected

    normal_velocity = _assignment_fingerprints(VELOCITY_CFG_PATH, "RewardsCfg")
    inherited_names = normal_velocity.keys() - actual.keys()
    assert inherited_names == MASTER_LOWER_INHERITED_FINGERPRINTS.keys()
    assert {
        name: normal_velocity[name] for name in MASTER_LOWER_INHERITED_FINGERPRINTS
    } == MASTER_LOWER_INHERITED_FINGERPRINTS

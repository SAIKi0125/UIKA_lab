from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UIKA_REGISTRY_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/__init__.py"
FLAT_CFG_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/flat_env_cfg.py"
VELOCITY_CFG_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py"
AGENT_CFG_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/agents/himloco_rsl_rl_cfg.py"


def _classes(path: Path) -> dict[str, ast.ClassDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def _class_source(path: Path, class_name: str) -> str:
    source = path.read_text(encoding="utf-8")
    node = _classes(path)[class_name]
    class_source = ast.get_source_segment(source, node)
    assert class_source is not None
    return class_source


def test_flat_tasks_are_registered_with_isolated_runner():
    source = UIKA_REGISTRY_PATH.read_text(encoding="utf-8")

    assert 'id="UIKA-Flat-Velocity"' in source
    assert '"env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatRobotEnvCfg"' in source
    assert 'id="UIKA-Flat-Velocity-Play"' in source
    assert '"env_cfg_entry_point": f"{__name__}.flat_env_cfg:FlatRobotPlayEnvCfg"' in source
    assert source.count("himloco_rsl_rl_cfg:UIKAFlatPPORunnerCfg") == 2


def test_flat_scene_only_replaces_rough_terrain_with_plane():
    source = FLAT_CFG_PATH.read_text(encoding="utf-8")
    scene_source = _class_source(FLAT_CFG_PATH, "FlatRobotSceneCfg")

    assert "class FlatRobotSceneCfg(RobotSceneCfg):" in source
    assert 'terrain_type="plane"' in scene_source
    assert "terrain_generator=None" in scene_source
    assert "collision_group=-1" in scene_source
    assert "static_friction=1.0" in scene_source
    assert "dynamic_friction=1.0" in scene_source


def test_flat_train_and_play_inherit_normal_height_velocity_behavior():
    classes = _classes(FLAT_CFG_PATH)
    train = classes["FlatRobotEnvCfg"]
    play = classes["FlatRobotPlayEnvCfg"]

    assert [base.id for base in train.bases if isinstance(base, ast.Name)] == ["RobotEnvCfg"]
    assert [base.id for base in play.bases if isinstance(base, ast.Name)] == ["RobotPlayEnvCfg"]
    assert "rewards" not in _class_source(FLAT_CFG_PATH, "FlatRobotEnvCfg")
    assert "commands" not in _class_source(FLAT_CFG_PATH, "FlatRobotEnvCfg")
    assert "rewards" not in _class_source(FLAT_CFG_PATH, "FlatRobotPlayEnvCfg")
    assert "commands" not in _class_source(FLAT_CFG_PATH, "FlatRobotPlayEnvCfg")


def test_flat_training_inherits_full_command_range_and_play_override():
    commands_source = _class_source(VELOCITY_CFG_PATH, "CommandsCfg")
    play_source = _class_source(VELOCITY_CFG_PATH, "RobotPlayEnvCfg")

    assert "lin_vel_x=(-1.0, 1.0)" in commands_source
    assert "lin_vel_y=(-1.0, 1.0)" in commands_source
    assert "ang_vel_z=(-1.0, 1.0)" in commands_source
    assert "resampling_time_range=(10.0, 10.0)" in commands_source
    assert "lin_vel_x=(1.0, 1.0)" in play_source
    assert "lin_vel_y=(-0.0, 0.0)" in play_source
    assert "ang_vel_z=(-0, 0)" in play_source


def test_flat_runner_uses_separate_experiment_directory():
    source = AGENT_CFG_PATH.read_text(encoding="utf-8")

    assert "class UIKAFlatPPORunnerCfg(UIKAPPORunnerCfg):" in source
    assert 'experiment_name = "uika_flat"' in source

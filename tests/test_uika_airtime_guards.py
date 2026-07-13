from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REWARDS_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/mdp/rewards.py"
VELOCITY_CFG_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py"
)
LOWER_CFG_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/lower_env_cfg.py"


def _class_assignments(path: Path, class_name: str) -> dict[str, ast.expr]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return {
        node.targets[0].id: node.value
        for node in class_node.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }


def _keyword(call: ast.Call, name: str) -> ast.expr:
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def _attribute_name(node: ast.expr) -> str:
    assert isinstance(node, ast.Attribute)
    assert isinstance(node.value, ast.Name)
    return f"{node.value.id}.{node.attr}"


def _params(call: ast.Call) -> dict[str, ast.expr]:
    params = _keyword(call, "params")
    assert isinstance(params, ast.Dict)
    return {
        ast.literal_eval(key): value
        for key, value in zip(params.keys, params.values, strict=True)
        if key is not None
    }


def test_prolonged_swing_uses_current_air_time_and_safe_gates():
    source = REWARDS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "prolonged_swing"
    )
    function_source = ast.get_source_segment(source, function)

    assert function_source is not None
    assert "contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]" in function_source
    assert "torch.clamp(air_time - max_swing_time, min=0.0)" in function_source
    assert "torch.sum(torch.square(excess), dim=1)" in function_source
    assert "get_command(command_name)[:, :2]" in function_source
    assert "_upright_gate(env)" in function_source


def test_velocity_registers_airtime_guards_without_changing_existing_airtime_terms():
    assignments = _class_assignments(VELOCITY_CFG_PATH, "RewardsCfg")

    feet_air_time = assignments["feet_air_time"]
    assert isinstance(feet_air_time, ast.Call)
    assert ast.literal_eval(_keyword(feet_air_time, "weight")) == 0.1
    assert ast.literal_eval(_params(feet_air_time)["threshold"]) == 0.5

    variance = assignments["feet_air_time_variance"]
    assert isinstance(variance, ast.Call)
    assert ast.literal_eval(_keyword(variance, "weight")) == -1.0

    prolonged = assignments["prolonged_swing"]
    assert isinstance(prolonged, ast.Call)
    assert _attribute_name(_keyword(prolonged, "func")) == "mdp.prolonged_swing"
    assert ast.literal_eval(_keyword(prolonged, "weight")) == -1.5
    prolonged_params = _params(prolonged)
    assert ast.literal_eval(prolonged_params["max_swing_time"]) == 0.60
    assert ast.literal_eval(prolonged_params["command_name"]) == "base_velocity"

    stationary = assignments["feet_air_without_cmd"]
    assert isinstance(stationary, ast.Call)
    assert _attribute_name(_keyword(stationary, "func")) == "mdp.feet_air_without_cmd"
    assert ast.literal_eval(_keyword(stationary, "weight")) == -2.0
    stationary_params = _params(stationary)
    assert ast.literal_eval(stationary_params["command_threshold"]) == 0.1
    assert ast.literal_eval(stationary_params["contact_threshold"]) == 1.0


def test_lower_disables_prolonged_swing_and_keeps_stationary_air_foot_penalty():
    assignments = _class_assignments(LOWER_CFG_PATH, "LowerRewardsCfg")

    assert ast.literal_eval(assignments["prolonged_swing"]) is None
    stationary = assignments["feet_air_without_cmd"]
    assert isinstance(stationary, ast.Call)
    assert _attribute_name(_keyword(stationary, "func")) == "mdp.feet_air_without_cmd"
    assert ast.literal_eval(_keyword(stationary, "weight")) == -2.0

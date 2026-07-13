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


def test_body_frame_height_mdp_functions_keep_distinct_semantics():
    source = REWARDS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }

    height_error = functions["feet_height_body"]
    lift_progress = functions["feet_lift_body"]

    assert "torch.square(footpos_in_body_frame[:, :, 2] - target_height)" in height_error
    assert "return reward" in height_error
    assert "torch.clamp(" in lift_progress
    assert "(foot_pos_body[:, :, 2] - minimum_height) / height_range" in lift_progress
    assert "return reward" in lift_progress


def test_velocity_registers_negative_height_error_and_positive_lift_terms():
    assignments = _class_assignments(VELOCITY_CFG_PATH, "RewardsCfg")
    height_term = assignments["feet_height_body"]
    lift_term = assignments["feet_lift_body"]

    assert isinstance(height_term, ast.Call)
    assert _attribute_name(_keyword(height_term, "func")) == "mdp.feet_height_body"
    assert ast.literal_eval(_keyword(height_term, "weight")) == -0.01
    height_params = _params(height_term)
    assert ast.literal_eval(height_params["target_height"]) == -0.20
    assert ast.literal_eval(height_params["tanh_mult"]) == 2.0
    assert ast.literal_eval(height_params["command_name"]) == "base_velocity"

    assert isinstance(lift_term, ast.Call)
    assert _attribute_name(_keyword(lift_term, "func")) == "mdp.feet_lift_body"
    assert ast.literal_eval(_keyword(lift_term, "weight")) == 2.0
    lift_params = _params(lift_term)
    assert ast.literal_eval(lift_params["minimum_height"]) == -0.30
    assert ast.literal_eval(lift_params["target_height"]) == -0.10
    assert ast.literal_eval(lift_params["tanh_mult"]) == 2.0
    assert ast.literal_eval(lift_params["command_name"]) == "base_velocity"


def test_lower_task_disables_both_inherited_body_frame_height_terms():
    assignments = _class_assignments(LOWER_CFG_PATH, "LowerRewardsCfg")

    assert ast.literal_eval(assignments["feet_height_body"]) is None
    assert ast.literal_eval(assignments["feet_lift_body"]) is None

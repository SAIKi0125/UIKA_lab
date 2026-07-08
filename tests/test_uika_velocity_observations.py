import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _install_observation_stubs():
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")

    isaaclab_sensors.ContactSensor = object
    isaaclab_math.euler_xyz_from_quat = lambda quat: (quat[:, 0], quat[:, 1], quat[:, 2])

    sys.modules.setdefault("isaaclab", isaaclab)
    sys.modules.setdefault("isaaclab.sensors", isaaclab_sensors)
    sys.modules.setdefault("isaaclab.utils", isaaclab_utils)
    sys.modules.setdefault("isaaclab.utils.math", isaaclab_math)


def _load_observations_module():
    _install_observation_stubs()
    observations_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "observations.py"
    )
    spec = importlib.util.spec_from_file_location("uika_velocity_observations_under_test", observations_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_joint_pos_rel_to_target_uses_joint_names_not_default_pose():
    observations = _load_observations_module()
    asset = SimpleNamespace(
        joint_names=["FL_hip_joint", "FR_hip_joint", "RR_calf_joint"],
        data=SimpleNamespace(
            joint_pos=torch.tensor([[-0.6, 0.9, 0.35]]),
            default_joint_pos=torch.tensor([[-0.78, 0.78, 0.70]]),
        ),
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[2, 0])
    target_joint_pos = {
        "FL_hip_joint": -0.7,
        "FR_hip_joint": 0.7,
        "RR_calf_joint": 0.20,
    }

    obs = observations.joint_pos_rel_to_target(env, asset_cfg=asset_cfg, target_joint_pos=target_joint_pos)

    torch.testing.assert_close(obs, torch.tensor([[0.15, 0.10]]))


def test_uika_velocity_policy_joint_pos_observation_targets_lower_pose():
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
    obs_source = source.split("joint_pos_rel = ObsTerm(", 1)[1].split("joint_vel_rel = ObsTerm(", 1)[0]

    assert "func=mdp.joint_pos_rel_to_target" in obs_source
    assert '"target_joint_pos": UIKA_LOWER_JOINT_POS_TARGET' in obs_source

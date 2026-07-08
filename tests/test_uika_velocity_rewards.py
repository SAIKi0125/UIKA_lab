import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch


def _install_isaaclab_stubs():
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")
    isaaclab_assets = types.ModuleType("isaaclab.assets")
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_envs_mdp = types.ModuleType("isaaclab.envs.mdp")
    isaaclab_managers = types.ModuleType("isaaclab.managers")
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")

    isaaclab_math.quat_apply_inverse = lambda quat, vec: vec
    isaaclab_assets.Articulation = object
    isaaclab_assets.RigidObject = object
    isaaclab_managers.ManagerTermBase = object
    isaaclab_managers.SceneEntityCfg = lambda *args, **kwargs: SimpleNamespace(
        name=args[0] if args else kwargs.get("name", "robot"),
        body_ids=kwargs.get("body_ids"),
        joint_ids=kwargs.get("joint_ids"),
    )
    isaaclab_managers.RewardTermCfg = object
    isaaclab_sensors.ContactSensor = object
    isaaclab_sensors.RayCaster = object

    sys.modules.setdefault("isaaclab", isaaclab)
    sys.modules.setdefault("isaaclab.utils", isaaclab_utils)
    sys.modules.setdefault("isaaclab.utils.math", isaaclab_math)
    sys.modules.setdefault("isaaclab.assets", isaaclab_assets)
    sys.modules.setdefault("isaaclab.envs", isaaclab_envs)
    sys.modules.setdefault("isaaclab.envs.mdp", isaaclab_envs_mdp)
    sys.modules.setdefault("isaaclab.managers", isaaclab_managers)
    sys.modules.setdefault("isaaclab.sensors", isaaclab_sensors)


def _load_rewards_module():
    _install_isaaclab_stubs()
    rewards_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "rewards.py"
    )
    spec = importlib.util.spec_from_file_location("uika_velocity_rewards_under_test", rewards_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Scene(dict):
    @property
    def sensors(self):
        return self["sensors"]


def test_single_foot_air_time_accumulates_only_air_time_above_threshold():
    rewards = _load_rewards_module()
    env = SimpleNamespace(
        scene=_Scene(
            {
                "robot": SimpleNamespace(
                    data=SimpleNamespace(projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]))
                ),
                "sensors": {
                    "contact_forces": SimpleNamespace(
                        data=SimpleNamespace(
                            current_air_time=torch.tensor(
                                [
                                    [0.10, 0.30, 0.70, 0.00],
                                    [0.26, 0.25, 1.00, 0.40],
                                ]
                            )
                        )
                    )
                },
            }
        )
    )
    sensor_cfg = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])

    penalty = rewards.single_foot_air_time(env, sensor_cfg=sensor_cfg, threshold=0.25)

    torch.testing.assert_close(penalty, torch.tensor([0.50, 0.91]))


def test_uika_velocity_registers_single_foot_air_time_penalty_for_all_commands():
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
    reward_source = source.split("single_foot_air_time = RewTerm(", 1)[1].split("feet_stumble = None", 1)[0]

    assert "func=mdp.single_foot_air_time" in reward_source
    assert "weight=-2.0" in reward_source
    assert '"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")' in reward_source
    assert '"threshold": 0.25' in reward_source
    assert '"command_name"' not in reward_source

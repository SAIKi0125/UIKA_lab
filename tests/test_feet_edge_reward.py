from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch


def _load_extreme_parkour_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "terrains"
        / "extreme_parkour.py"
    )
    spec = importlib.util.spec_from_file_location("extreme_parkour_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_reward_stubs(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_assets = types.ModuleType("isaaclab.assets")
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_envs_mdp = types.ModuleType("isaaclab.envs.mdp")
    isaaclab_managers = types.ModuleType("isaaclab.managers")
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")

    isaaclab_assets.Articulation = object
    isaaclab_assets.RigidObject = object
    isaaclab_envs_mdp.joint_deviation_l1 = lambda *args, **kwargs: None
    isaaclab_managers.ManagerTermBase = object
    isaaclab_managers.RewardTermCfg = object
    isaaclab_sensors.ContactSensor = object
    isaaclab_sensors.RayCaster = object
    isaaclab_utils.math = isaaclab_math
    isaaclab_math.quat_apply_inverse = lambda quat, vec: vec

    class SceneEntityCfg:
        def __init__(self, name, body_ids=None, joint_ids=None, **kwargs):
            self.name = name
            self.body_ids = body_ids
            self.joint_ids = joint_ids

    isaaclab_managers.SceneEntityCfg = SceneEntityCfg

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", isaaclab_assets)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", isaaclab_envs)
    monkeypatch.setitem(sys.modules, "isaaclab.envs.mdp", isaaclab_envs_mdp)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", isaaclab_managers)
    monkeypatch.setitem(sys.modules, "isaaclab.sensors", isaaclab_sensors)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", isaaclab_utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", isaaclab_math)


def _load_rewards_module(monkeypatch):
    _install_reward_stubs(monkeypatch)
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "rewards.py"
    )
    spec = importlib.util.spec_from_file_location("rewards_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_height_field_edge_mask_marks_drop_boundaries_not_flat_interior():
    module = _load_extreme_parkour_module()
    height_field = np.array(
        [
            [0, 0, 0, 0],
            [0, 8, 8, 0],
            [0, 8, 8, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int16,
    )

    edge_mask = module.build_height_field_edge_mask(
        height_field,
        vertical_scale=0.05,
        height_threshold=0.2,
        edge_width_px=0,
    )

    assert edge_mask.dtype == np.bool_
    assert not edge_mask[1, 1]
    assert edge_mask[0, 1]
    assert edge_mask[1, 0]
    assert edge_mask[3, 2]


def test_feet_edge_counts_contacting_feet_on_current_terrain_edge(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    edge_masks = {
        "BridgeA": np.zeros((8, 8), dtype=np.bool_),
        "BridgeB": np.zeros((8, 8), dtype=np.bool_),
    }
    edge_masks["BridgeA"][4, 4] = True

    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
            body_pos_w=torch.tensor(
                [
                    [[0.0, 0.0, 0.0], [0.4, 0.4, 0.0], [0.5, 0.5, 0.0], [0.6, 0.6, 0.0]],
                    [[8.0, 0.0, 0.0], [8.4, 0.4, 0.0], [8.5, 0.5, 0.0], [8.6, 0.6, 0.0]],
                ],
                dtype=torch.float,
            ),
        )
    )
    contact_sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(
            net_forces_w_history=torch.tensor(
                [
                    [[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 0.0, 5.0]]],
                    [[[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]]],
                ],
                dtype=torch.float,
            )
        )
    )
    terrain = types.SimpleNamespace(
        terrain_origins=torch.tensor([[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]]),
        cfg=types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(
                num_cols=2,
                sub_terrains={
                    "BridgeA": types.SimpleNamespace(proportion=1.0),
                    "BridgeB": types.SimpleNamespace(proportion=1.0),
                },
            )
        ),
    )

    class Scene:
        def __init__(self):
            self.sensors = {"contact_forces": contact_sensor}
            self.terrain = terrain
            self.env_origins = terrain.terrain_origins[0]

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    env = types.SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())
    sensor_cfg = types.SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
    asset_cfg = types.SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])

    reward = rewards.feet_edge(
        env,
        sensor_cfg=sensor_cfg,
        asset_cfg=asset_cfg,
        edge_masks=edge_masks,
        terrain_names=("BridgeA", "BridgeB"),
        tile_size=(0.8, 0.8),
        horizontal_scale=0.1,
        contact_threshold=1.0,
    )

    assert reward.tolist() == [1.0, 0.0]

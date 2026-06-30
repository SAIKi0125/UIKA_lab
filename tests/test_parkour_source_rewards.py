from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch


def _install_reward_stubs(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    isaaclab_assets = types.ModuleType("isaaclab.assets")
    isaaclab_envs = types.ModuleType("isaaclab.envs")
    isaaclab_envs_mdp = types.ModuleType("isaaclab.envs.mdp")
    isaaclab_managers = types.ModuleType("isaaclab.managers")
    isaaclab_sensors = types.ModuleType("isaaclab.sensors")
    isaaclab_utils = types.ModuleType("isaaclab.utils")
    isaaclab_math = types.ModuleType("isaaclab.utils.math")

    class ManagerTermBase:
        def __init__(self, cfg, env):
            self.cfg = cfg
            self.device = env.device

    class SceneEntityCfg:
        def __init__(self, name, body_ids=None, joint_ids=None, **kwargs):
            self.name = name
            self.body_ids = body_ids
            self.joint_ids = joint_ids

    isaaclab_assets.Articulation = object
    isaaclab_assets.RigidObject = object
    isaaclab_envs_mdp.joint_deviation_l1 = lambda *args, **kwargs: None
    isaaclab_managers.ManagerTermBase = ManagerTermBase
    isaaclab_managers.RewardTermCfg = object
    isaaclab_managers.SceneEntityCfg = SceneEntityCfg
    isaaclab_sensors.ContactSensor = object
    isaaclab_sensors.RayCaster = object
    isaaclab_utils.math = isaaclab_math
    isaaclab_math.quat_apply_inverse = lambda quat, vec: vec

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
    spec = importlib.util.spec_from_file_location("parkour_rewards_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env_with_robot(robot, command=None, action_manager=None):
    class Scene:
        def __getitem__(self, name):
            assert name == "robot"
            return robot

    if command is None:
        command = torch.zeros(robot.data.joint_pos.shape[0], 3)
    command_manager = types.SimpleNamespace(get_command=lambda name: command)
    return types.SimpleNamespace(
        num_envs=robot.data.joint_pos.shape[0],
        device=torch.device("cpu"),
        scene=Scene(),
        command_manager=command_manager,
        action_manager=action_manager,
    )


def test_joint_dof_error_l2_matches_source_sum_squared_default_offset(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=torch.tensor([[1.0, 2.0, 3.0], [0.5, -0.5, 1.5]]),
            default_joint_pos=torch.tensor([[0.0, 1.0, 1.0], [0.0, 0.5, 1.0]]),
        )
    )
    env = _env_with_robot(robot)

    reward = rewards.joint_dof_error_l2(env, asset_cfg=types.SimpleNamespace(name="robot", joint_ids=None))

    torch.testing.assert_close(reward, torch.tensor([6.0, 1.5]))


def test_hip_pos_l2_uses_selected_joint_ids(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            default_joint_pos=torch.tensor([[0.0, 1.0, 1.0, 1.0]]),
        )
    )
    env = _env_with_robot(robot)

    reward = rewards.hip_pos_l2(env, asset_cfg=types.SimpleNamespace(name="robot", joint_ids=[0, 2]))

    torch.testing.assert_close(reward, torch.tensor([5.0]))


def test_track_goal_vel_from_command_projects_body_velocity_onto_command_direction(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=torch.zeros(3, 1),
            root_lin_vel_b=torch.tensor([[0.5, 0.0, 0.0], [1.0, 1.0, 0.0], [-0.5, 0.0, 0.0]]),
        )
    )
    command = torch.tensor([[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    env = _env_with_robot(robot, command=command)

    reward = rewards.track_goal_vel_from_command(
        env, command_name="base_velocity", asset_cfg=types.SimpleNamespace(name="robot")
    )

    expected = torch.tensor([0.5, 1.0, -0.5])
    torch.testing.assert_close(reward, expected, atol=1e-5, rtol=1e-5)


def test_source_base_motion_penalties_match_squared_components_without_upright_gate(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_pos=torch.zeros(2, 1),
            root_lin_vel_b=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, -3.0]]),
            root_ang_vel_b=torch.tensor([[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]]),
        )
    )
    env = _env_with_robot(robot)

    lin_vel_z = rewards.source_lin_vel_z_l2(env, asset_cfg=types.SimpleNamespace(name="robot"))
    ang_vel_xy = rewards.source_ang_vel_xy_l2(env, asset_cfg=types.SimpleNamespace(name="robot"))

    torch.testing.assert_close(lin_vel_z, torch.tensor([4.0, 9.0]))
    torch.testing.assert_close(ang_vel_xy, torch.tensor([5.0, 25.0]))


def test_collision_contacts_counts_current_contact_violations_only(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    contact_sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(
            net_forces_w_history=torch.tensor(
                [
                    [
                        [[0.0, 0.0, 0.2], [0.0, 0.0, 0.0], [0.0, 0.0, 0.4]],
                        [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.3]],
                        [[0.0, 0.0, 5.0], [0.0, 0.0, 5.0], [0.0, 0.0, 5.0]],
                    ],
                ],
                dtype=torch.float,
            )
        )
    )
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.zeros(2, 1)))

    class Scene:
        sensors = {"contact_forces": contact_sensor}

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    env = types.SimpleNamespace(num_envs=2, device=torch.device("cpu"), scene=Scene())

    reward = rewards.collision_contacts(
        env,
        threshold=0.1,
        sensor_cfg=types.SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2]),
    )

    torch.testing.assert_close(reward, torch.tensor([2.0, 2.0]))


def test_action_rate_norm_uses_raw_action_delta_when_action_term_is_available(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(data=types.SimpleNamespace(joint_pos=torch.zeros(2, 1)))
    action_term = types.SimpleNamespace(raw_actions=torch.tensor([[3.0, 4.0], [1.0, 1.0]]))
    action_manager = types.SimpleNamespace(get_term=lambda name: action_term)
    env = _env_with_robot(robot, action_manager=action_manager)
    cfg = types.SimpleNamespace(
        params={"asset_cfg": types.SimpleNamespace(name="robot"), "action_term_name": "joint_pos"}
    )
    term = rewards.ActionRateNorm(cfg, env)

    first = term(env, asset_cfg=types.SimpleNamespace(name="robot"), action_term_name="joint_pos")
    action_term.raw_actions = torch.tensor([[0.0, 0.0], [4.0, 5.0]])
    second = term(env, asset_cfg=types.SimpleNamespace(name="robot"), action_term_name="joint_pos")

    torch.testing.assert_close(first, torch.tensor([5.0, 2.0**0.5]))
    torch.testing.assert_close(second, torch.tensor([5.0, 5.0]))


def test_delta_torques_l2_tracks_previous_applied_torque(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        num_joints=2,
        data=types.SimpleNamespace(
            joint_pos=torch.zeros(2, 1),
            applied_torque=torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        ),
    )
    env = _env_with_robot(robot)
    cfg = types.SimpleNamespace(params={"asset_cfg": types.SimpleNamespace(name="robot")})
    term = rewards.DeltaTorquesL2(cfg, env)

    first = term(env, asset_cfg=types.SimpleNamespace(name="robot"))
    robot.data.applied_torque = torch.tensor([[2.0, -1.0], [0.0, 4.0]])
    second = term(env, asset_cfg=types.SimpleNamespace(name="robot"))

    torch.testing.assert_close(first, torch.tensor([5.0, 1.0]))
    torch.testing.assert_close(second, torch.tensor([10.0, 9.0]))


def test_joint_dof_acc_l2_tracks_finite_difference_joint_velocity(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        num_joints=2,
        data=types.SimpleNamespace(
            joint_pos=torch.zeros(2, 1),
            joint_vel=torch.tensor([[1.0, 2.0], [0.0, 1.0]]),
        ),
    )
    env = _env_with_robot(robot)
    env.step_dt = 0.5
    cfg = types.SimpleNamespace(params={"asset_cfg": types.SimpleNamespace(name="robot")})
    term = rewards.JointDofAccL2(cfg, env)

    first = term(env, asset_cfg=types.SimpleNamespace(name="robot"))
    robot.data.joint_vel = torch.tensor([[2.0, -1.0], [0.0, 4.0]])
    second = term(env, asset_cfg=types.SimpleNamespace(name="robot"))

    torch.testing.assert_close(first, torch.tensor([20.0, 4.0]))
    torch.testing.assert_close(second, torch.tensor([40.0, 36.0]))

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


def _load_observations_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "tasks"
        / "locomotion"
        / "mdp"
        / "observations.py"
    )
    spec = importlib.util.spec_from_file_location("parkour_observations_under_test", module_path)
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


def _env_with_robot_and_terrain(robot, terrain_names, sub_terrains, env_origins):
    terrain_origins = torch.tensor([env_origins], dtype=torch.float)
    terrain = types.SimpleNamespace(
        terrain_origins=terrain_origins,
        cfg=types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(num_cols=len(env_origins), sub_terrains=sub_terrains)
        ),
    )

    class Scene:
        def __init__(self):
            self.env_origins = terrain_origins[0]
            self.terrain = terrain

        def __getitem__(self, name):
            assert name == "robot"
            return robot

    return types.SimpleNamespace(
        num_envs=robot.data.root_lin_vel_b.shape[0],
        device=torch.device("cpu"),
        scene=Scene(),
        terrain_names=terrain_names,
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


def test_source_lin_vel_z_matches_parkour_flat_nonflat_scaling(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_lin_vel_b=torch.tensor([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
            projected_gravity_b=torch.tensor([[0.5, 0.0, -1.0], [0.5, 0.0, -1.0]]),
        )
    )
    env = _env_with_robot_and_terrain(
        robot,
        terrain_names=("BridgeA", "Flat"),
        sub_terrains={
            "BridgeA": types.SimpleNamespace(proportion=1.0),
            "Flat": types.SimpleNamespace(proportion=1.0),
        },
        env_origins=[[-8.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
    )

    reward = rewards.source_lin_vel_z_l2(
        env,
        asset_cfg=types.SimpleNamespace(name="robot"),
        terrain_names=("BridgeA", "Flat"),
        flat_terrain_names=("Flat",),
        nonflat_scale=0.5,
    )

    torch.testing.assert_close(reward, torch.tensor([2.0, 4.0]))


def test_source_flat_orientation_only_penalizes_flat_terrain(monkeypatch):
    rewards = _load_rewards_module(monkeypatch)
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            root_lin_vel_b=torch.zeros(2, 3),
            projected_gravity_b=torch.tensor([[0.3, 0.4, -1.0], [0.3, 0.4, -1.0]]),
        )
    )
    env = _env_with_robot_and_terrain(
        robot,
        terrain_names=("BridgeA", "Flat"),
        sub_terrains={
            "BridgeA": types.SimpleNamespace(proportion=1.0),
            "Flat": types.SimpleNamespace(proportion=1.0),
        },
        env_origins=[[-8.0, 0.0, 0.0], [8.0, 0.0, 0.0]],
    )

    reward = rewards.source_flat_orientation_l2(
        env,
        asset_cfg=types.SimpleNamespace(name="robot"),
        terrain_names=("BridgeA", "Flat"),
        flat_terrain_names=("Flat",),
    )

    torch.testing.assert_close(reward, torch.tensor([0.0, 0.25]))


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


def test_waypoint_height_delta_observes_current_marker_height_relative_to_base():
    observations = _load_observations_module()
    command = types.SimpleNamespace(
        goal_idx=torch.tensor([1, 0]),
        _marker_routes=torch.tensor(
            [
                [[0.0, 0.0, 0.10], [0.0, 0.0, 0.45]],
                [[0.0, 0.0, -0.05], [0.0, 0.0, 0.25]],
            ]
        ),
        _terrain_route_ids=lambda: torch.tensor([0, 1]),
    )
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.10]]))
    )
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: command),
        scene={"robot": robot},
    )

    obs = observations.waypoint_height_delta(
        env, command_name="base_velocity", asset_cfg=types.SimpleNamespace(name="robot")
    )

    torch.testing.assert_close(obs, torch.tensor([[0.15], [-0.15]]))


def test_waypoint_height_delta_returns_zero_without_marker_routes():
    observations = _load_observations_module()
    command = types.SimpleNamespace(
        goal_idx=torch.tensor([0, 0]),
        _routes=torch.zeros(2, 1, 2),
        _terrain_route_ids=lambda: torch.tensor([0, 1]),
    )
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(root_pos_w=torch.tensor([[0.0, 0.0, 0.30], [0.0, 0.0, 0.10]]))
    )
    env = types.SimpleNamespace(
        command_manager=types.SimpleNamespace(get_term=lambda name: command),
        scene={"robot": robot},
    )

    obs = observations.waypoint_height_delta(
        env, command_name="base_velocity", asset_cfg=types.SimpleNamespace(name="robot")
    )

    torch.testing.assert_close(obs, torch.zeros(2, 1))


def test_foot_contact_state_reports_filtered_current_or_previous_contacts():
    observations = _load_observations_module()
    contact_sensor = types.SimpleNamespace(
        data=types.SimpleNamespace(
            net_forces_w_history=torch.tensor(
                [
                    [
                        [[0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]],
                    ],
                    [
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                    ],
                ],
                dtype=torch.float,
            )
        )
    )
    env = types.SimpleNamespace(scene=types.SimpleNamespace(sensors={"contact_forces": contact_sensor}))

    obs = observations.foot_contact_state(
        env, sensor_cfg=types.SimpleNamespace(name="contact_forces", body_ids=[0, 1]), threshold=2.0
    )

    torch.testing.assert_close(obs, torch.tensor([[0.5, 0.5], [-0.5, -0.5]]))


def test_terrain_route_one_hot_observes_current_waypoint_route_id():
    observations = _load_observations_module()
    command = types.SimpleNamespace(
        _route_names=("T_step_stl", "Slope", "Flat"),
        _terrain_route_ids=lambda: torch.tensor([2, 0]),
    )
    env = types.SimpleNamespace(command_manager=types.SimpleNamespace(get_term=lambda name: command))

    obs = observations.terrain_route_one_hot(env, command_name="base_velocity")

    torch.testing.assert_close(obs, torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]))


def test_terrain_route_one_hot_falls_back_to_scene_terrain_types():
    observations = _load_observations_module()
    command = types.SimpleNamespace()
    terrain = types.SimpleNamespace(
        terrain_types=torch.tensor([1, 0]),
        cfg=types.SimpleNamespace(
            terrain_generator=types.SimpleNamespace(
                sub_terrains={
                    "slope": types.SimpleNamespace(),
                    "stairs": types.SimpleNamespace(),
                    "obstacles": types.SimpleNamespace(),
                }
            )
        ),
    )
    env = types.SimpleNamespace(
        num_envs=2,
        device=torch.device("cpu"),
        command_manager=types.SimpleNamespace(get_term=lambda name: command),
        scene=types.SimpleNamespace(terrain=terrain),
    )

    obs = observations.terrain_route_one_hot(env, command_name="base_velocity")

    torch.testing.assert_close(obs, torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]))


def test_base_mass_com_observes_body_mass_and_com():
    observations = _load_observations_module()
    robot = types.SimpleNamespace(
        root_physx_view=types.SimpleNamespace(get_masses=lambda: torch.tensor([[10.0, 1.0], [12.0, 1.5]])),
        data=types.SimpleNamespace(
            com_pos_b=torch.tensor(
                [
                    [[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]],
                    [[0.4, 0.5, 0.6], [0.0, 0.0, 0.0]],
                ]
            )
        ),
    )
    env = types.SimpleNamespace(scene={"robot": robot})

    obs = observations.base_mass_com(env, asset_cfg=types.SimpleNamespace(name="robot", body_ids=[0]))

    torch.testing.assert_close(obs, torch.tensor([[10.0, 0.1, 0.2, 0.3], [12.0, 0.4, 0.5, 0.6]]))


def test_base_mass_com_moves_cpu_masses_to_asset_device_when_cuda_is_available():
    if not torch.cuda.is_available():
        return
    observations = _load_observations_module()
    robot = types.SimpleNamespace(
        root_physx_view=types.SimpleNamespace(get_masses=lambda: torch.tensor([[10.0], [12.0]], device="cpu")),
        data=types.SimpleNamespace(
            com_pos_b=torch.tensor([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], device="cuda:0")
        ),
    )
    env = types.SimpleNamespace(scene={"robot": robot})

    obs = observations.base_mass_com(env, asset_cfg=types.SimpleNamespace(name="robot", body_ids=[0]))

    assert obs.device.type == "cuda"
    torch.testing.assert_close(obs.cpu(), torch.tensor([[10.0, 0.1, 0.2, 0.3], [12.0, 0.4, 0.5, 0.6]]))


def test_body_friction_observes_first_material_static_friction():
    observations = _load_observations_module()
    robot = types.SimpleNamespace(
        root_physx_view=types.SimpleNamespace(
            get_material_properties=lambda: torch.tensor(
                [
                    [[0.8, 0.7, 0.0], [0.1, 0.1, 0.0]],
                    [[1.2, 1.1, 0.0], [0.1, 0.1, 0.0]],
                ]
            )
        )
    )
    env = types.SimpleNamespace(scene={"robot": robot})

    obs = observations.body_friction(env, asset_cfg=types.SimpleNamespace(name="robot"))

    torch.testing.assert_close(obs, torch.tensor([[0.8], [1.2]]))


def test_joint_stiffness_damping_scale_observes_randomization_ratios():
    observations = _load_observations_module()
    robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            joint_stiffness=torch.tensor([[2.0, 3.0], [1.0, 6.0]]),
            default_joint_stiffness=torch.tensor([[1.0, 3.0], [1.0, 3.0]]),
            joint_damping=torch.tensor([[4.0, 2.0], [2.0, 8.0]]),
            default_joint_damping=torch.tensor([[2.0, 2.0], [2.0, 4.0]]),
        )
    )
    env = types.SimpleNamespace(scene={"robot": robot})

    obs = observations.joint_stiffness_damping_scale(env, asset_cfg=types.SimpleNamespace(name="robot"))

    torch.testing.assert_close(obs, torch.tensor([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]]))


def test_parkour_waypoint_height_observation_is_not_in_actor_or_critic():
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
    policy_cfg = source.split("class PolicyCfg", maxsplit=1)[1].split("policy: PolicyCfg", maxsplit=1)[0]
    critic_cfg = source.split("class CriticCfg", maxsplit=1)[1].split("critic: CriticCfg", maxsplit=1)[0]

    assert "waypoint_height_delta" not in policy_cfg
    assert "waypoint_height_delta" not in critic_cfg


def test_parkour_policy_matches_normal_himloco_terms():
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
    policy_cfg = source.split("class PolicyCfg", maxsplit=1)[1].split("policy: PolicyCfg", maxsplit=1)[0]

    for snippet in [
        "velocity_commands = ObsTerm(",
        "base_ang_vel = ObsTerm(",
        "projected_gravity = ObsTerm(",
        "joint_pos_rel = ObsTerm(",
        "joint_vel_rel = ObsTerm(",
        "last_action = ObsTerm(",
    ]:
        assert snippet in policy_cfg

    for snippet in [
        "height_scanner = ObsTerm(",
        "foot_contact_state = ObsTerm(",
        "terrain_route_one_hot = ObsTerm(",
        "waypoint_height_delta = ObsTerm(",
        "base_mass_com = ObsTerm(",
        "body_friction = ObsTerm(",
        "joint_stiffness_damping_scale = ObsTerm(",
    ]:
        assert snippet not in policy_cfg


def test_parkour_critic_matches_normal_himloco_terms():
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
    critic_cfg = source.split("class CriticCfg", maxsplit=1)[1].split("critic: CriticCfg", maxsplit=1)[0]

    for snippet in [
        "velocity_commands = ObsTerm(",
        "base_ang_vel = ObsTerm(",
        "projected_gravity = ObsTerm(",
        "joint_pos_rel = ObsTerm(",
        "joint_vel_rel = ObsTerm(",
        "last_action = ObsTerm(",
        "base_lin_vel = ObsTerm(",
        "base_external_force = ObsTerm(",
    ]:
        assert snippet in critic_cfg

    for snippet in [
        "height_scanner = ObsTerm(",
        "func=mdp.height_scan_clip",
        "foot_contact_state = ObsTerm(",
        "terrain_route_one_hot = ObsTerm(",
        "waypoint_height_delta = ObsTerm(",
        "base_mass_com = ObsTerm(",
        "body_friction = ObsTerm(",
        "joint_stiffness_damping_scale = ObsTerm(",
    ]:
        assert snippet not in critic_cfg


def test_uika_observations_match_master_policy_and_critic_terms():
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
    policy_cfg = source.split("class PolicyCfg", maxsplit=1)[1].split("policy: PolicyCfg", maxsplit=1)[0]
    critic_cfg = source.split("class CriticCfg", maxsplit=1)[1].split("critic: CriticCfg", maxsplit=1)[0]

    for snippet in [
        "velocity_commands = ObsTerm(",
        "base_ang_vel = ObsTerm(",
        "projected_gravity = ObsTerm(",
        "joint_pos_rel = ObsTerm(",
        "joint_vel_rel = ObsTerm(",
        "last_action = ObsTerm(",
    ]:
        assert snippet in policy_cfg

    for snippet in [
        "height_scanner = ObsTerm(",
        "foot_contact_state = ObsTerm(",
        "terrain_route_one_hot = ObsTerm(",
        "waypoint_height_delta = ObsTerm(",
        "base_mass_com = ObsTerm(",
        "body_friction = ObsTerm(",
        "joint_stiffness_damping_scale = ObsTerm(",
    ]:
        assert snippet not in policy_cfg

    assert "base_lin_vel = ObsTerm(" in critic_cfg
    assert "base_external_force = ObsTerm(" in critic_cfg

    for snippet in [
        "height_scanner = ObsTerm(",
        "func=mdp.height_scan_clip",
        "foot_contact_state = ObsTerm(",
        "terrain_route_one_hot = ObsTerm(",
        "waypoint_height_delta = ObsTerm(",
        "base_mass_com = ObsTerm(",
        "body_friction = ObsTerm(",
        "joint_stiffness_damping_scale = ObsTerm(",
    ]:
        assert snippet not in critic_cfg

from __future__ import annotations

import ast
import importlib.util
import sys
import types
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
EVENTS_PATH = ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/mdp/events.py"
UIKA_ASSET_PATH = ROOT / "source/himloco_lab/himloco_lab/assets/uika.py"
DELAYED_MOTOR_PATH = ROOT / "source/himloco_lab/himloco_lab/assets/delayed_motor.py"
VELOCITY_CFG_PATH = (
    ROOT / "source/himloco_lab/himloco_lab/tasks/locomotion/robots/uika/velocity_env_cfg.py"
)

EXPECTED_RANGES = {
    "hip": {
        "armature": (0.012499551, 0.014251016),
        "viscous_friction": (0.000871807, 0.002837807),
        "friction": (0.005748361, 0.018038273),
    },
    "thigh": {
        "armature": (0.012442659, 0.014217399),
        "viscous_friction": (0.001674354, 0.003362268),
        "friction": (0.009873092, 0.022234440),
    },
    "calf": {
        "armature": (0.022163186, 0.024222745),
        "viscous_friction": (0.001198500, 0.002290815),
        "friction": (0.007793665, 0.015262470),
    },
}

EXPECTED_ACTUATORS = [
    f"{leg}_{joint_type}"
    for joint_type in ("hip", "thigh", "calf")
    for leg in ("FL", "FR", "RL", "RR")
]


def _load_uika_asset_module(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    sim = types.ModuleType("isaaclab.sim")
    assets = types.ModuleType("isaaclab.assets")
    articulation = types.ModuleType("isaaclab.assets.articulation")
    utils = types.ModuleType("isaaclab.utils")
    himloco_lab = types.ModuleType("himloco_lab")
    himloco_assets = types.ModuleType("himloco_lab.assets")
    delayed_motor = types.ModuleType("himloco_lab.assets.delayed_motor")

    class FakeCfg:
        def __init__(self, **kwargs):
            for name, value in kwargs.items():
                setattr(self, name, value)

    class FakeArticulationCfg(FakeCfg):
        InitialStateCfg = FakeCfg

    class FakeJointDriveCfg(FakeCfg):
        PDGainsCfg = FakeCfg

    class FakeUrdfConverterCfg:
        JointDriveCfg = FakeJointDriveCfg

    sim.UrdfFileCfg = FakeCfg
    sim.UrdfConverterCfg = FakeUrdfConverterCfg
    sim.ArticulationRootPropertiesCfg = FakeCfg
    sim.RigidBodyPropertiesCfg = FakeCfg
    articulation.ArticulationCfg = FakeArticulationCfg
    utils.configclass = lambda cls: cls
    delayed_motor.DelayedDCMotorCfg = FakeCfg

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.sim", sim)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.assets.articulation", articulation)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "himloco_lab", himloco_lab)
    monkeypatch.setitem(sys.modules, "himloco_lab.assets", himloco_assets)
    monkeypatch.setitem(sys.modules, "himloco_lab.assets.delayed_motor", delayed_motor)

    spec = importlib.util.spec_from_file_location("uika_asset_under_test", UIKA_ASSET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_delayed_motor_module(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    actuators = types.ModuleType("isaaclab.actuators")
    utils = types.ModuleType("isaaclab.utils")
    utils_types = types.ModuleType("isaaclab.utils.types")

    class FakeDCMotor:
        def __init__(self, cfg, *args, **kwargs):
            self.cfg = cfg
            self._num_envs = kwargs["num_envs"]
            self._device = kwargs["device"]
            self.reset_env_ids = None
            self.super_compute_input = None
            self.computed_effort = None
            self.applied_effort = None

        def reset(self, env_ids):
            self.reset_env_ids = env_ids

        def compute(self, control_action, joint_pos, joint_vel):
            self.super_compute_input = (
                control_action.joint_positions.clone(),
                control_action.joint_velocities.clone(),
                control_action.joint_efforts.clone(),
            )
            self.computed_effort = control_action.joint_efforts.clone()
            self.applied_effort = self.computed_effort.clone()
            return control_action

    class FakeDCMotorCfg:
        pass

    class FakeDelayBuffer:
        instances = []

        def __init__(self, history_length, batch_dim, device):
            self.history_length = history_length
            self.batch_dim = batch_dim
            self.device = device
            self.offset = len(self.instances) + 1
            self.time_lags = None
            self.time_lag_env_ids = None
            self.reset_env_ids = None
            self.instances.append(self)

        def set_time_lag(self, time_lags, env_ids):
            self.time_lags = time_lags.clone()
            self.time_lag_env_ids = env_ids

        def reset(self, env_ids):
            self.reset_env_ids = env_ids

        def compute(self, value):
            return value + self.offset

    actuators.DCMotor = FakeDCMotor
    actuators.DCMotorCfg = FakeDCMotorCfg
    utils.DelayBuffer = FakeDelayBuffer
    utils.configclass = lambda cls: cls
    utils_types.ArticulationActions = object

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.actuators", actuators)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.types", utils_types)

    spec = importlib.util.spec_from_file_location("uika_delayed_motor_under_test", DELAYED_MOTOR_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, FakeDelayBuffer


def _load_events_module(monkeypatch):
    isaaclab = types.ModuleType("isaaclab")
    assets = types.ModuleType("isaaclab.assets")
    envs = types.ModuleType("isaaclab.envs")
    managers = types.ModuleType("isaaclab.managers")
    utils = types.ModuleType("isaaclab.utils")
    math_utils = types.ModuleType("isaaclab.utils.math")

    assets.Articulation = object
    assets.RigidObject = object
    envs.ManagerBasedEnv = object
    class SceneEntityCfg:
        def __init__(self, name):
            self.name = name

    managers.SceneEntityCfg = SceneEntityCfg
    utils.math = math_utils

    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.assets", assets)
    monkeypatch.setitem(sys.modules, "isaaclab.envs", envs)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", math_utils)

    spec = importlib.util.spec_from_file_location("uika_pace_events_under_test", EVENTS_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeArticulation:
    def __init__(self):
        self.device = "cpu"
        self.armature_write = None
        self.friction_write = None

    def write_joint_armature_to_sim(self, armature, *, joint_ids, env_ids):
        self.armature_write = (armature.clone(), joint_ids, env_ids.clone())

    def write_joint_friction_coefficient_to_sim(
        self,
        joint_friction_coeff,
        joint_dynamic_friction_coeff=None,
        joint_viscous_friction_coeff=None,
        *,
        joint_ids,
        env_ids,
    ):
        self.friction_write = (
            joint_friction_coeff.clone(),
            None if joint_dynamic_friction_coeff is None else joint_dynamic_friction_coeff.clone(),
            joint_viscous_friction_coeff.clone(),
            joint_ids,
            env_ids.clone(),
        )


def test_group_randomization_samples_each_of_four_joints_independently(monkeypatch):
    events = _load_events_module(monkeypatch)
    articulation = _FakeArticulation()
    env = types.SimpleNamespace(scene={"robot": articulation}, num_envs=8, device="cpu")
    env_ids = torch.tensor([1, 3, 6])
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[0, 3, 6, 9])

    torch.manual_seed(7)
    events.randomize_actuator_group_parameters(
        env,
        env_ids,
        asset_cfg=asset_cfg,
        **EXPECTED_RANGES["hip"],
    )

    armature, armature_joint_ids, armature_env_ids = articulation.armature_write
    static_friction, dynamic_friction, viscous_friction, friction_joint_ids, friction_env_ids = (
        articulation.friction_write
    )

    assert armature.shape == static_friction.shape == viscous_friction.shape == (3, 4)
    assert torch.all(torch.any(armature != armature[:, :1], dim=1))
    assert torch.all(torch.any(static_friction != static_friction[:, :1], dim=1))
    assert torch.all(torch.any(viscous_friction != viscous_friction[:, :1], dim=1))
    assert EXPECTED_RANGES["hip"]["armature"][0] <= armature.min()
    assert armature.max() <= EXPECTED_RANGES["hip"]["armature"][1]
    assert EXPECTED_RANGES["hip"]["friction"][0] <= static_friction.min()
    assert static_friction.max() <= EXPECTED_RANGES["hip"]["friction"][1]
    assert EXPECTED_RANGES["hip"]["viscous_friction"][0] <= viscous_friction.min()
    assert viscous_friction.max() <= EXPECTED_RANGES["hip"]["viscous_friction"][1]
    assert dynamic_friction is not None
    assert torch.equal(dynamic_friction, static_friction)
    assert armature_joint_ids == friction_joint_ids == [0, 3, 6, 9]
    assert torch.equal(armature_env_ids, env_ids)
    assert torch.equal(friction_env_ids, env_ids)


def test_group_randomization_expands_none_to_all_environment_ids(monkeypatch):
    events = _load_events_module(monkeypatch)
    articulation = _FakeArticulation()
    env = types.SimpleNamespace(scene={"robot": articulation}, num_envs=5, device="cpu")
    asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[2, 5, 8, 11])

    events.randomize_actuator_group_parameters(
        env,
        None,
        asset_cfg=asset_cfg,
        **EXPECTED_RANGES["calf"],
    )

    armature, _, env_ids = articulation.armature_write
    assert armature.shape == (5, 4)
    assert torch.equal(env_ids, torch.arange(5))


def test_velocity_config_uses_report_ranges_without_encoder_bias():
    tree = ast.parse(VELOCITY_CFG_PATH.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "UIKA_PACE_ACTUATOR_RANGES"
    }

    assert assignments["UIKA_PACE_ACTUATOR_RANGES"] == EXPECTED_RANGES
    source = VELOCITY_CFG_PATH.read_text(encoding="utf-8")
    for group_name in EXPECTED_RANGES:
        assert f"randomize_pace_{group_name}" in source
    assert "randomize_pace_action_bias" not in source
    assert "UIKA_PACE_BIAS" not in source


def test_uika_asset_declares_twelve_single_joint_actuators(monkeypatch):
    uika = _load_uika_asset_module(monkeypatch)

    assert list(uika.UIKA_CFG.actuators) == EXPECTED_ACTUATORS
    assert len(uika.UIKA_CFG.actuators) == 12
    for actuator_name, actuator_cfg in uika.UIKA_CFG.actuators.items():
        assert actuator_cfg.joint_names_expr == [f"{actuator_name}_joint"]


def test_delayed_dc_motor_delays_commands_before_dc_motor_compute(monkeypatch):
    delayed_motor, fake_delay_buffer = _load_delayed_motor_module(monkeypatch)
    cfg = types.SimpleNamespace(min_delay=2, max_delay=3)
    motor = delayed_motor.DelayedDCMotor(cfg, num_envs=4, device="cpu")
    env_ids = torch.tensor([1, 3])

    torch.manual_seed(3)
    motor.reset(env_ids)

    assert len(fake_delay_buffer.instances) == 3
    time_lags = [buffer.time_lags for buffer in fake_delay_buffer.instances]
    assert all(torch.equal(time_lags[0], time_lag) for time_lag in time_lags[1:])
    assert torch.all((time_lags[0] >= 2) & (time_lags[0] <= 3))
    assert all(torch.equal(buffer.time_lag_env_ids, env_ids) for buffer in fake_delay_buffer.instances)
    assert all(torch.equal(buffer.reset_env_ids, env_ids) for buffer in fake_delay_buffer.instances)

    action = types.SimpleNamespace(
        joint_positions=torch.zeros((4, 4)),
        joint_velocities=torch.full((4, 4), 10.0),
        joint_efforts=torch.full((4, 4), 20.0),
    )
    motor.compute(action, torch.zeros((4, 4)), torch.zeros((4, 4)))

    position_command, velocity_command, effort_command = motor.super_compute_input
    assert torch.equal(position_command, torch.full((4, 4), 1.0))
    assert torch.equal(velocity_command, torch.full((4, 4), 12.0))
    assert torch.equal(effort_command, torch.full((4, 4), 23.0))
    assert torch.equal(motor.applied_effort, effort_command)


def test_uika_actuators_use_two_to_three_step_delay_with_nominal_kd_1_5(monkeypatch):
    uika = _load_uika_asset_module(monkeypatch)

    for actuator_cfg in uika.UIKA_CFG.actuators.values():
        assert actuator_cfg.min_delay == 2
        assert actuator_cfg.max_delay == 3
        assert actuator_cfg.damping == 1.5


def test_play_configs_disable_pace_randomization_and_use_fixed_delay():
    source = VELOCITY_CFG_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    classes = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef)
    }

    robot_env_source = classes["RobotEnvCfg"]
    for event_name in ("randomize_pace_hip", "randomize_pace_thigh", "randomize_pace_calf"):
        assert f"self.events.{event_name} = None" in robot_env_source
    assert "actuator.min_delay = 2" in robot_env_source
    assert "actuator.max_delay = 2" in robot_env_source

    for play_class in ("RobotPlayEnvCfg", "RobotParkourPlayEnvCfg"):
        assert "self._configure_pace_actuators_for_play()" in classes[play_class]

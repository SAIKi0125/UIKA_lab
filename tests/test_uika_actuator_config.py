import xml.etree.ElementTree as ET
from pathlib import Path


def test_uika_actuators_use_delayed_dc_motor_with_may31_constants():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "uika.py"
    ).read_text()

    assert "UIKA_JOINT_ARMATURE" not in source
    assert "UIKA_JOINT_VISCOUS_DAMPING" not in source
    assert "UIKA_JOINT_FRICTION" not in source
    assert "UIKA_ENCODER_BIAS" not in source
    assert "PaceDCMotorCfg" not in source
    assert "DelayedDCMotorCfg" in source
    assert "min_delay=5" in source
    assert "max_delay=7" in source
    assert "delay_scale_range" not in source
    assert "damping=1.5" in source
    assert "armature=0.0042" in source
    assert "friction=0.0" in source


def test_delayed_dc_motor_keeps_dc_motor_semantics_and_delays_commands():
    source = (
        Path(__file__).resolve().parents[1]
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "delayed_motor.py"
    ).read_text()
    assert "class DelayedDCMotor(DCMotor)" in source
    assert "class DelayedDCMotorCfg(DCMotorCfg)" in source
    assert "class_type: type = DelayedDCMotor" in source
    assert "torques_delay_buffer" not in source
    assert "positions_delay_buffer.compute(control_action.joint_positions)" in source
    assert "velocities_delay_buffer.compute(control_action.joint_velocities)" in source
    assert "efforts_delay_buffer.compute(control_action.joint_efforts)" in source


def test_uika_training_command_range_matches_may31_config():
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

    assert "lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-1.0, 1.0)" in source


def test_uika_uses_original_collision_urdf():
    repo_root = Path(__file__).resolve().parents[1]
    uika_source = (repo_root / "source" / "himloco_lab" / "himloco_lab" / "assets" / "uika.py").read_text()
    urdf_path = (
        repo_root
        / "source"
        / "himloco_lab"
        / "himloco_lab"
        / "assets"
        / "uika"
        / "urdf"
        / "uika.urdf"
    )

    assert "uika.urdf" in uika_source
    assert "uika_simple_collision.urdf" not in uika_source

    root = ET.parse(urdf_path).getroot()
    collision_geometries = [collision.find("geometry") for collision in root.findall(".//collision")]
    assert collision_geometries
    assert all(geometry.find("mesh") is not None for geometry in collision_geometries)

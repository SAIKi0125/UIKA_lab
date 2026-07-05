from pathlib import Path


def test_uika_actuators_keep_delay_but_match_may31_motor_constants():
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
    assert "max_delay=UIKA_MOTOR_DELAY_STEPS" in source
    assert "delay_scale_range=UIKA_MOTOR_DELAY_SCALE_RANGE" in source
    assert "damping=1.0" in source
    assert "armature=0.0042" in source
    assert "friction=0.0" in source


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

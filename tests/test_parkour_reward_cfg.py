from pathlib import Path


def test_parkour_rewards_cfg_uses_source_parkour_regularizers_with_velocity_tracking():
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
    parkour_cfg = source.split("class ParkourRewardsCfg", maxsplit=1)[1].split(
        "@configclass\nclass TerminationsCfg", maxsplit=1
    )[0]

    expected_snippets = [
        "reward_collision = RewTerm(",
        "weight=-10.0",
        '"threshold": 0.1',
        'body_names=["base", ".*_calf", ".*_thigh"]',
        "reward_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)",
        "reward_dof_error = RewTerm(",
        "func=mdp.joint_dof_error_l2",
        "weight=-0.04",
        "reward_hip_pos = RewTerm(",
        "func=mdp.hip_pos_l2",
        "weight=-0.5",
        "reward_action_rate = RewTerm(",
        "func=mdp.ActionRateNorm",
        "weight=-0.1",
        '"action_term_name": "JointPositionAction"',
        "reward_dof_acc = RewTerm(",
        "func=mdp.JointDofAccL2",
        "weight=-2.5e-7",
        "track_lin_vel_xy = RewTerm(",
        "func=mdp.track_lin_vel_xy_exp",
        "weight=1.5",
        "track_ang_vel_z = RewTerm(",
        "func=mdp.track_ang_vel_z_exp",
        "weight=0.5",
        "reward_delta_torques = RewTerm(",
        "func=mdp.DeltaTorquesL2",
        "weight=-1.0e-7",
        "feet_stumble = RewTerm(",
        "weight=-1.0",
    ]

    for snippet in expected_snippets:
        assert snippet in parkour_cfg

    assert "reward_tracking_goal_vel" not in parkour_cfg
    assert "track_goal_vel_from_command" not in parkour_cfg

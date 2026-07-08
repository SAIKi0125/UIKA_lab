from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "source" / "himloco_lab" / "himloco_lab"


def _read(relative: str) -> str:
    return (SRC / relative).read_text()


def test_uika_takeoff_tasks_are_registered_with_standard_rsl_rl_runner():
    source = _read("tasks/locomotion/robots/uika/__init__.py")

    assert 'id="UIKA-Takeoff"' in source
    assert 'id="UIKA-Takeoff-Play"' in source
    assert '"env_cfg_entry_point": f"{__name__}.takeoff_env_cfg:TakeoffEnvCfg"' in source
    assert '"env_cfg_entry_point": f"{__name__}.takeoff_env_cfg:TakeoffPlayEnvCfg"' in source
    assert "himloco_rsl_rl_cfg" not in source.split('id="UIKA-Takeoff"')[1].split(")", 1)[0]
    assert (
        '"rsl_rl_cfg_entry_point": '
        'f"himloco_lab.tasks.locomotion.agents.rsl_rl_takeoff_cfg:UIKATakeoffPPORunnerCfg"'
    ) in source


def test_spring_jump_command_uses_single_jump_flag_semantics():
    source = _read("tasks/locomotion/mdp/commands/commands.py")
    cfg_source = _read("tasks/locomotion/mdp/commands/commands_cfg.py")

    assert "class SpringJumpCommand(" in source
    assert "self._command = torch.zeros(self.num_envs, 1" in source
    assert "jump_flag" in source
    assert "target_x" not in source.split("class SpringJumpCommand", 1)[1]
    assert "target_y" not in source.split("class SpringJumpCommand", 1)[1]
    assert "self._command[env_ids, 0] = 0.0" in source
    assert "self._command[active, 0] = 1.0" in source
    assert "self._command[env_ids, 1]" not in source
    assert "self._command[env_ids, 2]" not in source
    assert "ang_vel_z" not in source.split("class SpringJumpCommand", 1)[1]
    assert "class SpringJumpCommandCfg" in cfg_source
    assert "class Ranges" not in cfg_source
    assert "target_x: tuple[float, float]" not in cfg_source
    assert "target_y: tuple[float, float]" not in cfg_source
    assert "setting_frame_range: tuple[int, int] = (50, 60)" in cfg_source


def test_uika_takeoff_env_has_actor_history_critic_obs_and_spring_jump_rewards():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")

    assert "class TakeoffEnvCfg" in source
    assert "episode_length_s = 5.0" in source
    assert "base_velocity = mdp.SpringJumpCommandCfg" in source
    assert "actor_history: ActorHistoryCfg = ActorHistoryCfg()" in source
    assert "class ActorHistoryCfg" in source
    assert "history_length = 10" in source
    assert "privileged_target: PrivilegedTargetCfg = PrivilegedTargetCfg()" not in source
    assert "class PrivilegedTargetCfg" not in source
    assert "func=mdp.base_lin_vel" in source
    critic_source = source.split("class CriticCfg", 1)[1].split("critic:", 1)[0]
    expected_critic_order = [
        "velocity_commands = ObsTerm(",
        "joint_pos_rel = ObsTerm(",
        "joint_pos_abs = ObsTerm(",
        "joint_vel_rel = ObsTerm(",
        "last_action = ObsTerm(",
        "base_lin_vel = ObsTerm(",
        "base_ang_vel = ObsTerm(",
        "base_euler_xyz = ObsTerm(",
        "contact_mask = ObsTerm(",
        "has_jumped = ObsTerm(",
    ]
    indices = [critic_source.index(term) for term in expected_critic_order]
    assert indices == sorted(indices)
    assert "landing_xy_from_takeoff" not in critic_source
    assert "landing_xy_from_start" not in critic_source
    assert "base_height = ObsTerm(" not in critic_source
    assert "projected_gravity = ObsTerm(" not in critic_source

    for reward_name in [
        "before_setting",
        "line_z",
        "flight",
        "base_height_flight",
        "land_pos",
        "tracking_lin_vel_jump",
        "line_vel_stance",
        "foot_clearance_jump",
    ]:
        assert f"{reward_name} = RewTerm(" in source
    assert "successful_jump_sj" not in source
    assert "line_vel_x_setting = RewTerm(" not in source
    assert "line_vel_y_setting = RewTerm(" not in source
    assert "dof_pos_penalty_prepare_sj = RewTerm(" not in source


def test_uika_takeoff_land_pos_uses_fixed_target_without_command_xy_or_obs():
    reward_source = _read("tasks/locomotion/mdp/spring_jump.py")
    reset_source = reward_source.split("def spring_jump_state_reset", 1)[1].split("def is_too_low", 1)[0]
    land_pos_source = reward_source.split("def land_pos", 1)[1].split("def successful_jump_sj", 1)[0]
    env_source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    land_pos_term_source = env_source.split("land_pos = RewTerm(", 1)[1].split("tracking_lin_vel_jump", 1)[0]

    assert "env._sj_init_xy[env_ids] = asset.data.root_pos_w[env_ids, :2]" in reset_source
    assert "target_xy = env._sj_init_xy + target_offset" in land_pos_source
    assert "cmd[:, :2]" not in land_pos_source
    assert "target_xy = env._sj_takeoff_xy + cmd[:, :2]" not in land_pos_source
    assert '"target_xy": (1.0, 0.0)' in land_pos_term_source
    assert "landing_xy_from_takeoff = ObsTerm(" not in env_source
    assert "landing_xy_from_start = ObsTerm(" not in env_source


def test_uika_takeoff_uses_original_spring_jump_generic_penalties():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    reward_source = _read("tasks/locomotion/mdp/spring_jump.py")

    removed_terms = [
        "\n    successful_jump_sj = RewTerm(",
        "\n    joint_torques_l2 = RewTerm(",
        "\n    joint_acc_l2 = RewTerm(",
        "\n    undesired_contacts = RewTerm(",
        "\n    contact_forces = RewTerm(",
    ]
    for term in removed_terms:
        assert term not in source

    expected_terms = {
        "ang_vel_xy": "-0.2",
        "torques": "-0.0001",
        "joint_pos_limits": "-10.0",
        "dof_vel_limits": "-1.0",
        "dof_vel": "-0.001",
        "collision": "-50.0",
        "action_rate_l2": "-0.01",
        "feet_contact_forces": "-0.1",
    }
    for term, weight in expected_terms.items():
        term_source = source.split(f"{term} = RewTerm(", 1)[1].split(")", 1)[0]
        assert f"weight={weight}" in term_source

    for func_name in [
        "def spring_jump_ang_vel_xy",
        "def spring_jump_torques",
        "def spring_jump_dof_vel_limits",
        "def spring_jump_dof_vel",
        "def spring_jump_collision",
        "def spring_jump_feet_contact_forces",
    ]:
        assert func_name in reward_source


def test_uika_takeoff_actor_history_has_no_dummy_ball_and_command_is_last_for_plain_actor():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    obs_source = _read("tasks/locomotion/mdp/observations.py")
    runner_source = _read("tasks/locomotion/agents/rsl_rl_takeoff_cfg.py")

    policy_source = source.split("class PolicyCfg", 1)[1].split("def __post_init__", 1)[0]
    expected_order = [
        "base_ang_vel = ObsTerm(",
        "projected_gravity = ObsTerm(",
        "joint_pos_rel = ObsTerm(",
        "joint_vel_rel = ObsTerm(",
        "last_action = ObsTerm(",
        "velocity_commands = ObsTerm(",
    ]
    indices = [policy_source.index(term) for term in expected_order]

    assert indices == sorted(indices)
    assert "dummy_ball" not in policy_source
    assert "zero_observation" not in obs_source
    assert "RslRlMLPModelCfg" in runner_source
    assert "RslRlMlpAdaptModelCfg" not in runner_source
    assert "takeoff_adapt_model" not in runner_source
    assert "ball_dim" not in runner_source
    assert "catch_target_dim" not in runner_source


def test_uika_takeoff_uses_robot_default_pose_for_action_zero_reset_and_before_setting():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    asset_source = _read("assets/uika.py")

    assert "TAKEOFF_CROUCH_JOINT_ANGLES" not in source
    assert "TAKEOFF_ROBOT_CFG" not in source
    assert "reset_to_crouch_pose" not in source
    assert 'robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")' in source
    assert '"FL_thigh_joint": 0.05' in asset_source
    assert '"FL_calf_joint": 0.70' in asset_source
    before_setting_source = source.split("before_setting = RewTerm(", 1)[1].split("line_z", 1)[0]
    assert "target_joint_angles" not in before_setting_source


def test_uika_takeoff_setting_height_is_confirmed_pre_jump_height():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")

    assert '"target_height": 0.50' in source
    assert '"stance_target": 0.3357' in source
    assert '"setting_target": 0.24' in source


def test_uika_takeoff_terminations_match_spring_jump():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    term_source = source.split("class TerminationsCfg", 1)[1].split("class CurriculumCfg", 1)[0]

    assert "time_out = DoneTerm(func=mdp.time_out, time_out=True)" in term_source
    assert 'too_low = DoneTerm(func=mdp.is_too_low, params={"asset_cfg": SceneEntityCfg("robot"), "threshold": 0.15})' in term_source
    assert '"threshold": 1.0' in term_source
    assert 'body_names="base"' in term_source


def test_uika_takeoff_push_matches_springjump_and_is_disabled_in_play():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    reward_source = _read("tasks/locomotion/mdp/spring_jump.py")
    event_source = source.split("spring_jump_update = EventTerm(", 1)[1].split("class CommandsCfg", 1)[0]
    play_source = source.split("class TakeoffPlayEnvCfg", 1)[1]

    assert reward_source.count("push_vel_z_range: tuple[float, float] = (1.5, 2.2)") == 2
    assert '"push_vel_z_range": (1.5, 2.2)' in event_source
    assert '"push_initial_prob": 0.8' in event_source
    assert '"push_decay_steps": 1200' in event_source
    assert "current_prob = max(8 - int(step_count / float(push_decay_steps)), 0) / 10.0" in reward_source
    assert 'self.events.spring_jump_update.params["push_initial_prob"] = 0.0' in play_source


def test_uika_takeoff_uses_same_delayed_motor_asset_as_velocity():
    takeoff_source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    velocity_source = _read("tasks/locomotion/robots/uika/velocity_env_cfg.py")
    asset_source = _read("assets/uika.py")

    assert "from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG" in takeoff_source
    assert "from himloco_lab.assets.uika import UIKA_CFG as ROBOT_CFG" in velocity_source
    assert "TAKEOFF_ROBOT_CFG" not in takeoff_source
    assert 'robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")' in takeoff_source
    assert 'robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")' in velocity_source
    assert "DelayedDCMotorCfg(" in asset_source
    assert "min_delay=5" in asset_source
    assert "max_delay=7" in asset_source


def test_uika_takeoff_dof_pos_rewards_match_spring_jump_gates():
    env_source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    reward_source = _read("tasks/locomotion/mdp/spring_jump.py")
    landing_term_source = env_source.split("dof_pos_penalty_sj = RewTerm(", 1)[1].split(
        "dof_hip_pos_penalty_sj", 1
    )[0]
    landing_func_source = reward_source.split("def dof_pos_penalty_sj", 1)[1].split(
        "def dof_hip_pos_penalty_sj", 1
    )[0]
    hip_term_source = env_source.split("dof_hip_pos_penalty_sj = RewTerm(", 1)[1].split("ang_vel_xy", 1)[0]
    hip_func_source = reward_source.split("def dof_hip_pos_penalty_sj", 1)[1].split("def spring_jump_ang_vel_xy", 1)[0]

    assert "dof_pos_penalty_prepare_sj = RewTerm(" not in env_source
    assert "weight=-0.1" in landing_term_source
    assert '"command_name": "base_velocity"' not in landing_term_source
    assert "env._sj_has_jumped.float()" not in landing_func_source
    assert "return err" in landing_func_source
    assert "weight=-1.0" in hip_term_source
    assert '"command_name": "base_velocity"' not in hip_term_source
    assert "jump_flag == 1.0" not in hip_func_source


def test_takeoff_runner_uses_plain_ppo_actor_critic_not_adaptation_network():
    source = _read("tasks/locomotion/agents/rsl_rl_takeoff_cfg.py")

    assert not (SRC / "tasks/locomotion/agents/takeoff_adapt_model.py").exists()
    assert not (SRC / "tasks/locomotion/agents/rsl_rl_takeoff_adapt_cfg.py").exists()
    assert "RslRlMLPModelCfg" in source
    assert "RslRlMlpAdaptModelCfg" not in source
    assert "class UIKATakeoffPPORunnerCfg" in source
    assert 'experiment_name = "uika_takeoff"' in source
    assert '"actor": ["actor_history"]' in source
    assert '"critic": ["critic"]' in source
    assert "privileged_target" not in source
    assert "adaptation_loss_coef" not in source
    assert "takeoff_adapt_model" not in source
    assert "HIMActorCritic" not in source
    assert "HIMPPO" not in source


def test_himloco_train_script_can_dispatch_standard_rsl_rl_for_takeoff():
    source = (REPO_ROOT / "scripts" / "himloco_rsl_rl" / "train.py").read_text()

    assert "RslRlVecEnvWrapper" in source
    assert "OnPolicyRunner" in source
    assert "isinstance(agent_cfg, HIMOnPolicyRunnerCfg)" in source
    assert "logs/rsl_rl" in source


def test_himloco_play_script_can_dispatch_standard_rsl_rl_for_takeoff():
    source = (REPO_ROOT / "scripts" / "himloco_rsl_rl" / "play.py").read_text()

    assert "RslRlVecEnvWrapper" in source
    assert "OnPolicyRunner" in source
    assert "isinstance(agent_cfg, HIMOnPolicyRunnerCfg)" in source
    assert "logs/rsl_rl" in source
    assert "Standard RSL-RL environment wrapped successfully" in source

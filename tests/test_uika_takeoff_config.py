from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "source" / "himloco_lab" / "himloco_lab"


def _read(relative: str) -> str:
    return (SRC / relative).read_text()


def test_uika_takeoff_tasks_are_registered_with_standard_rsl_rl_adaptation_runner():
    source = _read("tasks/locomotion/robots/uika/__init__.py")

    assert 'id="UIKA-Takeoff"' in source
    assert 'id="UIKA-Takeoff-Play"' in source
    assert '"env_cfg_entry_point": f"{__name__}.takeoff_env_cfg:TakeoffEnvCfg"' in source
    assert '"env_cfg_entry_point": f"{__name__}.takeoff_env_cfg:TakeoffPlayEnvCfg"' in source
    assert "himloco_rsl_rl_cfg" not in source.split('id="UIKA-Takeoff"')[1].split(")", 1)[0]
    assert (
        '"rsl_rl_cfg_entry_point": '
        'f"himloco_lab.tasks.locomotion.agents.rsl_rl_takeoff_adapt_cfg:UIKATakeoffPPOAdaptRunnerCfg"'
    ) in source


def test_spring_jump_command_uses_my_unitree_three_channel_target_jump_flag_semantics():
    source = _read("tasks/locomotion/mdp/commands/commands.py")
    cfg_source = _read("tasks/locomotion/mdp/commands/commands_cfg.py")

    assert "class SpringJumpCommand(" in source
    assert "self._command = torch.zeros(self.num_envs, 3" in source
    assert "target_x" in source
    assert "target_y" in source
    assert "jump_flag" in source
    assert "self._command[env_ids, 2] = 0.0" in source
    assert "self._command[active, 2] = 1.0" in source
    assert "ang_vel_z" not in source.split("class SpringJumpCommand", 1)[1]
    assert "class SpringJumpCommandCfg" in cfg_source
    assert "target_x: tuple[float, float] = (0.8, 1.2)" in cfg_source
    assert "target_y: tuple[float, float] = (0.0, 0.0)" in cfg_source
    assert "setting_frame_range: tuple[int, int] = (50, 60)" in cfg_source


def test_uika_takeoff_env_has_actor_history_privileged_target_and_spring_jump_rewards():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")

    assert "class TakeoffEnvCfg" in source
    assert "episode_length_s = 5.0" in source
    assert "base_velocity = mdp.SpringJumpCommandCfg" in source
    assert "actor_history: ActorHistoryCfg = ActorHistoryCfg()" in source
    assert "history_length = 10" in source
    assert "privileged_target: PrivilegedTargetCfg = PrivilegedTargetCfg()" in source
    assert "func=mdp.base_lin_vel" in source
    critic_source = source.split("class CriticCfg", 1)[1].split("critic:", 1)[0]
    assert "velocity_commands = ObsTerm(" in critic_source
    assert "func=mdp.generated_commands" in critic_source
    assert "func=mdp.landing_xy_from_start" in critic_source
    privileged_target_source = source.split("class PrivilegedTargetCfg", 1)[1].split("privileged_target:", 1)[0]
    assert "func=mdp.landing_xy_from_start" in privileged_target_source
    assert "func=mdp.base_height" not in privileged_target_source

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


def test_uika_takeoff_actor_history_has_no_dummy_ball_and_command_is_last_for_takeoff_adapt_model():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    obs_source = _read("tasks/locomotion/mdp/observations.py")
    runner_source = _read("tasks/locomotion/agents/rsl_rl_takeoff_adapt_cfg.py")
    model_source = _read("tasks/locomotion/agents/takeoff_adapt_model.py")

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
    assert "TakeoffMlpAdaptModel" in model_source
    assert "self.proprioception_dim = int(self.obs_per_step - self.cmd_dim)" in model_source
    assert "cmd = x[:, -1, self.proprioception_dim : self.proprioception_dim + self.cmd_dim]" in model_source
    assert "catch_estimator" not in model_source
    assert "cmd_dim=3" in runner_source
    assert 'class_name="himloco_lab.tasks.locomotion.agents.takeoff_adapt_model:TakeoffMlpAdaptModel"' in runner_source
    assert "ball_dim" not in runner_source
    assert "catch_target_dim" not in runner_source


def test_uika_takeoff_uses_confirmed_crouch_pose_for_reset_and_before_setting():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")

    expected_angles = {
        "FL_hip_joint": "-0.78",
        "FL_thigh_joint": "0.40",
        "FL_calf_joint": "0.20",
        "FR_hip_joint": "0.78",
        "FR_thigh_joint": "0.40",
        "FR_calf_joint": "0.20",
        "RL_hip_joint": "-0.78",
        "RL_thigh_joint": "-0.05",
        "RL_calf_joint": "0.20",
        "RR_hip_joint": "0.78",
        "RR_thigh_joint": "-0.05",
        "RR_calf_joint": "0.20",
    }

    for joint_name, angle in expected_angles.items():
        assert f'"{joint_name}": {angle}' in source
    assert '"target_joint_angles": TAKEOFF_CROUCH_JOINT_ANGLES' in source


def test_uika_takeoff_setting_height_is_confirmed_pre_jump_height():
    source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")

    assert '"setting_target": 0.24' in source


def test_uika_takeoff_dof_pos_rewards_use_reference_stage_gates():
    env_source = _read("tasks/locomotion/robots/uika/takeoff_env_cfg.py")
    reward_source = _read("tasks/locomotion/mdp/spring_jump.py")
    prepare_term_source = env_source.split("dof_pos_penalty_prepare_sj = RewTerm(", 1)[1].split(
        "dof_pos_penalty_sj", 1
    )[0]
    landing_term_source = env_source.split("dof_pos_penalty_sj = RewTerm(", 1)[1].split(
        "dof_hip_pos_penalty_sj", 1
    )[0]
    prepare_func_source = reward_source.split("def dof_pos_penalty_prepare_sj", 1)[1].split(
        "def dof_pos_penalty_sj", 1
    )[0]
    landing_func_source = reward_source.split("def dof_pos_penalty_sj", 1)[1].split(
        "def dof_hip_pos_penalty_sj", 1
    )[0]

    assert "weight=-3.0" in prepare_term_source
    assert '"command_name": "base_velocity"' in prepare_term_source
    assert '"target_joint_angles": TAKEOFF_CROUCH_JOINT_ANGLES' in prepare_term_source
    assert "jump_flag = env.command_manager.get_command(command_name)[:, 2]" in prepare_func_source
    assert "jump_flag == 0.0" in prepare_func_source
    assert "env._sj_has_jumped.float()" in landing_func_source
    assert '"command_name": "base_velocity"' in landing_term_source


def test_takeoff_adaptation_runner_uses_mlp_adapt_model_not_himloco_network():
    source = _read("tasks/locomotion/agents/rsl_rl_takeoff_adapt_cfg.py")

    assert "RslRlMlpAdaptModelCfg" in source
    assert "class UIKATakeoffPPOAdaptRunnerCfg" in source
    assert '"actor": ["actor_history"]' in source
    assert '"critic": ["critic"]' in source
    assert "cmd_dim=3" in source
    assert "max_length=10" in source
    assert "privileged_target_key=\"privileged_target\"" in source
    assert "privileged_target_dim=5" in source
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

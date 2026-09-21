user@qm:/mnt/c/bipedal_robot$ ./venv_wsl/bin/python -m pytest tests/ -v
================== test session starts ==================
platform linux -- Python 3.12.3, pytest-9.1.1, pluggy-1.6.0 -- /mnt/c/bipedal_robot/venv_wsl/bin/python
cachedir: .pytest_cache
rootdir: /mnt/c/bipedal_robot
collected 37 items                                      

tests/test_actuator_gains.py::test_robot_config_gains_are_applied_to_mjx_model PASSED [  2%]
tests/test_actuator_gains.py::test_robot_config_gains_are_applied_to_brax_system PASSED [  5%]
tests/test_improved_rewards.py::test_curriculum_learning PASSED [  8%]
tests/test_improved_rewards.py::test_stability_metrics PASSED [ 10%]
tests/test_improved_rewards.py::test_adaptive_scaling PASSED [ 13%]
tests/test_improved_rewards.py::test_stance_penalty_discourages_wide_foot_spacing PASSED [ 16%]
tests/test_phase0_eval_diagnostics.py::test_classify_termination_reason_priority PASSED [ 18%]
tests/test_phase0_eval_diagnostics.py::test_classify_combined_reasons PASSED [ 21%]
tests/test_phase0_eval_diagnostics.py::test_kaplan_meier_survival_basic PASSED [ 24%]
tests/test_phase0_eval_diagnostics.py::test_kaplan_meier_survival_empty PASSED [ 27%]
tests/test_phase0_eval_diagnostics.py::test_diagnose_failure_timing_early_concentration PASSED [ 29%]
tests/test_phase0_eval_diagnostics.py::test_diagnose_failure_timing_empty PASSED [ 32%]
tests/test_phase0_eval_diagnostics.py::test_summarize_episode_alive_empty PASSED [ 35%]
tests/test_phase0_eval_diagnostics.py::test_summarize_episode_alive_basic PASSED [ 37%]
tests/test_physical_mass_contract.py::test_visual_geoms_have_no_inertia_mass PASSED [ 40%]
tests/test_physical_mass_contract.py::test_model_total_mass_is_within_design_bound PASSED [ 43%]
tests/test_policy_bounds.py::test_policy_distribution_bounds PASSED [ 45%]
tests/test_policy_bounds.py::test_policy_network_forward_pass_bounds PASSED [ 48%]
tests/test_policy_bounds.py::test_entrypoints_import_shared_policy_factory PASSED [ 51%]
tests/test_sensor_contract.py::test_sensor_order_contract_matches_xml_layout PASSED [ 54%]
tests/test_sensor_contract.py::test_fsr_positions_match_left_then_right_xml_layout PASSED [ 56%]
tests/test_sensor_contract.py::test_joint_order_matches_actuator_qpos_contract PASSED [ 59%]
tests/test_sensor_contract.py::test_domain_randomization_is_applied_to_physics_model PASSED [ 62%]
tests/test_sensor_contract.py::test_domain_randomization_torque_uses_actuator_qvel_mapping PASSED [ 64%]
tests/test_standing_only.py::test_standing_only_constraints PASSED [ 67%]
tests/test_standing_requirements.py::test_standing_mission_forbids_walking_and_stepping PASSED [ 70%]
tests/test_standing_requirements.py::test_success_summary_is_not_episode_alive_only PASSED [ 72%]
tests/test_teensy_telemetry.py::test_valid_telemetry_updates_values PASSED [ 75%]
tests/test_teensy_telemetry.py::test_out_of_range_telemetry_keeps_last_good_values PASSED [ 78%]
tests/test_teensy_telemetry.py::test_three_rejected_packets_raise_timeout_flag PASSED [ 81%]
tests/test_training_eval_contract.py::test_reset_training_progress_is_full_for_direct_eval PASSED [ 83%]
tests/test_training_eval_contract.py::test_direct_eval_policies_are_not_zeroed_by_progress_scale PASSED [ 86%]
tests/test_training_integration.py::test_vmap_reset_then_step_no_crash PASSED [ 89%]
tests/test_training_integration.py::test_domain_randomization_differs_per_env_under_vmap PASSED [ 91%]
tests/test_training_integration.py::test_auto_reset_resets_episode_scoped_info PASSED [ 94%]
tests/test_training_wrapper.py::test_training_progress_wrapper_counters PASSED [ 97%]
tests/test_training_wrapper.py::test_training_progress_wrapper_progress_saturates PASSED [100%]

=================== warnings summary ====================
venv_wsl/lib/python3.12/site-packages/jaxopt/__init__.py:59
  /mnt/c/bipedal_robot/venv_wsl/lib/python3.12/site-packages/jaxopt/__init__.py:59: DeprecationWarning: JAXopt is no longer maintained. See https://docs.jax.dev/en/latest/ for alternatives.
    warnings.warn(

tests/test_actuator_gains.py: 2 warnings
tests/test_sensor_contract.py: 4 warnings
tests/test_training_eval_contract.py: 2 warnings
tests/test_training_integration.py: 3 warnings
  /mnt/c/bipedal_robot/venv_wsl/lib/python3.12/site-packages/brax/io/mjcf.py:480: UserWarning: Brax System, piplines and environments are not actively being maintained. Please see MJX for a well maintained JAX-based physics engine: https://github.com/google-deepmind/mujoco/tree/main/mjx. For a host of environments that use MJX, see: https://github.com/google-deepmind/mujoco_playground.
    warnings.warn(

tests/test_training_eval_contract.py::test_reset_training_progress_is_full_for_direct_eval
tests/test_training_eval_contract.py::test_direct_eval_policies_are_not_zeroed_by_progress_scale
tests/test_training_integration.py::test_vmap_reset_then_step_no_crash
tests/test_training_integration.py::test_domain_randomization_differs_per_env_under_vmap
tests/test_training_integration.py::test_auto_reset_resets_episode_scoped_info
  /mnt/c/bipedal_robot/venv_wsl/lib/python3.12/site-packages/jax/_src/core.py:687: RuntimeWarning: overflow encountered in cast
    c_arg = dtypes.canonicalize_value(arg)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====== 37 passed, 17 warnings in 371.57s (0:06:11) ======
user@qm:/mnt/c/bipedal_robot$ 
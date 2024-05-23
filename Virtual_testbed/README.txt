./Hyperparameter_testing/: includes results of hyperparameter searches, python files for plotting
algorithm: testing out different algorithms (Fig. 2b, 12)
CustomTensorboardCallback: Callback for tensorboard
Env_fiber_simulated_goal:gymnasium environment for virtual testbed
min_max_positions: minimum and maximum actuator position over which scan was done to create virtual testbed
MovePowerUp: Functions called to move to a higher power during reset in virtual testbed
param_fit: Fit parameters for virtual testbed creation
reset_methods_comparison: testing out power and actuator position probabilities of different reset methods (Fig. 9b)
TQC_one_run: Training TQC on virtual testbed
TQC_different_obs: testing leaving out parts of observation (Fig. 2a)
TQC_episode_length: testing out different episode legths (Fig. 8)
TQC_episode_length_and_max_action: testing out different episode legths and max actions (Fig. 8)
TQC_goal_linear_function: testing out increasing goal power linearly over time (Fig. 10b)
TQC_goal_step_function: testing out increasing goal power in steps over time (Fig. 10b)
TQC_max_action: testing out different max actions (Fig. 11b)
TQC_number_obs: testing out different history lengths (Fig. 11a)
TQC_P_goal: testing out different goal powers (Fig. 10a) 
TQC_reward_alpha_beta_fail_optimization: testing out different alphas and betas for fail reward (Fig. 7)
TQC_reward_alpha_beta_goal_optimization: testing out different alphas and betas for goal reward (Fig. 6)
TQC_reward_alpha_beta_step_optimization: testing out different alphas and betas for step reward (Fig. 5)
TQC_reward_prefactor_optimization: testing out different prefactors for reward (Fig. 4)
TQC_test_resets_diff_models: getting power and actuator distribution after reset for different resets (Fig. 9a, for "move_power_up" first training runs with this reset method have to be performed)





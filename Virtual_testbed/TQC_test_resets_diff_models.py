import warnings

import numpy as np
from sb3_contrib import TQC
from Env_fiber_simulated_goal import *
from CustomTensorboardCallback import *
import pandas as pd


def main():
    neutralxm1 = (max_xm1 - min_xm1) * params[0] + min_xm1
    neutralym1 = (max_ym1 - min_ym1) * params[2] + min_ym1
    neutralxm2 = (max_xm2 - min_xm2) * params[4] + min_xm2
    neutralym2 = (max_ym2 - min_ym2) * params[6] + min_ym2
    startvalues_mean = np.array([neutralxm1, neutralym1, neutralxm2, neutralym2], dtype=int)
    min_actuators_grid_scan = np.array([min_xm1, min_ym1, min_xm2, min_ym2])
    max_actuators_grid_scan = np.array([max_xm1, max_ym1, max_xm2, max_ym2])
    initial_radius = 109227
    min_power = 0.05
    number_obs_saved = 4
    max_actioninsteps = 6 * 10 ** 3
    minmirrorintervalsteps = 3 * 10 ** 6
    maxmirrorintervalsteps = 7 * 10 ** 6
    reset_power_fail = 0.05
    reset_power_goal = 0.85
    reset_step_size = 10 ** 3
    min_power_after_reset = 0.2
    max_power_after_reset = reset_power_goal
    max_random_reset_step = 10 ** 3
    min_actioninsteps = 1
    max_power_to_neutral = 0.04
    number_of_random_steps_low_power = 10
    min_power_stop_random_steps = 0.04
    max_random_reset_step_low_power = 10 ** 4
    max_random_reset_step_high_power = 10 * max_random_reset_step
    max_episode_steps = 30
    reset_method = "move_power_up"

    number_tries = 100
    df = pd.DataFrame(data=None, index=None,
                      columns=["reset_method", "timestamp", "episode_number",
                               "Power", "act1x", "act1y", "act2x", "act2y"])

    beta_step = 5
    beta_fail_1 = 5
    beta_fail_2 = 5
    beta_goal_1 = 5
    beta_goal_2 = 1
    alpha_goal = 0.5
    alpha_fail = 0.5
    alpha_step = 0.9
    prefactor_step = 10
    prefactor_goal = 100
    prefactor_fail = 100
    save_name = (f"TQC_reward_2024_04_22_betas_{beta_step}_{beta_fail_1}_{beta_fail_2}_{beta_goal_1}"
                 f"_{beta_goal_2}_alphas_{alpha_step}_{alpha_fail}_{alpha_goal}_prefactors_{prefactor_step}"
                 f"_{prefactor_fail}_{prefactor_goal}_resets.csv")
    # test different resets
    for reset_method in ["move_power_up", "interval"]:
        parameters = [[21000, 1], [24000, 10]]
        for initial_radius, number_episode_to_neutral in parameters:
            reset_descriptor = reset_method
            if reset_method == "move_power_up":
                reset_descriptor += str(number_episode_to_neutral)
            if reset_method == "interval":
                reset_descriptor += str(initial_radius)
            reward_fct_descriptor_2024_04_22 = (
                f"reward_2024_04_22_betas_{beta_step}_{beta_fail_1}_{beta_fail_2}"
                f"_{beta_goal_1}_{beta_goal_2}_prefactor_{prefactor_step}_"
                f"{prefactor_fail}_{prefactor_goal}_alphas_{alpha_step}_{alpha_fail}_{alpha_goal}")

            def reward_fct_2024_04_22(avg_power, max_power, power, reset_power_fail, max_episode_steps,
                                      reset_power_goal, min_power_after_reset, current_step):
                if power > reset_power_goal:
                    reward = prefactor_goal * (
                            ((1 - alpha_goal) * np.exp(-beta_goal_1 * current_step / max_episode_steps))
                            + alpha_goal * np.exp(beta_goal_2 * power / reset_power_goal))
                elif power < reset_power_fail:
                    reward = - prefactor_fail * (
                            (1 - alpha_fail) * np.exp(-beta_fail_1 * current_step / max_episode_steps)
                            + alpha_fail * np.exp(-beta_fail_2 * power / reset_power_fail))
                else:
                    reward = prefactor_step / max_episode_steps * ((1 - alpha_step) * np.exp(
                        beta_step * (power - reset_power_goal)) + alpha_step * (power - min_power_after_reset))
                return reward
            # all other rest methods tested are independent from the episodes in between, for this kind of reset we need
            # to go through episodes
            if reset_method == "move_power_up" and number_episode_to_neutral == 10:
                dir_names = f"reset_tests/" + reset_method
                files = os.listdir('./models/' + str(dir_names))
                for file in files:
                    timestamp = int(file)
                    print(reset_method, initial_radius, number_episode_to_neutral, timestamp)
                    env = Env_fiber_simulated(max_actioninsteps, minmirrorintervalsteps, maxmirrorintervalsteps,
                                              min_actuators_grid_scan,
                                              max_actuators_grid_scan, startvalues_mean, initial_radius,
                                              reset_power_fail,
                                              reset_power_goal, reward_fct_2024_04_22,
                                              reward_fct_descriptor_2024_04_22,
                                              max_random_reset_step_high_power, max_random_reset_step_low_power,
                                              min_power_stop_random_steps, max_power_to_neutral,
                                              number_of_random_steps_low_power, reset_step_size,
                                              min_power_after_reset, max_power_after_reset, min_power, reset_method,
                                              max_steps_under_min_power=3,
                                              average_over=10, number_obs_saved=number_obs_saved,
                                              max_episode_steps=max_episode_steps,
                                              timestamp=timestamp,
                                              random_reset=True, dir_names=dir_names, save_replay=True,
                                              number_episode_to_neutral=number_episode_to_neutral)
                    env.reset()
                    for num in [0, 100000]:
                        reset_descriptor = reset_method
                        if reset_method == "move_power_up":
                            reset_descriptor += str(number_episode_to_neutral)
                        reset_descriptor += "_"+str(num)
                        if num == 0:
                            policy_kwargs = dict(n_critics=2, n_quantiles=25)
                            model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1,
                                        policy_kwargs=policy_kwargs, tensorboard_log=env.logdir)
                        else:
                            models_dir = env.models_dir
                            model_path = models_dir + "/" + str(num)
                            log_path = env.logdir
                            model = TQC.load(model_path, tensorboard_log=log_path)
                        for j in range(number_tries):
                            obs, info = env.reset()
                            # always note starting powers, actuator positions in each episode
                            Power = obs[-1]
                            act1x, act1y, act2x, act2y = env.actuator_positions
                            df = pd.concat([df, pd.DataFrame([[reset_descriptor, timestamp, j,
                                                               Power, act1x, act1y, act2x, act2y]], columns=df.columns)],
                                           ignore_index=True)
                            terminated = False
                            truncated = False
                            if reset_method == "move_power_up" and number_episode_to_neutral == 10:
                                while not (terminated or truncated):
                                    action, _states = model.predict(obs)
                                    obs, rewards, terminated, truncated, info = env.step(action)
                        df.to_csv(save_name)
            else:
                for k in range(5):
                    print(reset_method, initial_radius, number_episode_to_neutral, k)
                    num = 0
                    env = Env_fiber_simulated(max_actioninsteps, minmirrorintervalsteps, maxmirrorintervalsteps,
                                          min_actuators_grid_scan,
                                          max_actuators_grid_scan, startvalues_mean, initial_radius,
                                          reset_power_fail,
                                          reset_power_goal, reward_fct_2024_04_22,
                                          reward_fct_descriptor_2024_04_22,
                                          max_random_reset_step_high_power, max_random_reset_step_low_power,
                                          min_power_stop_random_steps, max_power_to_neutral,
                                          number_of_random_steps_low_power, reset_step_size,
                                          min_power_after_reset, max_power_after_reset, min_power, reset_method,
                                          max_steps_under_min_power=3,
                                          average_over=10, number_obs_saved=number_obs_saved,
                                          max_episode_steps=max_episode_steps,
                                          timestamp=None,
                                          random_reset=True, dir_names=None, save_replay=True,
                                          number_episode_to_neutral=number_episode_to_neutral)
                    env.reset()
                    timestamp = env.timestamp
                    for j in range(number_tries):
                        obs, info = env.reset()
                        # always note starting powers, actuator positions in each episode
                        Power = obs[-1]
                        act1x, act1y, act2x, act2y = env.actuator_positions
                        df = pd.concat([df, pd.DataFrame([[reset_descriptor, timestamp, j,
                                                           Power, act1x, act1y, act2x, act2y]], columns=df.columns)],
                                       ignore_index=True)
                    df.to_csv(save_name)


if __name__ == '__main__':
    main()

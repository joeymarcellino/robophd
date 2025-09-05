#%%
import time

from stable_baselines3 import SAC
from EnvFibreGoal220424 import *
from CustomTensorboardCallback import *
import safe_exit
from sb3_contrib import TQC, CrossQ

import os 
import sys 
devices_relative_path = "../"
file_abs_path = os.path.abspath(__file__)
devices_abs_path = os.path.join(os.path.dirname(file_abs_path), devices_relative_path)

if devices_abs_path not in sys.path:
    sys.path.insert(0, devices_abs_path)

from devices.steppermotor_ble import StepMo
from devices.powermeter_pmodad5 import PmodAd5 
from devices.liveplotter_heavy import LivePlotAgent

#%%

def main():
    # enable pd after reset using sudo chmod 777 /dev/ttyACM0
    log_dir = "/home/robophd/Documents/github/robophd/devices/"
    pds: PmodAd5 = PmodAd5(address = "/dev/ttyACM0")
    actuators: StepMo = StepMo(log_dir = log_dir)
    liveplotter: LivePlotAgent = LivePlotAgent()
    
    actuators.zero_positions([1, 2, 3, 4])
    print('Stepper positions re-zeroed to current positions')

    def get_data():
        data = pds.get_measurement()
        #np.append(data,data[1]/data[0]*100) # add relative power in %
        return data

    plot_args ={
                'refresh_interval': 0.01,
                'title': "Live Powermeter",
                'xlabel': "Time (0.1s per bin)",
                'ylabel': "Power (mW)",
                'no_plots': 3,
                'plot_labels': None,
            }

    ### data_func is a method that returns an array of arrays
    liveplotter.new_liveplot(data_func=get_data, kill_func = None, **plot_args)



    max_actioninsteps = 400
    reset_power_fail = 0.05
    reset_power_goal = 0.8
    min_power_after_reset = 0.2
    max_power_after_reset = reset_power_goal
    max_cycles_per_episode = 30

    # close power meters when program is stopped or an error occurs
    @safe_exit.register
    def cleanup():
        pds.__exit__()
        actuators.close()
        print("cleanup called")

    # reward parameters
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
    reward_fct_descriptor_2024_04_22 = (
        f"reward_2024_04_22_betas_{beta_step}_{beta_fail_1}_{beta_fail_2}"
        f"_{beta_goal_1}_{beta_goal_2}_prefactor_{prefactor_step}_"
        f"{prefactor_fail}_{prefactor_goal}_alphas_{alpha_step}_{alpha_fail}_{alpha_goal}")
    dir_names = (f"goal/{reward_fct_descriptor_2024_04_22}/min_power_after_reset{min_power_after_reset}/"
                 f"reset_power_fail{reset_power_fail}/reset_power_goal{reset_power_goal}/"
                 f"max_actioninsteps{max_actioninsteps}/max_cycles_per_episode{max_cycles_per_episode}")

    def reward_fct_2024_04_22(avg_power, max_power, power, reset_power_fail, max_cycles_per_episode,
                              reset_power_goal, min_power_after_reset, current_step):
        if power > reset_power_goal:
            reward = prefactor_goal * (
                    ((1 - alpha_goal) * np.exp(-beta_goal_1 * current_step / max_cycles_per_episode))
                    + alpha_goal * np.exp(beta_goal_2 * power / reset_power_goal))
        elif power < reset_power_fail:
            reward = - prefactor_fail * (
                    (1 - alpha_fail) * np.exp(-beta_fail_1 * current_step / max_cycles_per_episode)
                    + alpha_fail * np.exp(-beta_fail_2 * power / reset_power_fail))
        else:
            reward = prefactor_step / max_cycles_per_episode * ((1 - alpha_step) * np.exp(
                beta_step * (power - reset_power_goal)) + alpha_step * (power - min_power_after_reset))
        return reward

    # new model (comment this part out when using pretrained model)
    env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                                       reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, 
                                       min_power_after_reset, max_power_after_reset,
                                       max_cycles_per_episode = max_cycles_per_episode, 
                                       dir_names = dir_names, 
                                       mirror_pos_lower_bound = -3 * 10 ** 5, mirror_pos_upper_bound = 3 * 10 ** 5,
                                        ref_pd_intercept = 0, ref_pd_slope = 1,
                                        min_ref_power = 1 * 10 ** (-1), 
                                        grad_ascent_step_size = 5 * 10 ** 1,
                                        extra_random_step_magnitude = 5 * 10 ** 1, 
                                        min_actioninsteps = 1, max_power_to_neutral = 0.01,
                                        number_of_random_actions_low_power = 10, 
                                        min_power_stop_random_actions_neutral_failure = 0.04,
                                        neutral_flailing_step_magnitude = int(200), high_power_flailing_step_magnitude = int(50),
                                        wait_time_pd = 0, 
                                        number_obs_saved = 4, 
                                        timestamp = None,
                                        random_reset = True, 
                                        save_replay = True)





    env.reset()
    policy_kwargs = dict(n_critics=2, n_quantiles=25)  # new TQC model
    model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs,
                tensorboard_log=env.logdir, device="cpu")  # use this for TQC
    # model = CrossQ("MlpPolicy", env, tensorboard_log=env.logdir) #use this for CrossQ
    # model = SAC("MlpPolicy", env, verbose=1, tensorboard_log=env.logdir)  #use this for SAC
    num = 0
    """
    # load model (comment this part out when wanting fresh model)
    first_timestamp = 1714485701
    first_num = 98000
    old_timestamp = 1715453496
    num = 193000
    # if we don't want to change goal power
    timestamp = old_timestamp
    #if we want to change goal power
    dir_names = (f"goal/{reward_fct_descriptor_2024_04_22}/min_power_after_reset{min_power_after_reset}/"
                 f"reset_power_fail{reset_power_fail}/reset_power_goal{reset_power_goal}/"
                 f"max_actioninsteps{max_actioninsteps}/max_cycles_per_episode{max_cycles_per_episode}/"
                f"start_with_{first_timestamp}_{first_num}")
    #timestamp = None
    env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                 reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, min_power_after_reset, max_power_after_reset,
                                       timestamp=timestamp, dir_names=dir_names, save_replay=True,
                                       max_cycles_per_episode=max_cycles_per_episode)
    env.reset()
    old_model_path = (f"models/goal/{reward_fct_descriptor_2024_04_22}/min_power_after_reset{min_power_after_reset}/"
                      f"reset_power_fail{reset_power_fail}/reset_power_goal{0.9}/max_actioninsteps{max_actioninsteps}/"
                      f"max_cycles_per_episode{max_cycles_per_episode}/"
                      f"start_with_{first_timestamp}_{first_num}/"
                      f"{old_timestamp}/{num}")
    old_replay_path = (f"replay/goal/{reward_fct_descriptor_2024_04_22}/min_power_after_reset{min_power_after_reset}/"
                       f"reset_power_fail{reset_power_fail}/reset_power_goal{0.9}/max_actioninsteps{max_actioninsteps}/"
                       f"max_cycles_per_episode{max_cycles_per_episode}/"
                       f"start_with_{first_timestamp}_{first_num}/"
                       f"{old_timestamp}/{num}")
    models_dir = env.models_dir
    model_path = models_dir+"/"+str(num)
    replay_dir = env.replay_dir
    replay_path = replay_dir+"/"+str(num)
    log_path = env.logdir
    model = TQC.load(old_model_path, tensorboard_log=log_path)
    model.set_env(env)
    model.load_replay_buffer(old_replay_path, truncate_last_traj=True)
    """
    # start training (for 200k training steps)

    TIMESTEPS = 1000
    for i in range(40):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TQC",
                    callback=CustomTensorboardCallback(env))
        model.save(f"{env.models_dir}/{num + TIMESTEPS * (i + 1)}")
        model.save_replay_buffer(f"{env.replay_dir}/{num + TIMESTEPS * (i + 1)}")

    env.close()


if __name__ == '__main__':
    main()

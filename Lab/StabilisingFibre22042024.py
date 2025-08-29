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

from devices.steppermotor_ble import StepMo, SelfLoggingStepper
from devices.powermeter_pmodad5 import PmodAd5 

#%%

def main():
    # enable pd after reset using sudo chmod 777 /dev/ttyACM0
    log_dir = "/home/robophd/Documents/github/robophd/devices/"
    pds: PmodAd5 = PmodAd5(address = "/dev/ttyACM0")
    actuators: SelfLoggingStepper = SelfLoggingStepper(log_dir)
    for i in range(4):
        action_sign = 1 if actuators.current_position[i] < 0 else 0
        actuators.move_stepper_logged(i+1, action_sign, int(np.abs(actuators.current_position[i])))
    print('Stepper positions reset')

    max_actioninsteps = 1000
    reset_power_fail = 0.05
    reset_power_goal = 0.8
    min_power_after_reset = 0.2
    max_power_after_reset = reset_power_goal
    max_episode_steps = 30

    # close power meters when program is stopped or an error occurs
    @safe_exit.register
    def cleanup():
        pds.__exit__()
        actuators.__exit__()
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
                 f"max_actioninsteps{max_actioninsteps}/max_episode_steps{max_episode_steps}")

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

    # new model (comment this part out when using pretrained model)
    env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                                       reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, min_power_after_reset,
                                       max_power_after_reset,
                                       max_random_reset_step_high_power=max_actioninsteps,
                                       max_episode_steps=max_episode_steps, dir_names=dir_names)  # load environment
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
                 f"max_actioninsteps{max_actioninsteps}/max_episode_steps{max_episode_steps}/"
                f"start_with_{first_timestamp}_{first_num}")
    #timestamp = None
    env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                 reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, min_power_after_reset, max_power_after_reset,
                                       timestamp=timestamp, dir_names=dir_names, save_replay=True,
                                       max_episode_steps=max_episode_steps)
    env.reset()
    old_model_path = (f"models/goal/{reward_fct_descriptor_2024_04_22}/min_power_after_reset{min_power_after_reset}/"
                      f"reset_power_fail{reset_power_fail}/reset_power_goal{0.9}/max_actioninsteps{max_actioninsteps}/"
                      f"max_episode_steps{max_episode_steps}/"
                      f"start_with_{first_timestamp}_{first_num}/"
                      f"{old_timestamp}/{num}")
    old_replay_path = (f"replay/goal/{reward_fct_descriptor_2024_04_22}/min_power_after_reset{min_power_after_reset}/"
                       f"reset_power_fail{reset_power_fail}/reset_power_goal{0.9}/max_actioninsteps{max_actioninsteps}/"
                       f"max_episode_steps{max_episode_steps}/"
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
    for i in range(200):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TQC",
                    callback=CustomTensorboardCallback(env))
        model.save(f"{env.models_dir}/{num + TIMESTEPS * (i + 1)}")
        model.save_replay_buffer(f"{env.replay_dir}/{num + TIMESTEPS * (i + 1)}")

    env.close()


if __name__ == '__main__':
    main()

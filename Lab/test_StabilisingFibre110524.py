from stable_baselines3 import SAC
from EnvFibreGoal220424 import *
import Photodetector
from pyvisa.highlevel import ResourceManager
from pylablib.devices import Thorlabs
import os
import safe_exit
import time
import pandas as pd
from sb3_contrib import TQC, CrossQ

def main():
    actxm1 = Thorlabs.KinesisMotor("26004585")
    actym1 = Thorlabs.KinesisMotor("26004587")
    actxm2 = Thorlabs.KinesisMotor("26003852")
    actym2 = Thorlabs.KinesisMotor("26003794")
    actuators = [actxm1, actym1, actxm2, actym2]

    pd2 = Photodetector.Photodetector('USB0::0x1313::0x807B::1922851::0::INSTR')
    pd1 = Photodetector.Photodetector('USB0::0x1313::0x807B::1922850::0::INSTR')  # reference
    pd1.clear() # L: added. this should solve the famous Visa Error.
    pd2.clear()

    pds = [pd1, pd2]
    max_actioninsteps = 6000
    reset_power_fail = 0.05
    min_power_after_reset = 0.2
    max_episode_steps = 30

    number_tries = 100  # number of times each model is tested
    # close power meters when program is stopped or an error occurs
    @safe_exit.register
    def cleanup():
        pd1.close()
        pd2.close()
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
    # list of dir_names, timestamp, timesteps to define models tested, goal power at which this model should be tested
    model_list = [["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
                       "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.87/max_actioninsteps6000/"
                       "max_episode_steps30", 1714915417, 27000, 0.87],
                  ["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
                   "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.87/max_actioninsteps6000/"
                   "max_episode_steps30", 1714915417, 35000, 0.87],
                    ["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
                       "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.85/max_actioninsteps6000/"
                       "max_episode_steps30", 1714153135, 37000, 0.85]]
    
    # CrossQ agents
    # model_list = [["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
    #                "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.9/max_actioninsteps6000/"
    #                "max_episode_steps30", 1750325230, 113000, 0.9],
    #                 ["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
    #                    "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.85/max_actioninsteps6000/"
    #                    "max_episode_steps30", 1749908988, 34000, 0.85]]

    # SAC agents
    # model_list = [["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
    #                "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.9/max_actioninsteps6000/"
    #                "max_episode_steps30", 1750687596, 122000, 0.9],
    #               ["goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/"
    #                "min_power_after_reset0.2/reset_power_fail0.05/reset_power_goal0.85/max_actioninsteps6000/"
    #                "max_episode_steps30", 1714656374, 34000, 0.85]]
    i = 0
    for dir_names, timestamp, num, reset_power_goal in model_list:
        print(reset_power_fail, min_power_after_reset, reset_power_goal)
        max_power_after_reset = reset_power_goal
        i += 1
        # open environment
        env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                                           reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, min_power_after_reset,
                                           max_power_after_reset, timestamp=timestamp, dir_names=dir_names,
                                           max_episode_steps=30)
        env.reset()
        # load model
        models_dir = env.models_dir
        model_path = models_dir + "/" + str(num) + ".zip"
        log_path = env.logdir
        model = TQC.load(model_path, tensorboard_log=log_path)
        model.set_env(env)
        # make/load dataframe for saving testing results
        save_name = (f"testing/reset_until_goal_reached/{reward_fct_descriptor_2024_04_22}/{timestamp}_{num}_goal{reset_power_goal}_"
                     f"fail{reset_power_fail}_start{min_power_after_reset}.csv")
        if os.path.exists(save_name):
            df = pd.read_csv(save_name, index_col=0)
        else:
            df = pd.DataFrame(data=None, index=None, columns=["try", "reset_number", "time_in_s", "Power"])
        df.to_csv(save_name)
        # start testing
        for j in range(number_tries):
            print(f'This is try number {j+1} out of {number_tries}')
            terminated = False
            times = []
            powers = []
            power = 0
            how_many_resets = -1
            reset_list = []
            while power < reset_power_goal:  # reset and do steps as long as it takes to reach goal power
                obs, info = env.reset()  # reset
                how_many_resets += 1
                power = obs[-1]  # get power
                if how_many_resets == 0:
                    time_start = time.time()
                timej = time.time() - time_start  # get time
                times.append(timej)
                powers.append(power)
                reset_list.append(how_many_resets)
                truncated = False
                terminated = False
                while not (terminated or truncated):  # do steps (noting time and power) as long as episode takes
                    action, _states = model.predict(obs)
                    obs, rewards, terminated, truncated, info = env.step(action)
                    power = obs[-1]
                    timej = time.time()-time_start
                    times.append(timej)
                    powers.append(power)
                    reset_list.append(how_many_resets)
            try_number = [j for _ in range(len(times))]
            # lists to dataframes
            df = pd.concat([df, pd.DataFrame({"try": try_number, "reset_number": reset_list,
                                              "time_in_s": times, "Power": powers})],ignore_index=True)
            df.to_csv(save_name)
    env.close()



if __name__ == '__main__':
    # profiler = cProfile.Profile()
    # profiler.enable()
    # time1 = time.time()
    main()
    # time2 = time.time()
    # profiler.disable()
    # profiler.dump_stats('p16.stats')

    # stats = pstats.Stats('p16.stats')
    # stats.print_stats()
    # print(time2-time1)

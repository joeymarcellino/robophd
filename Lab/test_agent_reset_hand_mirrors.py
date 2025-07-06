
from MovePowerUp import *
import Photodetector
from pylablib.devices import Thorlabs
import os
import safe_exit
import time
import pandas as pd
import threading
import keyboard
from stable_baselines3 import SAC
from EnvFibreGoal220424 import *
import Photodetector
from pylablib.devices import Thorlabs
import os
import safe_exit
import time
import pandas as pd
from sb3_contrib import TQC
from send_mail import *





# Flag to indicate which process is currently running
first_process_running = True
global i
i = 0
global j
j = 0
actxm1 = Thorlabs.KinesisMotor("26004585")
actym1 = Thorlabs.KinesisMotor("26004587")
actxm2 = Thorlabs.KinesisMotor("26003852")
actym2 = Thorlabs.KinesisMotor("26003794")
actuators = [actxm1, actym1, actxm2, actym2]

pd2 = Photodetector.Photodetector('USB0::0x1313::0x807B::1922851::0::INSTR')
pd1 = Photodetector.Photodetector('USB0::0x1313::0x807B::1922850::0::INSTR')  # reference
pd1.clear() 
pd2.clear()

pds = [pd1, pd2]
max_actioninsteps = 6*10**3 # new value 27/03/24, before 1e4
minmirrorintervalsteps = 3 * 10 ** 6
maxmirrorintervalsteps = 7 * 10 ** 6
wait_time_pd = 0
min_ref_power = 3 * 10 ** (-4)
power_multiplier = 2.363  # new values March 2024
power_adder = (-1)* 5.653  * 10 ** (-6)
neutralxm1 = 5461300  # 5461333 changes to new neutral position
neutralym1 = 5570600  # 5570560 changes to new neutral position
neutralxm2 = 5461300  # 5461333 changes to new neutral position
neutralym2 = 5177300  # 5177344 changes to new neutral position
neutral_positions = [neutralxm1, neutralym1, neutralxm2, neutralym2]
reset_power_fail = 0.05  # 0.0009
reset_power_goal = 0.9
reset_step_size = 10 ** 3  # L: currently used for the (somewhat) gradient ascent
min_power_after_reset = 0.2  # 0.001
max_power_after_reset = reset_power_goal
max_random_reset_step = 10 ** 3  min_actioninsteps = 1
max_power_to_neutral = 0.04

number_of_random_steps_low_power = 10
min_power_stop_random_steps = 0.04
max_random_reset_step_low_power = 1e4 
max_random_reset_step_high_power = 1e4 


@safe_exit.register
def cleanup():
    pd1.close()
    pd2.close()
    print("cleanup called")

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

dir_names = ("/goal/reward_2024_04_22_betas_5_5_5_5_1_prefactor_10_100_100_alphas_0.9_0.5_0.5/min_power_after_reset0.2"
             "/reset_power_fail0.05/reset_power_goal0.9/max_actioninsteps6000/max_episode_steps30/"
             "start_with_1714326368_98000")
timestamp = 1714375298
num = 204000
save_name = (f"testing/reset_until_goal_reached/different_hand_mirror_pos_test.csv")
if os.path.exists(save_name):
    df = pd.read_csv(save_name, index_col=0)
else:
    df = pd.DataFrame(data=None, index=None, columns=["try", "reset_number", "step", "time_in_s", "Power", "act1x",
                                          "act1y", "act2x", "act2y"])
df.to_csv(save_name)

env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                                               reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, min_power_after_reset,
                                               max_power_after_reset, timestamp=timestamp, dir_names=dir_names,
                                               max_episode_steps=30)
models_dir = env.models_dir
model_path = models_dir + "/" + str(num) + ".zip"
log_path = env.logdir
model = TQC.load(model_path, tensorboard_log=log_path)
model.set_env(env)
# Define your first process as a function
def first_process():
    print(f"start reset using hand mirrors, try {i}")
    #env.reset()
    print(f"try {i}, after resetting, press ctrl+t")


# Define your second process as a function
def second_process():
    global i
    print(f"agent starts optimizing, try {i}")
    i = i + 1
    times = []
    powers = []
    power = 0
    how_many_resets = -1
    reset_list = []
    act1x_list = []
    act1y_list = []
    act2x_list = []
    act2y_list = []
    t_start = time.time()
    df = pd.read_csv(save_name, index_col=0)
    step_list = []
    if not first_process_running:
        while power < reset_power_goal:
            obs, info = env.reset_without_changing_positions()
            how_many_resets += 1
            power = obs[-1]
            if how_many_resets == 0:
                time_start = time.time()
            timej = time.time() - time_start
            times.append(timej)
            powers.append(power)
            reset_list.append(how_many_resets)
            act1x_list.append(env.actuator_positions[0])
            act1y_list.append(env.actuator_positions[1])
            act2x_list.append(env.actuator_positions[2])
            act2y_list.append(env.actuator_positions[3])
            truncated = False
            terminated = False
            step_number = 0
            step_list.append(step_number)
            while not (terminated or truncated):
                action, _states = model.predict(obs)
                obs, rewards, terminated, truncated, info = env.step(action)
                power = obs[-1]
                timej = time.time() - time_start
                times.append(timej)
                powers.append(power)
                reset_list.append(how_many_resets)
                act1x_list.append(env.actuator_positions[0])
                act1y_list.append(env.actuator_positions[1])
                act2x_list.append(env.actuator_positions[2])
                act2y_list.append(env.actuator_positions[3])
                step_number += 1
                step_list.append(step_number)
        try_number = [i for _ in range(len(times))]
        df = pd.concat([df, pd.DataFrame({"try": try_number, "reset_number": reset_list, "step": step_list,
                                          "time_in_s": times, "Power": powers, "act1x": act1x_list,
                                          "act1y": act1y_list, "act2x": act2x_list, "act2y": act2y_list})],
                       ignore_index=True)
        df.to_csv(save_name)
        print(f"goal reached, try {i}, press ctrl+t to reset")


# Define a function to toggle between the processes when a key is pressed
def toggle_processes():
    global first_process_running
    first_process_running = not first_process_running
    print("Processes toggled.")
    if not first_process_running:
        # Start the second process in a separate thread
        second_thread = threading.Thread(target=second_process)
        second_thread.start()
    if first_process_running:
        # Start the second process in a separate thread
        second_thread = threading.Thread(target=first_process)
        second_thread.start()

# Start the first process in a separate thread
first_thread = threading.Thread(target=first_process)
first_thread.start()
# Listen for a key press to toggle between the processes
keyboard.add_hotkey('ctrl+t', toggle_processes)



# Keep the main thread running indefinitely
try:
    while True:
        pass
except KeyboardInterrupt:
    # Cleanup code if needed
    pds[0].close()
    pds[1].close()
    pass



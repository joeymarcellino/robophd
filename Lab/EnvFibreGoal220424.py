import numpy as np
import time as time
import gymnasium as gym
from gymnasium import spaces
import random
import os
import warnings
from MovePowerUp import *
import pandas as pd


neutralxm1 = 0  # 5461333 changes to new neutral position
neutralym1 = 0  # 5570560 changes to new neutral position
neutralxm2 = 0  # 5461333 changes to new neutral position
neutralym2 = 0  # 5177344 changes to new neutral position


class Env_fiber_move_by_grad_reset(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                 reward_fct, reward_fct_descriptor, min_power_after_reset, max_power_after_reset,
                 minmirrorintervalsteps = 3 * 10 ** 6, maxmirrorintervalsteps = 7 * 10 ** 6,
                 neutral_positions=[neutralxm1, neutralym1, neutralxm2, neutralym2],
                 power_adder=0, power_multiplier=1,
                 min_ref_power=3 * 10 ** (-4), reset_step_size=10 ** 3,
                 max_random_reset_step=10 ** 3, min_actioninsteps=1, max_power_to_neutral=0.04,
                 number_of_random_steps_low_power=10, min_power_stop_random_steps=0.04,
                 max_random_reset_step_low_power=1e4, max_random_reset_step_high_power=1e4,
                 wait_time_pd=0, number_obs_saved=4, max_episode_steps=20, timestamp=None,
                 random_reset=True, dir_names=None, save_replay=True):
        """
        :param list actuators: list of objects, e.g. pylablib stages, representing the actuators (act_1x, act_1y, act_2x, act_2y)
        that should have class functions, is_moving, move_by, move_to, get_position, wait_move
        :param list pds: list of objects representing photodetectors/powermeters (reference pd, pd after fiber)
        that should have class functions get_power, close
        :param int max_actioninsteps: maximal action that can be taken in steps
        :param int minmirrorintervalsteps: minimum position each motor is allowed to be in
        :param int maxmirrorintervalsteps: maximum position each motor is allowed to be in
        :param list neutral_positions: list of ints, position of actuators with the maximum near it (after trying to couple
        fiber as a human) before first reset (afterwards backlash kills meaning of absolute positions)
        :param float reset_power_fail: If power is smaller or equal than this power is seen during a step, the agent failed,
        probably a big negative reward is given, the episode is terminated and reset is called
        :param float reset_power_goal: If power is larger or equal than this power is seen after a step, the agent reached the goal,
        probably a big positive reward is given, the episode is terminated and reset is called
        :param callable reward_fct: Reward function
        :param str reward_fct_descriptor: Descriptor of reward function, only used for log/model directory
        :param float power_adder: the maximum power is a linear function of the reference power, this is the offset
        :param float power_multiplier: the maximum power is a linear function of the reference power, this is the slope
        :param float min_ref_power: minimal reference power. If we are under that value, probably no light is coming
        to the experiment
        :param int reset_step_size: approximate step size for gradient ascent
        :param float min_power_after_reset: the minimal power we should have after reset
        :param float max_power_after_reset: the maximal power we should have after reset (at least when starting high)
        :param int max_random_reset_step: random steps at the end of reset
        :param int min_actioninsteps: minimal action that can be taken in steps
        :param float max_power_to_neutral: when resetting, this is the maximal power from which we would go back to the
        neutral positions and do random steps from there, as gradient ascent will probably fail
        :param int number_of_random_steps_low_power: when we have reached such a low power, this is the maximal number of times
        we do random steps before again going to neutral positions
        :param float min_power_stop_random_steps: we stop these random steps after reaching this value
        :param int max_random_reset_step_low_power: when we have reached such a low power, this is the approximate stepsize
        with which we do random steps
        :param int max_random_reset_step_high_power:  when we have a too high power in the reset step, this is the approximate stepsize
        with which we do random steps
        :param float wait_time_pd: time between pd measurements
        :param int number_obs_saved: number of time steps we save in observation
        :param int max_episode_steps: the maximal number of steps per episode. if it's reached, the episode is truncated
        and the reset function is called
        :param None or int timestamp: Timestamp when training is first started (for logging)
        :param bool random_reset: True if we want to do random steps and move power up when resetting, False if not
        :param None or str dir_names: Name of directories to save stuff in without logs/models and timestamp in
        case we want to train a policy further after changing parameters like the episode length or goal power
        :param bool save_replay: True, if the replay buffer should be saved, else False
        """
        super(Env_fiber_move_by_grad_reset, self).__init__()
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(1+8*number_obs_saved,), dtype=np.float64)

        # variables
        self.actuators = actuators
        self.pds = pds
        # variables
        self.max_episode_steps = max_episode_steps
        self.max_actioninsteps = max_actioninsteps
        self.minmirrorintervalsteps = minmirrorintervalsteps
        self.maxmirrorintervalsteps = maxmirrorintervalsteps
        self.neutral_positions = neutral_positions
        self.wait_time_pd = wait_time_pd
        self.reset_power_fail = reset_power_fail
        self.reset_power_goal = reset_power_goal
        self.min_ref_power = min_ref_power
        self.powermultiplier = power_multiplier
        self.poweradder = power_adder
        self.number_obs_saved = number_obs_saved
        self.reward_fct = reward_fct
        self.number_episodes = 0
        self.random_reset = random_reset
        self.reset_step_size = reset_step_size
        self.min_power_after_reset = min_power_after_reset
        self.max_random_reset_step = max_random_reset_step
        self.min_actioninsteps = min_actioninsteps
        self.actioninsteps = np.array([0, 0, 0, 0])
        self.max_power_after_reset = max_power_after_reset
        self.max_power_to_neutral = max_power_to_neutral
        self.number_of_random_steps_low_power = number_of_random_steps_low_power
        self.min_power_stop_random_steps = min_power_stop_random_steps
        self.max_random_reset_step_low_power = max_random_reset_step_low_power
        self.max_random_reset_step_high_power = max_random_reset_step_high_power
        # self.extra_threshold = extra_threshold # L: added
        if timestamp == None:
            timestamp = int(time.time())
        if dir_names == None:
            self.models_dir = f"models/{timestamp}"
            self.logdir = f"logs/{timestamp}"
            if save_replay:
                self.replay_dir = f"replay/{timestamp}"
        else:
            self.models_dir = "models/"+dir_names+"/"+str(timestamp)
            self.logdir = "logs/" + dir_names+"/"+str(timestamp)
            if save_replay:
                self.replay_dir = "replay/" + dir_names+"/"+str(timestamp)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if save_replay:
            if not os.path.exists(self.replay_dir):
                os.makedirs(self.replay_dir)
        self.episode_number = 0
        self.timestamp = timestamp
        self.df = pd.DataFrame(data=None, index=None, columns=["episode", "number_movements_reset", "time_reset"])

    def step(self, action):
        how_long_ref_power_under_min_ref_power = 0
        # test if we have reference power. Otherwise, wait until we have (in case no laser beam gets to experiment)
        while self.pds.get_measurement()[0][-1] < self.min_ref_power:
            time.sleep(self.wait_time_pd)
            how_long_ref_power_under_min_ref_power += 1
            if how_long_ref_power_under_min_ref_power > 10:
                warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
        self.episode_steps += 1
        # get max power from reference powermeter
        self.max_power = (self.pds.get_measurement()[0][-1]) * self.powermultiplier + self.poweradder
        # delete first parts of observation
        self.observation = np.delete(self.observation, [i for i in range(8)])
        # calculate the action in steps from the normalized action
        self.actioninsteps = np.around(action * self.max_actioninsteps).astype(int)
        print(f'Current action in steps: {self.actioninsteps}')
        # test if action would lead out of the interval and clip action so that it stays in the interval (only a
        # safeguard, is this way most of the time)
        self.actuator_positions = [self.actuators.motor_params[i+1]['pos'] for i in range(4)]
        for i in range(4):
            if self.actioninsteps[i] + self.actuator_positions[i] >= self.maxmirrorintervalsteps:
                self.actioninsteps[i] = self.maxmirrorintervalsteps - self.actuator_positions[i]
            if self.actioninsteps[i] + self.actuator_positions[i] <= self.minmirrorintervalsteps:
                self.actioninsteps[i] = self.minmirrorintervalsteps - self.actuator_positions[i]
        # measure power
        power = (self.pds.get_measurement()[1][-1]) / self.max_power
        power_list = [power]
        # perform action
        for i in range(4):
            action_sign = [1 if np.sign(self.actioninsteps[i]) >= 0 else 0]
            self.actuators.move_stepper(i+1, action_sign, np.abs(self.actioninsteps[i]))
        # get power from last second of measurement
        power = (self.pds.get_measurement()[1]) / self.max_power
        np.append(power_list, power)
        time.sleep(self.wait_time_pd)
        # calculate argmax, max, ave for observation
        power_array = np.array(power_list)
        power_argmax = np.argmax(power_array)
        power_meas_max = power_list[power_argmax]
        power_argmax = power_argmax/len(power_list)
        power_ave = np.mean(power_array)
        # normalise action for observation
        actionnormalized = self.actioninsteps / self.max_actioninsteps

        # append last history points
        self.observation = np.append(self.observation, np.array([actionnormalized[0], actionnormalized[1], actionnormalized[2],
                                                                 actionnormalized[3], power_ave, power_meas_max, power_argmax, power]))

        # calculate reward
        reward = self.reward_fct(power_ave, power_meas_max, power, self.reset_power_fail, self.max_episode_steps,
                                 self.reset_power_goal, self.min_power_after_reset, self.episode_steps)
        # reset if agent failed or reached its goal (terminated)
        if power < self.reset_power_fail:
            self.terminated = True
            print(power, "failed")
            self.fail = True
        if power > self.reset_power_goal:
            self.terminated = True
            print(power, "goal reached")
            self.goal = True
        # reset if agent reached max. episode length (truncated)
        if self.max_episode_steps == self.episode_steps:
            self.truncated = True
        # info
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0], "act_1y_pos": self.actuator_positions[1],
                 "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": power}
        print(self.info)
        return self.observation, reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        print('Reset called')
        super().reset(seed=seed)
        time_reset_start = time.time()
        self.episode_number += 1
        number_reset_movements = 0  # number of movements performed during the reset
        # test if reference powermeter has power, otherwise wait until it has
        how_long_ref_power_under_min_ref_power = 0
        while self.pds.get_measurement()[0][-1] < self.min_ref_power:
            time.sleep(self.wait_time_pd)
            how_long_ref_power_under_min_ref_power += 1
            if how_long_ref_power_under_min_ref_power > 10:
                warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
        # calculate max power from ref power
        self.max_power = (self.pds.get_measurement()[0][-1]) * self.powermultiplier + self.poweradder
        self.episode_steps = 0
        self.terminated = False
        self.truncated = False
        # move actuators for reset
        print(f'Resetting. Current action in steps: {self.actioninsteps}')  # L: added #not last?
        if self.random_reset:
            power_old = self.pds.get_measurement()[1][-1] / self.max_power
            print(f'Power when reset is called: {power_old}')
            ###############
            sgn_last_action = np.sign(self.actioninsteps)
            # first: reverse the last action if power < reset_power_fail
            if power_old < self.reset_power_fail and not np.array_equal(self.actioninsteps, np.array([0, 0, 0, 0])):  # this should only be done if the reset is called because of low power.
                for i in range(4):
                    reverse_steps = (-1)*(self.actioninsteps[i])
                    action_sign = [1 if np.sign(reverse_steps) >= 0 else 0]
                    self.actuators.move_stepper(i+1, action_sign, np.abs(reverse_steps))
                    print(f'reverse steps {i}: {reverse_steps}')
                    number_reset_movements += 1
                power_new = self.pds.get_measurement()[1][-1] / self.max_power
                print(f'Power after reversing last action: {power_new}')
                power_old = power_new
            # second: move to neutral positions and do some random steps if power is very small or every ten episodes
            if self.episode_number % 10 == 0 or power_old < self.max_power_to_neutral+0.05:
                number_moves_to_neutral, power_new = to_neutral_positions_random_steps(self.pds, self.actuators,
                                                                                       self.max_power,
                                                                                       self.neutral_positions,
                                                                                       self.max_power_to_neutral,
                                                                                       self.number_of_random_steps_low_power,
                                                                                       self.max_random_reset_step_low_power,
                                                                                       self.min_power_stop_random_steps)
                number_reset_movements += number_moves_to_neutral
                power_old = power_new
            # third, if power now is high, choose a power randomly and do random steps until we are below that power
            if power_old > self.min_power_after_reset:  # case where we have high powers when resetting
                appr_reset_power = np.random.uniform(low=self.min_power_after_reset+0.1, high=self.max_power_after_reset)
                print('want to have reset power < '+str(appr_reset_power))
                power_new = power_old
                while power_new > appr_reset_power:
                    for i in range(4):
                        add_random_steps = random.randint(- self.max_random_reset_step_high_power,
                                                          self.max_random_reset_step_high_power)
                        action_sign = [1 if np.sign(add_random_steps) >= 0 else 0]
                        self.actuators.move_stepper(i+1, action_sign, np.abs(add_random_steps))
                        number_reset_movements += 1
                        # the next episode will start even if these random steps move below min_power_after_reset.
                        # That's good, so we have different start conditions.
                        print(f'Actuator {i}: {add_random_steps} random steps moved.')
                    power_new = self.pds.get_measurement()[1][-1] / self.max_power
                    print(f'Power after doing random steps: {power_new}')
            # call move_power_up (see case 2 paper, in the case of small power)
            start_dir = (-1) * sgn_last_action
            if np.array_equal(start_dir, np.array([0, 0, 0, 0])):
                start_dir = np.array([(2*random.randint(0, 1) - 1) for _ in range(4)])
            number_move_power_up_movements = move_power_up(self.pds, self.actuators, self.max_power, start_dir, self.neutral_positions, self.min_power_after_reset,
                          self.max_power_to_neutral, self.number_of_random_steps_low_power, self.max_random_reset_step_low_power,
                          self.min_power_stop_random_steps, self.reset_step_size, self.min_ref_power,
                          self.wait_time_pd, self.powermultiplier, self.poweradder)
            number_reset_movements += number_move_power_up_movements
        # Do some random steps (to get some extra randomness...)
        if self.random_reset:
            for i in range(4):
                add_random_steps = random.randint(- self.max_random_reset_step, self.max_random_reset_step)
                action_sign = [1 if np.sign(add_random_steps) >= 0 else 0]
                self.actuators.move_stepper(i+1, action_sign, np.abs(add_random_steps))
                number_reset_movements += 1
                # the next episode will start even if these random steps move below min_power_after_reset.
                # That's good, so we have different start conditions.
                print(f'Actuator {i}: {add_random_steps} random steps moved.')
            for i in range(4):
                self.actuators[i].wait_move()
        self.actuator_positions = [self.actuators.motor_params[i+1]['pos'] for i in range(4)]
        # observation:
        power = self.pds.get_measurement()[1][-1] / self.max_power
        self.observation = np.array([power])
        for i in range(self.number_obs_saved):
            self.observation = np.append(self.observation, np.array([0.0, 0.0, 0.0, 0.0, power, power, 0.0, power]))
        # the [0, 0, 0, 0] corresponds to the fact that in the initial observation, before the first episode step
        # happens, there is no action
        # obs_t' for t'=t-(number_obs_saved+1) and act0_t', act1_t', act2_t', act3_t', average_power_t'-1_t',
        # max_power_t'-1_t', maxpos_t'-1_t', power_t' for t' = t,...,t-number_obs_saved
        # info
        self.info = {"episode_step": self.episode_steps, "act_1y_pos": self.actuators.motor_params[1]['pos'],
                "act_1x_pos": self.actuators.motor_params[2]['pos'],
                "act_2y_pos": self.actuators.motor_params[3]['pos'], "act_2x_pos": self.actuators.motor_params[4]['pos'], "power": power}
        print(self.info)
        time_reset_end = time.time()
        # save how long reset took
        print(time_reset_end-time_reset_start, number_reset_movements)
        self.df = self.df._append({"episode": self.episode_number,
                                  "number_movements_reset": number_reset_movements,
                                  "time_reset":time_reset_end-time_reset_start}, ignore_index = True)
        self.df.to_csv(f"reset_time_{self.timestamp}.csv")
        self.goal = False
        self.fail = False
        return self.observation, self.info  # reward, done can't be included

    def close(self):
        # close powermeters
        for pd in self.pds:
            pd.__exit__()


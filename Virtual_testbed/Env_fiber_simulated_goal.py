import pandas as pd
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import random
import time
import os
from MovePowerUp import *

params = pd.read_csv('param_fit.csv')['value']
min_max = pd.read_csv('min_max_positions.csv')['value']
min_xm1 = min_max[4]
min_ym1 = min_max[5]
min_xm2 = min_max[6]
min_ym2 = min_max[7]
max_xm1 = min_max[0]
max_ym1 = min_max[1]
max_xm2 = min_max[2]
max_ym2 = min_max[3]

def fit(x1, y1, x2, y2):
    return 0.92*(np.exp(-(x1-params[0])**2/(2*params[1]**2))
     *np.exp(-(y1-params[2])**2/(2*params[3]**2))
     *np.exp(-(x2-params[4])**2/(2*params[5]**2))
     * np.exp(-(y2 - params[6]) ** 2 / (2 * params[7] ** 2))
     )




class Env_fiber_simulated(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, max_actioninsteps, minmirrorintervalsteps, maxmirrorintervalsteps, min_actuators_grid_scan,
                 max_actuators_grid_scan, startvalues_mean, initial_radius, reset_power_fail, reset_power_goal,
                 reward_fct, reward_fct_descriptor, max_random_reset_step_high_power, max_random_reset_step_low_power,
                 min_power_stop_random_steps, max_power_to_neutral, number_of_random_steps_low_power, reset_step_size,
                 min_power_after_reset, max_power_after_reset, min_power, reset_method, max_steps_under_min_power=3,
                  average_over=10,  number_obs_saved=4, max_episode_steps=100, timestamp=None, random_reset=True,
                 dir_names=None, save_replay=True, number_episode_to_neutral=10):
        """
        :param int max_actioninsteps: maximal action that can be taken in steps
        :param int minmirrorintervalsteps: minimum position each motor is allowed to be in
        :param int maxmirrorintervalsteps: maximum position each motor is allowed to be in
        :param min_actuators_grid_scan: lists of ints, minimum actuator position observed during scan
        :param max_actuators_grid_scan: lists of ints, maximum actuator position observed during scan
        :param list startvalues_mean: list of ints, mean of fit
        :param int initial_radius: radius in which to reset, if reset method is "interval"
        :param float reset_power_fail: If power is smaller or equal than this power is seen during a step, the agent
        failed, probably a big negative reward is given, the episode is terminated and reset is called
        :param float reset_power_goal: If power is larger or equal than this power is seen after a step, the agent
        reached the goal, probably a big positive reward is given, the episode is terminated and reset is called
        :param callable reward_fct: Reward function
        :param str reward_fct_descriptor: Descriptor of reward function, only used for log/model directory
        :param float min_power_after_reset: the minimal power we should have after reset
        :param float max_power_after_reset: the maximal power we should have after reset (at least when starting high)
        :param int max_random_reset_step_high_power:  when we have a too high power in the reset step, this is the approximate stepsize
        with which we do random steps (if reset-method == "move_power_up")
        :param int max_random_reset_step_low_power: when we have reached such a low power, this is the approximate stepsize
        with which we do random steps (if reset-method == "move_power_up")
        :param int max_random_reset_step: random steps at the end of reset (if reset-method == "move_power_up")
        :param float min_power_stop_random_steps: we stop these random steps after reaching this value
        :param float max_power_to_neutral: when resetting, this is the maximal power from which we would go back to the
        neutral positions and do random steps from there, as gradient ascent will probably fail
        :param int number_of_random_steps_low_power: when we have reached such a low power, this is the maximal number of times
        we do random steps before again going to neutral positions
        :param int reset_step_size: approximate step size for gradient ascent
        :param int average_over: number of powers in power list to average over/take the max
        :param int number_obs_saved: number of time steps we save in observation
        :param int max_episode_steps: the maximal number of steps per episode. if it's reached, the episode is truncated
        and the reset function is called
        :param None or int timestamp: Timestamp when training is first started (for logging)
        :param bool random_reset: True if we want to do random steps and move power up when resetting, False if not
        :param None or str dir_names: Name of directories to save stuff in without logs/models and timestamp in
        case we want to train a policy further after changing parameters like the episode length or goal power
        :param bool save_replay: True, if the replay buffer should be saved, else False
        """
        super(Env_fiber_simulated, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(1 + 8 * number_obs_saved,), dtype=np.float64)

        # variables
        self.max_episode_steps = max_episode_steps
        self.max_actioninsteps = max_actioninsteps
        self.minmirrorintervalsteps = minmirrorintervalsteps
        self.maxmirrorintervalsteps = maxmirrorintervalsteps
        self.max_actuators_grid_scan = max_actuators_grid_scan
        self.min_actuators_grid_scan = min_actuators_grid_scan
        self.initial_radius = initial_radius
        self.average_over = average_over
        self.startvalues_mean = startvalues_mean
        self.number_obs_saved = number_obs_saved
        self.min_power = min_power
        self.max_steps_under_min_power = max_steps_under_min_power
        self.number_episodes = 0
        self.actioninsteps = np.array([0, 0, 0, 0])
        self. reward_fct = reward_fct
        self.max_power_after_reset = max_power_after_reset
        self.min_power_after_reset = min_power_after_reset
        self.random_reset = random_reset
        self.reset_power_goal = reset_power_goal
        self.reset_power_fail = reset_power_fail
        self.reset_method = reset_method
        self.max_random_reset_step_high_power = max_random_reset_step_high_power
        self.max_random_reset_step_low_power = max_random_reset_step_low_power
        self.min_power_stop_random_steps = min_power_stop_random_steps
        self.max_power_to_neutral = max_power_to_neutral
        self.number_of_random_steps_low_power = number_of_random_steps_low_power
        self.reset_step_size = reset_step_size
        self.episode_number = 0
        self.number_episode_to_neutral = number_episode_to_neutral
        if timestamp == None:
            timestamp = int(time.time())
        self.timestamp = timestamp
        if dir_names == None:
            self.models_dir = f"models/{timestamp}"
            self.logdir = f"logs/{timestamp}"
            if save_replay:
                self.replay_dir = f"replay/{timestamp}"
        else:
            self.models_dir = "models/" + dir_names + "/" + str(timestamp)
            self.logdir = "logs/" + dir_names + "/" + str(timestamp)
            if save_replay:
                self.replay_dir = "replay/" + dir_names + "/" + str(timestamp)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if save_replay:
            if not os.path.exists(self.replay_dir):
                os.makedirs(self.replay_dir)
        self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius,
                                                          self.startvalues_mean[0] + self.initial_radius),
                                           random.randint(self.startvalues_mean[1] - self.initial_radius,
                                                          self.startvalues_mean[1] + self.initial_radius),
                                           random.randint(self.startvalues_mean[2] - self.initial_radius,
                                                          self.startvalues_mean[2] + self.initial_radius),
                                           random.randint(self.startvalues_mean[3] - self.initial_radius,
                                                          self.startvalues_mean[3] + self.initial_radius)])
        self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                    self.max_actuators_grid_scan - self.min_actuators_grid_scan)

    def step(self, action):
        self.observation = np.delete(self.observation, [i for i in range(8)])   # delete first parts of observation
        #time1=time.time()
        self.episode_steps += 1
        # test if action would lead out of the interval and clip action so that it stays in the interval (only a
        # safeguard, is this way most of the time)
        actioninsteps = np.around(action * self.max_actioninsteps)
        action_standardized = actioninsteps/(self.max_actuators_grid_scan-self.min_actuators_grid_scan)
        for i in range(4):
            if actioninsteps[i] + self.actuator_positions[i] >= self.maxmirrorintervalsteps:
                actioninsteps[i] = self.maxmirrorintervalsteps - self.actuator_positions[i]
            if actioninsteps[i] + self.actuator_positions[i] <= self.minmirrorintervalsteps:
                actioninsteps[i] = self.minmirrorintervalsteps - self.actuator_positions[i]
        actuatorstops = [self.actuator_positions_standardized+action_standardized/(self.average_over-1)*i for i in range(self.average_over)]
        # list of "measured" powers
        avg_obs = 0.0
        obs_list = []
        for stop in actuatorstops:
            obs = fit(stop[0], stop[1], stop[2], stop[3])
            avg_obs += obs / self.average_over
            obs_list.append(obs)
        obs_array = np.array(obs_list)
        argmax_obs = np.argmax(obs_array)
        max_obs = obs_list[argmax_obs]
        argmax_obs = argmax_obs / (self.average_over - 1)
        self.actuator_positions_standardized += action_standardized
        self.actuator_positions = self.actuator_positions + actioninsteps
        actionnormalized = actioninsteps / self.max_actioninsteps  # normalize action
        # append new observation
        self.observation = np.append(self.observation,
                                     np.array([actionnormalized[0], actionnormalized[1], actionnormalized[2],
                                               actionnormalized[3], avg_obs, max_obs, argmax_obs, obs]))
        # calculate reward
        reward = self.reward_fct(avg_obs, max_obs, obs, self.reset_power_fail, self.max_episode_steps,
                                 self.reset_power_goal, self.min_power_after_reset, self.episode_steps)
        # reset if agent failed or reached its goal (terminated)
        if obs < self.reset_power_fail:
            self.terminated = True
            self.fail = True
        if obs > self.reset_power_goal:
            self.terminated = True
            self.goal = True

        # reset if agent reached max. episode length (truncated)
        if self.max_episode_steps == self.episode_steps:
            self.truncated = True
        # info
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0],
                     "act_1y_pos": self.actuator_positions[1],
                     "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": obs}
        #print(self.episode_steps, actionnormalized[0], actionnormalized[1], actionnormalized[2],
        #                                       actionnormalized[3], avg_obs, max_obs, argmax_obs, obs)
        return self.observation, reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps_under_min_power = 0
        self.episode_steps = 0
        self.episode_number += 1
        self.terminated = False
        self.truncated = False
        self.fail = False
        self.goal = False
        sgn_last_action = np.sign(self.actioninsteps)
        if self.reset_method == "interval":
        # have to put mirrors to random state not too far of
            self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius, self.startvalues_mean[0] + self.initial_radius),
                                  random.randint(self.startvalues_mean[1] - self.initial_radius, self.startvalues_mean[1] + self.initial_radius),
                                  random.randint(self.startvalues_mean[2] - self.initial_radius, self.startvalues_mean[2] + self.initial_radius),
                                  random.randint(self.startvalues_mean[3] - self.initial_radius, self.startvalues_mean[3] + self.initial_radius)])
            self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan)/(self.max_actuators_grid_scan-self.min_actuators_grid_scan)
        elif self.reset_method == "move_power_up":
            # in first episode: reset to random positions in interval
            if self.number_episodes == 0:
                self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius,
                                                                  self.startvalues_mean[0] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[1] - self.initial_radius,
                                                                  self.startvalues_mean[1] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[2] - self.initial_radius,
                                                                  self.startvalues_mean[2] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[3] - self.initial_radius,
                                                                  self.startvalues_mean[3] + self.initial_radius)])
                self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                            self.max_actuators_grid_scan - self.min_actuators_grid_scan)
            power_old = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                            self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
            # first: reverse the last action if power < reset_power_fail
            if power_old < self.reset_power_fail and not self.number_episodes == 0:
                self.actuator_positions = self.actuator_positions - self.actioninsteps
                self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                        self.max_actuators_grid_scan - self.min_actuators_grid_scan)
                power_new = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
                power_old = power_new
            # second: move to neutral positions and do some random steps if power is very small or every ten episodes
            if self.episode_number % self.number_episode_to_neutral == 0 or power_old < self.max_power_to_neutral+0.05:
                power_new, self.actuator_positions, self.actuator_positions_standardized = to_neutral_positions_random_steps(
                    self.startvalues_mean, self.initial_radius,
                    self.max_power_to_neutral, self.number_of_random_steps_low_power,
                    self.max_random_reset_step_low_power, self.min_actuators_grid_scan, self.max_actuators_grid_scan,
                    self.min_power_stop_random_steps)
                power_old = power_new
            # third, if power now is high, choose a power randomly and do random steps until we are below that power
            if power_old > self.min_power_after_reset:  # case where we have high powers when resetting
                appr_reset_power = np.random.uniform(low=self.min_power_after_reset+0.1, high=self.max_power_after_reset)
                power_new = power_old
                while power_new > appr_reset_power:
                    add_random_steps = np.array([random.randint(- self.max_random_reset_step_high_power,
                                                          self.max_random_reset_step_high_power) for _ in range(4)])
                    self.actuator_positions = self.actuator_positions + add_random_steps
                    self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                                                                      self.max_actuators_grid_scan - self.min_actuators_grid_scan)
                    power_new = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
            start_dir = (-1) * sgn_last_action
            # call move_power_up (see case 2 paper, in the case of small power)
            if np.array_equal(start_dir, np.array([0, 0, 0, 0])):
                start_dir = np.array([(2 * random.randint(0, 1) - 1) for _ in range(4)])
            self.actuator_positions = move_power_up(self.actuator_positions, start_dir, self.startvalues_mean, self.initial_radius,
                      self.min_power_after_reset, self.max_power_to_neutral, self.number_of_random_steps_low_power,
                      self.max_random_reset_step_low_power, self.min_actuators_grid_scan, self.max_actuators_grid_scan,
                      self.min_power_stop_random_steps, self.reset_step_size)
            self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                self.max_actuators_grid_scan - self.min_actuators_grid_scan)
        # observation:
        obs = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
        self.observation = np.array([obs])
        for i in range(self.number_obs_saved):
            self.observation = np.append(self.observation, np.array([0.0, 0.0, 0.0, 0.0, obs, obs, 0.0, obs]))
        # averageobs_t'-1_t', obs_t' for t'=t-(number_obs_saved+1) and act0_t', act1_t', act2_t', act3_t', averageobs_t'-1_t', obs_t' for t' = t,...,t-number_obs_saved
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0],
                     "act_1y_pos": self.actuator_positions[1],
                     "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": obs}
        #print(self.info)
        return self.observation, self.info  # reward, done, info can't be included



class Env_fiber_simulated_callable_goal(gym.Env):
    """Same Environment, only P_goal is callable"""
    metadata = {'render.modes': ['human']}

    def __init__(self, max_actioninsteps, minmirrorintervalsteps, maxmirrorintervalsteps,min_actuators_grid_scan,
                 max_actuators_grid_scan, startvalues_mean, initial_radius, reset_power_fail, reset_power_goal,
                 reset_power_goal_descriptor, reward_fct, reward_fct_descriptor,
                 max_random_reset_step_high_power, max_random_reset_step_low_power, min_power_stop_random_steps, max_power_to_neutral,
                 number_of_random_steps_low_power, reset_step_size,
                 min_power_after_reset, max_power_after_reset, min_power, reset_method, max_steps_under_min_power=3,
                 average_over=10,  number_obs_saved=4, max_episode_steps=100, timestamp=None, random_reset=True, dir_names=None,
                 save_replay=True):
        super(Env_fiber_simulated_callable_goal, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(1 + 8 * number_obs_saved,), dtype=np.float64)

        # variables
        self.max_episode_steps = max_episode_steps
        self.max_actioninsteps = max_actioninsteps
        self.minmirrorintervalsteps = minmirrorintervalsteps
        self.maxmirrorintervalsteps = maxmirrorintervalsteps
        self.max_actuators_grid_scan = max_actuators_grid_scan
        self.min_actuators_grid_scan = min_actuators_grid_scan
        self.initial_radius = initial_radius
        self.average_over = average_over
        self.startvalues_mean = startvalues_mean
        self.number_obs_saved = number_obs_saved
        self.min_power = min_power
        self.max_steps_under_min_power = max_steps_under_min_power
        self.number_episodes = 0
        self.actioninsteps = np.array([0, 0, 0, 0])
        self. reward_fct = reward_fct
        self.max_power_after_reset = max_power_after_reset
        self.min_power_after_reset = min_power_after_reset
        self.random_reset = random_reset
        self.reset_power_goal = reset_power_goal
        self.reset_power_fail = reset_power_fail
        self.reset_method = reset_method
        self.max_random_reset_step_high_power = max_random_reset_step_high_power
        self.max_random_reset_step_low_power = max_random_reset_step_low_power
        self.min_power_stop_random_steps = min_power_stop_random_steps
        self.max_power_to_neutral = max_power_to_neutral
        self.number_of_random_steps_low_power = number_of_random_steps_low_power
        self.reset_step_size = reset_step_size
        self.episode_number = 0
        self.total_steps = 0
        if timestamp == None:
            timestamp = int(time.time())
        self.timestamp = timestamp
        if dir_names == None:
            self.models_dir = f"models/{timestamp}"
            self.logdir = f"logs/{timestamp}"
            if save_replay:
                self.replay_dir = f"replay/{timestamp}"
        else:
            self.models_dir = "models/" + dir_names + "/" + str(timestamp)
            self.logdir = "logs/" + dir_names + "/" + str(timestamp)
            if save_replay:
                self.replay_dir = "replay/" + dir_names + "/" + str(timestamp)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if save_replay:
            if not os.path.exists(self.replay_dir):
                os.makedirs(self.replay_dir)
        self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius,
                                                          self.startvalues_mean[0] + self.initial_radius),
                                           random.randint(self.startvalues_mean[1] - self.initial_radius,
                                                          self.startvalues_mean[1] + self.initial_radius),
                                           random.randint(self.startvalues_mean[2] - self.initial_radius,
                                                          self.startvalues_mean[2] + self.initial_radius),
                                           random.randint(self.startvalues_mean[3] - self.initial_radius,
                                                          self.startvalues_mean[3] + self.initial_radius)])
        self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                    self.max_actuators_grid_scan - self.min_actuators_grid_scan)

    def step(self, action):
        self.total_steps += 1
        self.observation = np.delete(self.observation, [i for i in range(8)])
        #time1=time.time()
        self.episode_steps += 1
        # test if action would lead out of the interval
        actioninsteps = np.around(action * self.max_actioninsteps)
        action_standardized = actioninsteps/(self.max_actuators_grid_scan-self.min_actuators_grid_scan)
        for i in range(4):
            if actioninsteps[i] + self.actuator_positions[i] >= self.maxmirrorintervalsteps:
                actioninsteps[i] = self.maxmirrorintervalsteps - self.actuator_positions[i]
            if actioninsteps[i] + self.actuator_positions[i] <= self.minmirrorintervalsteps:
                actioninsteps[i] = self.minmirrorintervalsteps - self.actuator_positions[i]
        actuatorstops = [self.actuator_positions_standardized+action_standardized/(self.average_over-1)*i for i in range(self.average_over)]
        avg_obs = 0.0
        obs_list = []
        for stop in actuatorstops:
            obs = fit(stop[0], stop[1], stop[2], stop[3])
            avg_obs += obs / self.average_over
            obs_list.append(obs)
        obs_array = np.array(obs_list)
        argmax_obs = np.argmax(obs_array)
        max_obs = obs_list[argmax_obs]
        argmax_obs = argmax_obs / (self.average_over - 1)
        self.actuator_positions_standardized += action_standardized
        self.actuator_positions = self.actuator_positions + actioninsteps
        actionnormalized = actioninsteps / (self.max_actioninsteps)
        self.observation = np.append(self.observation,
                                     np.array([actionnormalized[0], actionnormalized[1], actionnormalized[2],
                                               actionnormalized[3], avg_obs, max_obs, argmax_obs, obs]))
        # calculate reward
        #reward = avg_obs - np.log(1 - avg_obs) - 1  # what reward? #right now same as in interferobot paper
        reward = self.reward_fct(avg_obs, max_obs, obs, self.reset_power_fail, self.max_episode_steps,
                                 self.reset_power_goal(self.total_steps), self.min_power_after_reset, self.episode_steps)
        if obs < self.reset_power_fail:
            self.terminated = True
            self.fail = True
        if obs > self.reset_power_goal(self.total_steps):
            self.terminated = True
            self.goal = True

        # done:
        if self.max_episode_steps == self.episode_steps:
            self.truncated = True
        # info
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0],
                     "act_1y_pos": self.actuator_positions[1],
                     "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": obs}
        #print(self.episode_steps, actionnormalized[0], actionnormalized[1], actionnormalized[2],
        #                                       actionnormalized[3], avg_obs, max_obs, argmax_obs, obs)
        return self.observation, reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps_under_min_power = 0
        self.episode_steps = 0
        self.episode_number += 1
        self.terminated = False
        self.truncated = False
        self.fail = False
        self.goal = False
        sgn_last_action = np.sign(self.actioninsteps)
        if self.reset_method == "interval":
        # have to put mirrors to random state not too far of
            self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius, self.startvalues_mean[0] + self.initial_radius),
                                  random.randint(self.startvalues_mean[1] - self.initial_radius, self.startvalues_mean[1] + self.initial_radius),
                                  random.randint(self.startvalues_mean[2] - self.initial_radius, self.startvalues_mean[2] + self.initial_radius),
                                  random.randint(self.startvalues_mean[3] - self.initial_radius, self.startvalues_mean[3] + self.initial_radius)])
            self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan)/(self.max_actuators_grid_scan-self.min_actuators_grid_scan)
        elif self.reset_method == "move_power_up":
            if self.number_episodes == 0:
                self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius,
                                                                  self.startvalues_mean[0] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[1] - self.initial_radius,
                                                                  self.startvalues_mean[1] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[2] - self.initial_radius,
                                                                  self.startvalues_mean[2] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[3] - self.initial_radius,
                                                                  self.startvalues_mean[3] + self.initial_radius)])
                self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                            self.max_actuators_grid_scan - self.min_actuators_grid_scan)
            power_old = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                            self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
            if power_old < self.reset_power_fail and not self.number_episodes == 0:
                self.actuator_positions = self.actuator_positions - self.actioninsteps
                self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                        self.max_actuators_grid_scan - self.min_actuators_grid_scan)
                power_new = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
                power_old = power_new
            if self.episode_number % 10 == 0 or power_old < self.max_power_to_neutral+0.05:
                power_new, self.actuator_positions, self.actuator_positions_standardized = to_neutral_positions_random_steps(
                    self.startvalues_mean, self.initial_radius,
                    self.max_power_to_neutral, self.number_of_random_steps_low_power,
                    self.max_random_reset_step_low_power, self.min_actuators_grid_scan, self.max_actuators_grid_scan,
                    self.min_power_stop_random_steps)
                power_old = power_new
            if power_old > self.min_power_after_reset:  # case where we have high powers when resetting
                appr_reset_power = np.random.uniform(low=self.min_power_after_reset+0.1,
                                                     high=self.max_power_after_reset(self.total_steps))
                power_new = power_old
                while power_new > appr_reset_power:
                    add_random_steps = np.array([random.randint(- self.max_random_reset_step_high_power,
                                                          self.max_random_reset_step_high_power) for _ in range(4)])
                    self.actuator_positions = self.actuator_positions + add_random_steps
                    self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                                                                      self.max_actuators_grid_scan - self.min_actuators_grid_scan)
                    power_new = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
            start_dir = (-1) * sgn_last_action
            if np.array_equal(start_dir, np.array([0, 0, 0, 0])):
                start_dir = np.array([(2 * random.randint(0, 1) - 1) for _ in range(4)])
            self.actuator_positions = move_power_up(self.actuator_positions, start_dir, self.startvalues_mean, self.initial_radius,
                      self.min_power_after_reset, self.max_power_to_neutral, self.number_of_random_steps_low_power,
                      self.max_random_reset_step_low_power, self.min_actuators_grid_scan, self.max_actuators_grid_scan,
                      self.min_power_stop_random_steps, self.reset_step_size)
            self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                self.max_actuators_grid_scan - self.min_actuators_grid_scan)
        # observation:
        obs = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
        self.observation = np.array([obs])
        for i in range(self.number_obs_saved):
            self.observation = np.append(self.observation, np.array([0.0, 0.0, 0.0, 0.0, obs, obs, 0.0, obs]))
        # averageobs_t'-1_t', obs_t' for t'=t-(number_obs_saved+1) and act0_t', act1_t', act2_t', act3_t', averageobs_t'-1_t', obs_t' for t' = t,...,t-number_obs_saved
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0],
                     "act_1y_pos": self.actuator_positions[1],
                     "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": obs}
        #print(self.info)
        return self.observation, self.info  # reward, done, info can't be included

class Env_fiber_simulated_diff_obs(gym.Env):
    """Same Environment with different observations"""
    metadata = {'render.modes': ['human']}

    def __init__(self, max_actioninsteps, minmirrorintervalsteps, maxmirrorintervalsteps,min_actuators_grid_scan,
                 max_actuators_grid_scan, startvalues_mean, initial_radius, reset_power_fail, reset_power_goal,
                 reward_fct, reward_fct_descriptor,
                 max_random_reset_step_high_power, max_random_reset_step_low_power, min_power_stop_random_steps, max_power_to_neutral,
                 number_of_random_steps_low_power, reset_step_size,
                 min_power_after_reset, max_power_after_reset, min_power, reset_method, max_steps_under_min_power=3,
                 average_over=10,  number_obs_saved=4, max_episode_steps=100, timestamp=None, random_reset=True, dir_names=None,
                 save_replay=True, obs_P_ave=True, obs_P_max=True, obs_x_max=True):
        super(Env_fiber_simulated_diff_obs, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(1 + (5+int(obs_P_ave)+int(obs_x_max)+int(obs_P_max))*number_obs_saved,),
                                            dtype=np.float64)

        # variables
        self.max_episode_steps = max_episode_steps
        self.max_actioninsteps = max_actioninsteps
        self.minmirrorintervalsteps = minmirrorintervalsteps
        self.maxmirrorintervalsteps = maxmirrorintervalsteps
        self.max_actuators_grid_scan = max_actuators_grid_scan
        self.min_actuators_grid_scan = min_actuators_grid_scan
        self.initial_radius = initial_radius
        self.average_over = average_over
        self.startvalues_mean = startvalues_mean
        self.number_obs_saved = number_obs_saved
        self.min_power = min_power
        self.max_steps_under_min_power = max_steps_under_min_power
        self.number_episodes = 0
        self.actioninsteps = np.array([0, 0, 0, 0])
        self. reward_fct = reward_fct
        self.max_power_after_reset = max_power_after_reset
        self.min_power_after_reset = min_power_after_reset
        self.random_reset = random_reset
        self.reset_power_goal = reset_power_goal
        self.reset_power_fail = reset_power_fail
        self.reset_method = reset_method
        self.max_random_reset_step_high_power = max_random_reset_step_high_power
        self.max_random_reset_step_low_power = max_random_reset_step_low_power
        self.min_power_stop_random_steps = min_power_stop_random_steps
        self.max_power_to_neutral = max_power_to_neutral
        self.number_of_random_steps_low_power = number_of_random_steps_low_power
        self.reset_step_size = reset_step_size
        self.episode_number = 0
        self.total_steps = 0
        self.obs_P_ave = obs_P_ave
        self.obs_P_max = obs_P_max
        self.obs_x_max = obs_x_max
        if timestamp == None:
            timestamp = int(time.time())
        self.timestamp = timestamp
        if dir_names == None:
            self.models_dir = f"models/{timestamp}"
            self.logdir = f"logs/{timestamp}"
            if save_replay:
                self.replay_dir = f"replay/{timestamp}"
        else:
            self.models_dir = "models/" + dir_names + "/" + str(timestamp)
            self.logdir = "logs/" + dir_names + "/" + str(timestamp)
            if save_replay:
                self.replay_dir = "replay/" + dir_names + "/" + str(timestamp)
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if save_replay:
            if not os.path.exists(self.replay_dir):
                os.makedirs(self.replay_dir)
        self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius,
                                                          self.startvalues_mean[0] + self.initial_radius),
                                           random.randint(self.startvalues_mean[1] - self.initial_radius,
                                                          self.startvalues_mean[1] + self.initial_radius),
                                           random.randint(self.startvalues_mean[2] - self.initial_radius,
                                                          self.startvalues_mean[2] + self.initial_radius),
                                           random.randint(self.startvalues_mean[3] - self.initial_radius,
                                                          self.startvalues_mean[3] + self.initial_radius)])
        self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                    self.max_actuators_grid_scan - self.min_actuators_grid_scan)

    def step(self, action):
        self.total_steps += 1
        self.observation = np.delete(self.observation, [i for i in range(5+int(self.obs_P_ave)
                                                                         +int(self.obs_x_max)+int(self.obs_P_max))])
        #time1=time.time()
        self.episode_steps += 1
        # test if action would lead out of the interval
        actioninsteps = np.around(action * self.max_actioninsteps)
        action_standardized = actioninsteps/(self.max_actuators_grid_scan-self.min_actuators_grid_scan)
        for i in range(4):
            if actioninsteps[i] + self.actuator_positions[i] >= self.maxmirrorintervalsteps:
                actioninsteps[i] = self.maxmirrorintervalsteps - self.actuator_positions[i]
            if actioninsteps[i] + self.actuator_positions[i] <= self.minmirrorintervalsteps:
                actioninsteps[i] = self.minmirrorintervalsteps - self.actuator_positions[i]
        actuatorstops = [self.actuator_positions_standardized+action_standardized/(self.average_over-1)*i for i in range(self.average_over)]
        avg_obs = 0.0
        obs_list = []
        for stop in actuatorstops:
            obs = fit(stop[0], stop[1], stop[2], stop[3])
            avg_obs += obs / self.average_over
            obs_list.append(obs)
        obs_array = np.array(obs_list)
        argmax_obs = np.argmax(obs_array)
        max_obs = obs_list[argmax_obs]
        argmax_obs = argmax_obs / (self.average_over - 1)
        self.actuator_positions_standardized += action_standardized
        self.actuator_positions = self.actuator_positions + actioninsteps
        actionnormalized = actioninsteps / (self.max_actioninsteps)
        append_list = [actionnormalized[0], actionnormalized[1], actionnormalized[2],
                       actionnormalized[3]]
        if self.obs_P_ave:
            append_list.append(avg_obs)
        if self.obs_P_max:
            append_list.append(max_obs)
        if self.obs_x_max:
            append_list.append(argmax_obs)
        append_list.append(obs)
        self.observation = np.append(self.observation,
                                     np.array(append_list))
        # calculate reward
        #reward = avg_obs - np.log(1 - avg_obs) - 1  # what reward? #right now same as in interferobot paper
        reward = self.reward_fct(avg_obs, max_obs, obs, self.reset_power_fail, self.max_episode_steps,
                                 self.reset_power_goal, self.min_power_after_reset, self.episode_steps)
        if obs < self.reset_power_fail:
            self.terminated = True
            self.fail = True
        if obs > self.reset_power_goal:
            self.terminated = True
            self.goal = True

        # done:
        if self.max_episode_steps == self.episode_steps:
            self.truncated = True
        # info
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0],
                     "act_1y_pos": self.actuator_positions[1],
                     "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": obs}
        #print(self.episode_steps, actionnormalized[0], actionnormalized[1], actionnormalized[2],
        #                                       actionnormalized[3], avg_obs, max_obs, argmax_obs, obs)
        return self.observation, reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.steps_under_min_power = 0
        self.episode_steps = 0
        self.episode_number += 1
        self.terminated = False
        self.truncated = False
        self.fail = False
        self.goal = False
        sgn_last_action = np.sign(self.actioninsteps)
        if self.reset_method == "interval":
        # have to put mirrors to random state not too far of
            self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius, self.startvalues_mean[0] + self.initial_radius),
                                  random.randint(self.startvalues_mean[1] - self.initial_radius, self.startvalues_mean[1] + self.initial_radius),
                                  random.randint(self.startvalues_mean[2] - self.initial_radius, self.startvalues_mean[2] + self.initial_radius),
                                  random.randint(self.startvalues_mean[3] - self.initial_radius, self.startvalues_mean[3] + self.initial_radius)])
            self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan)/(self.max_actuators_grid_scan-self.min_actuators_grid_scan)
        elif self.reset_method == "move_power_up":
            if self.number_episodes == 0:
                self.actuator_positions = np.array([random.randint(self.startvalues_mean[0] - self.initial_radius,
                                                                  self.startvalues_mean[0] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[1] - self.initial_radius,
                                                                  self.startvalues_mean[1] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[2] - self.initial_radius,
                                                                  self.startvalues_mean[2] + self.initial_radius),
                                                   random.randint(self.startvalues_mean[3] - self.initial_radius,
                                                                  self.startvalues_mean[3] + self.initial_radius)])
                self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                            self.max_actuators_grid_scan - self.min_actuators_grid_scan)
            power_old = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                            self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
            if power_old < self.reset_power_fail and not self.number_episodes == 0:
                self.actuator_positions = self.actuator_positions - self.actioninsteps
                self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                        self.max_actuators_grid_scan - self.min_actuators_grid_scan)
                power_new = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
                power_old = power_new
            if self.episode_number % 10 == 0 or power_old < self.max_power_to_neutral+0.05:
                power_new, self.actuator_positions, self.actuator_positions_standardized = to_neutral_positions_random_steps(
                    self.startvalues_mean, self.initial_radius,
                    self.max_power_to_neutral, self.number_of_random_steps_low_power,
                    self.max_random_reset_step_low_power, self.min_actuators_grid_scan, self.max_actuators_grid_scan,
                    self.min_power_stop_random_steps)
                power_old = power_new
            if power_old > self.min_power_after_reset:  # case where we have high powers when resetting
                appr_reset_power = np.random.uniform(low=self.min_power_after_reset+0.1,
                                                     high=self.max_power_after_reset)
                power_new = power_old
                while power_new > appr_reset_power:
                    add_random_steps = np.array([random.randint(- self.max_random_reset_step_high_power,
                                                          self.max_random_reset_step_high_power) for _ in range(4)])
                    self.actuator_positions = self.actuator_positions + add_random_steps
                    self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                                                                      self.max_actuators_grid_scan - self.min_actuators_grid_scan)
                    power_new = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
            start_dir = (-1) * sgn_last_action
            if np.array_equal(start_dir, np.array([0, 0, 0, 0])):
                start_dir = np.array([(2 * random.randint(0, 1) - 1) for _ in range(4)])
            self.actuator_positions = move_power_up(self.actuator_positions, start_dir, self.startvalues_mean, self.initial_radius,
                      self.min_power_after_reset, self.max_power_to_neutral, self.number_of_random_steps_low_power,
                      self.max_random_reset_step_low_power, self.min_actuators_grid_scan, self.max_actuators_grid_scan,
                      self.min_power_stop_random_steps, self.reset_step_size)
            self.actuator_positions_standardized = (self.actuator_positions - self.min_actuators_grid_scan) / (
                self.max_actuators_grid_scan - self.min_actuators_grid_scan)
        # observation:
        
        obs = fit(self.actuator_positions_standardized[0], self.actuator_positions_standardized[1],
                        self.actuator_positions_standardized[2], self.actuator_positions_standardized[3])
        self.observation = np.array([obs])
        append_list = [0, 0, 0, 0]
        if self.obs_P_ave:
            append_list.append(obs)
        if self.obs_P_max:
            append_list.append(obs)
        if self.obs_x_max:
            append_list.append(0.0)
        append_list.append(obs)
        for i in range(self.number_obs_saved):
            self.observation = np.append(self.observation, np.array(append_list))
        # averageobs_t'-1_t', obs_t' for t'=t-(number_obs_saved+1) and act0_t', act1_t', act2_t', act3_t', averageobs_t'-1_t', obs_t' for t' = t,...,t-number_obs_saved
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0],
                     "act_1y_pos": self.actuator_positions[1],
                     "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": obs}
        #print(self.info)
        return self.observation, self.info  # reward, done, info can't be included


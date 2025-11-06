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

    #### CHANGE ARGUMENTS IN OTHER SCRIPT WHEN INSTANTIATING CLASS, NOT THE DEFAULTS 

    def __init__(self, actuators, pds, 
                 max_actioninsteps, 
                 reset_power_fail, reset_power_goal,
                 reward_fct, reward_fct_descriptor, 
                 min_power_after_reset, max_power_after_reset,
                 mirror_pos_lower_bound = -3 * 10 ** 5, mirror_pos_upper_bound = 3 * 10 ** 5,
                 neutral_positions=[neutralxm1, neutralym1, neutralxm2, neutralym2],
                 ref_pd_intercept=0, ref_pd_slope=1,
                 min_ref_power=1 * 10 ** (-1), 
                 grad_ascent_step_size = 5 * 10 ** 1,
                 extra_random_step_magnitude = 5 * 10 ** 1, 
                 min_actioninsteps = 1, max_power_to_neutral = 0.01,
                 number_of_random_actions_low_power=10, 
                 min_power_stop_random_actions_neutral_failure=0.04,
                 neutral_flailing_step_magnitude=int(100), high_power_flailing_step_magnitude=int(50),
                 wait_time_pd=0, 
                 time_wait_pd=0,
                 number_obs_saved=4, 
                 max_cycles_per_episode=20, 
                 timestamp=None,
                 random_reset=True, 
                 dir_names=None, 
                 save_replay=True):
        """
        a 'step' is a movement of a motor
        an 'action' is a change in the position of all four motors
        a 'cycle' is an action followed by readout of reward
        an 'episode' is a sequence of cycles terminating in failure (low power) or success (high power)

        :param list actuators: list of objects, e.g. pylablib stages, representing the actuators (act_1x, act_1y, act_2x, act_2y)
        that should have class functions, is_moving, move_by, move_to, get_position, wait_move
        :param list pds: list of objects representing photodetectors/powermeters (reference pd, pd after fiber)
        that should have class functions get_power, close
        :param int max_actioninsteps: maximal action that can be taken in steps
        :param int min_actioninsteps: minimal action that can be taken in steps
        :param int mirror_pos_lower_bound: minimum position each motor is allowed to be in
        :param int mirror_pos_upper_bound: maximum position each motor is allowed to be in
        :param list neutral_positions: list of ints, position of actuators with the maximum near it (after trying to couple
        fiber as a human) before first reset (afterwards backlash kills meaning of absolute positions)
        :param float reset_power_fail: If power smaller or equal than this power is seen during a cycle, the agent failed,
        probably a big negative reward is given, the episode is terminated and reset is called
        :param float reset_power_goal: If power is larger or equal than this power is seen after a cycle, the agent reached the goal,
        probably a big positive reward is given, the episode is terminated and reset is called
        :param callable reward_fct: Reward function
        :param str reward_fct_descriptor: Descriptor of reward function, only used for log/model directory
        :param float ref_pd_intercept: the maximum power is a linear function of the reference power, this is the offset
        :param float ref_pd_slope: the maximum power is a linear function of the reference power, this is the slope
        :param float min_ref_power: minimal reference power. If we are under that value, probably no light is coming
        to the experiment
        :param int grad_ascent_step_size: approximate step size for gradient ascent
        :param float min_power_after_reset: the minimal power we should have after reset
        :param float max_power_after_reset: the maximal power we should have after reset (at least when starting high)
        :param int extra_random_step_magnitude: random motor steps at the end of reset
        :param float max_power_to_neutral: when resetting, this is the maximal power from which we would go back to the
        neutral positions and do random motor steps from there, as gradient ascent will probably fail
        :param int number_of_random_actions_low_power: when we have reached such a low power, this is the maximal number of times
        we do random actions before again going to neutral positions
        :param float min_power_stop_random_actions_neutral_failure: we stop these random actions after reaching this value
        :param int neutral_flailing_step_magnitude: when we have reached such a low power, this is the approximate stepsize
        with which we do random steps
        :param int high_power_flailing_step_magnitude:  when we have a too high power in the reset step, this is the approximate stepsize
        with which we do random steps
        :param float wait_time_pd: time between pd measurements
        :param float time_wait_pd: time to wait for pd measurements after moving actuators
        :param int number_obs_saved: number of time steps we save in observation
        :param int max_cycles_per_episode: the maximal number of cycles per episode. if it's reached, the episode is truncated
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
        self.max_cycles_per_episode = max_cycles_per_episode
        self.max_actioninsteps = max_actioninsteps
        self.mirror_pos_lower_bound = mirror_pos_lower_bound
        self.mirror_pos_upper_bound = mirror_pos_upper_bound
        self.neutral_positions = neutral_positions
        self.wait_time_pd = wait_time_pd
        self.time_wait_pd = time_wait_pd
        self.reset_power_fail = reset_power_fail
        self.reset_power_goal = reset_power_goal
        self.min_ref_power = min_ref_power
        self.ref_pd_slope = ref_pd_slope
        self.ref_pd_intercept = ref_pd_intercept
        self.number_obs_saved = number_obs_saved
        self.reward_fct = reward_fct
        self.number_episodes = 0
        self.random_reset = random_reset
        self.grad_ascent_step_size = grad_ascent_step_size
        self.min_power_after_reset = min_power_after_reset
        self.extra_random_step_magnitude = extra_random_step_magnitude
        self.min_actioninsteps = min_actioninsteps
        self.actioninsteps = np.array([0, 0, 0, 0])
        self.max_power_after_reset = max_power_after_reset
        self.max_power_to_neutral = max_power_to_neutral
        self.number_of_random_actions_low_power = number_of_random_actions_low_power
        self.min_power_stop_random_actions_neutral_failure = min_power_stop_random_actions_neutral_failure
        self.neutral_flailing_step_magnitude = neutral_flailing_step_magnitude
        self.high_power_flailing_step_magnitude = high_power_flailing_step_magnitude

        self.reset_times = np.zeros(10)
        self.reset_time_rolling_average = 0.
        self.power_ratio = 0.
        #self.find_new_neutral_position = False

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

    def check_new_neutral(self,threshold=None):
        if threshold is None:
            threshold = self.min_power_after_reset + .1
        self.power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
        if self.power_ratio > threshold and self.power_ratio < self.max_power_after_reset - .3:
            for j in range(4):
                self.neutral_positions[j] = self.actuators.motor_params[j + 1]['pos']
            print("Found new neutral positions:", self.neutral_positions)
        return self.neutral_positions

    def random_flailing(self, step_magnitude):
        number_movements = 0
        for i in range(4):
            add_random_steps = random.randint(- step_magnitude, step_magnitude)
            direction = 1 if np.sign(add_random_steps) >= 0 else 0
            self.actuators.move_stepper(i+1, direction, int(np.abs(add_random_steps)))
            number_movements += 1
            # the next episode will start even if these random steps move below min_power_after_reset.
            # That's good, so we have different start conditions.
            print(f'Actuator {i}: {add_random_steps} random steps moved.')

        time.sleep(self.time_wait_pd)
        return number_movements
     
    def actuator_action(self, actioninsteps, reverse = False, check_new_neutral = False):

        number_movements = 0
        if not reverse:
            sign = 1
            printout = 'stepping motor'
        else:
            sign = -1
            printout = 'reverse motor'

        for i in range(4):
            steps = sign * (actioninsteps[i])
            direction = 1 if np.sign(steps) >= 0 else 0
            self.actuators.move_stepper(i+1, direction, int(np.abs(steps)))
            print(f'{printout} {i}: {steps}')
            number_movements += 1

        time.sleep(self.time_wait_pd)
        if check_new_neutral:
            self.check_new_neutral()

        return number_movements

    def step(self, action):

        print('\n >>>>> Starting next Action <<<<< \n')
        how_long_ref_power_under_min_ref_power = 0
        # test if we have reference power. Otherwise, wait until we have (in case no laser beam gets to experiment)
        while self.pds.get_measurement()[0][-1] < self.min_ref_power:
            time.sleep(self.wait_time_pd)
            how_long_ref_power_under_min_ref_power += 1
            if how_long_ref_power_under_min_ref_power > 10:
                warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
        self.episode_steps += 1
        # get max power from reference powermeter
        self.max_power = (self.pds.get_measurement()[0][-1]) * self.ref_pd_slope + self.ref_pd_intercept
        # delete first parts of observation
        self.observation = np.delete(self.observation, [i for i in range(8)])
        # calculate the action in steps from the normalized action
        self.actioninsteps = np.around(action * self.max_actioninsteps).astype(int)
        print(f'Current action in steps: {self.actioninsteps}')
        # test if action would lead out of the interval and clip action so that it stays in the interval (only a
        # safeguard, is this way most of the time)
        self.actuator_positions = [self.actuators.motor_params[i+1]['pos'] for i in range(4)]
        for i in range(4):
            if self.actioninsteps[i] + self.actuator_positions[i] >= self.mirror_pos_upper_bound:
                self.actioninsteps[i] = self.mirror_pos_upper_bound - self.actuator_positions[i]
            if self.actioninsteps[i] + self.actuator_positions[i] <= self.mirror_pos_lower_bound:
                self.actioninsteps[i] = self.mirror_pos_lower_bound - self.actuator_positions[i]
        # perform action
        number_movements = self.actuator_action(self.actioninsteps, reverse = False, check_new_neutral = True)
        time.sleep(self.time_wait_pd)
        # get power from last second of measurement
        power_list = (self.pds.get_measurement()[1]) / self.max_power
        time.sleep(self.wait_time_pd)
        # calculate argmax, max, ave for observation
        power_array = np.array(power_list)
        power_argmax = np.argmax(power_array)
        power_meas_max = power_list[power_argmax]
        power_argmax = power_argmax/len(power_list)
        power_ave = np.mean(power_array)
        power = power_list[-1]

        # normalise action for observation
        actionnormalized = self.actioninsteps / self.max_actioninsteps

        # append last history points
        self.observation = np.append(self.observation, np.array([actionnormalized[0], actionnormalized[1], actionnormalized[2],
                                                                 actionnormalized[3], power_ave, power_meas_max, power_argmax, power]))

        # calculate reward
        reward = self.reward_fct(power_ave, power_meas_max, power, self.reset_power_fail, self.max_cycles_per_episode,
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
        if self.max_cycles_per_episode == self.episode_steps:
            self.truncated = True
        # info 
        self.info = {"episode_step": self.episode_steps, "act_1x_pos": self.actuator_positions[0], "act_1y_pos": self.actuator_positions[1],
                 "act_2x_pos": self.actuator_positions[2], "act_2y_pos": self.actuator_positions[3], "power": power}
        print(self.info)
        return self.observation, reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None):
        print( '\n >>>>> Reset called <<<<< \n')
        start_time = time.time()
        super().reset(seed=seed)
        time_reset_start = time.time()
        self.episode_number += 1
        self.number_reset_movements = 0  # number of movements performed during the reset
        # test if reference powermeter has power, otherwise wait until it has
        how_long_ref_power_under_min_ref_power = 0
        while self.pds.get_measurement()[0][-1] < self.min_ref_power:
            time.sleep(self.wait_time_pd)
            how_long_ref_power_under_min_ref_power += 1
            if how_long_ref_power_under_min_ref_power > 10:
                warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
        # calculate max power from ref power
        self.max_power = (self.pds.get_measurement()[0][-1]) * self.ref_pd_slope + self.ref_pd_intercept
        self.episode_steps = 0
        self.terminated = False
        self.truncated = False
        # move actuators for reset

        print(f'Resetting. Current action in steps: {self.actioninsteps}')  # L: added #not last?
        if self.random_reset:
            self.power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
            print(f'Power when reset is called: {self.power_ratio}')
            ###############
            # first: reverse the last action if power < reset_power_fail
            if self.power_ratio < self.reset_power_fail and not np.array_equal(self.actioninsteps, np.array([0, 0, 0, 0])):  # this should only be done if the reset is called because of low power.
                number_movements = self.actuator_action(self.actioninsteps, reverse = True, check_new_neutral = True)
                self.number_reset_movements += number_movements
                self.power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
                print(f'Power after reversing last action: {self.power_ratio}')

            # second: move to neutral positions and do some random steps if power is very small or every ten episodes
            if self.episode_number % 5 == 0 or self.power_ratio < self.max_power_to_neutral+0.05:
                number_moves_to_neutral, power_ratio = to_neutral_positions_random_steps(self.pds, self.actuators,
                                                                                       self.max_power,
                                                                                       self.neutral_positions,
                                                                                       self.max_power_to_neutral,
                                                                                       self.number_of_random_actions_low_power,
                                                                                       self.neutral_flailing_step_magnitude,
                                                                                       self.min_power_stop_random_actions_neutral_failure,
                                                                                       self.time_wait_pd)
                self.number_reset_movements += number_moves_to_neutral
                self.power_ratio = power_ratio

            # third, if power now is high, choose a power randomly and do random steps until we are below that power
            time.sleep(self.time_wait_pd)
            if self.power_ratio > self.min_power_after_reset:  # case where we have high powers when resetting
                self.check_new_neutral()
                appr_reset_power = np.random.uniform(low = self.min_power_after_reset+0.1, high = self.max_power_after_reset)
                print('HIGH POWER PROTOCOL, we want to have reset power < '+str(appr_reset_power))
                while self.power_ratio > appr_reset_power:
                    number_movements = self.random_flailing(self.high_power_flailing_step_magnitude)
                    self.number_reset_movements += number_movements
                    self.check_new_neutral()
            # call grad_ascent (see case 2 paper, in the case of small power)
            ## we pass it the last actioninsteps as starting input as well
            # try simple grad ascent and scuffed beamwalking a couple of times first
            if self.power_ratio < self.min_power_after_reset:
                simple_grad_ascent(self.pds, self.actuators, move_increment=self.grad_ascent_step_size)
                number_grad_ascent_movements = 0  # we don't count the movements inside simple_grad_ascent for now
                power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
                self.power_ratio = power_ratio

            self.check_new_neutral(threshold=self.min_power_after_reset)

            if self.power_ratio > .001:
                if self.power_ratio < self.min_power_after_reset:
                    power_history = scuffed_beamwalking(self.actuators, 
                                                        self.pds, 
                                                        goal_power=self.min_power_after_reset, move_increment= self.grad_ascent_step_size+5)
                    power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
                    self.power_ratio = power_ratio

                self.check_new_neutral(threshold=self.min_power_after_reset)

                if self.power_ratio < self.min_power_after_reset:
                    simple_grad_ascent(self.pds, self.actuators, move_increment=self.grad_ascent_step_size)
                    number_grad_ascent_movements = 0  # we don't count the movements inside simple_grad_ascent for now
                    power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
                    self.power_ratio = power_ratio

                self.check_new_neutral(threshold=self.min_power_after_reset)

                if self.power_ratio < self.min_power_after_reset:
                    power_history = scuffed_beamwalking(self.actuators, 
                                                        self.pds, 
                                                        goal_power=self.min_power_after_reset, move_increment= self.grad_ascent_step_size+5)
                power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
                self.power_ratio = power_ratio

                self.check_new_neutral(threshold=self.min_power_after_reset)
            
            if self.power_ratio < self.min_power_after_reset:
                number_grad_ascent_movements, power_ratio = grad_ascent(self.pds, self.actuators, 
                                                                self.max_power, self.actioninsteps, self.neutral_positions, self.min_power_after_reset,
                                                                self.max_power_to_neutral,
                                                                self.number_of_random_actions_low_power, 
                                                                self.neutral_flailing_step_magnitude,
                                                                self.min_power_stop_random_actions_neutral_failure, 3*self.grad_ascent_step_size, self.min_ref_power,
                                                                self.wait_time_pd,
                                                                self.time_wait_pd, 
                                                                self.ref_pd_slope, self.ref_pd_intercept)
            
            # update neutral positions if necessary
            self.check_new_neutral(threshold=self.min_power_after_reset)
            # Do some random steps (to get some extra randomness...)
            number_movements = self.random_flailing(self.extra_random_step_magnitude)
            self.number_reset_movements += number_movements
            self.check_new_neutral()

        # observation:
        self.actuator_positions = [self.actuators.motor_params[i+1]['pos'] for i in range(4)]
        self.power_ratio = self.pds.get_measurement()[1][-1] / self.max_power
        self.observation = np.array([self.power_ratio])
        for i in range(self.number_obs_saved):
            self.observation = np.append(self.observation, np.array([0.0, 0.0, 0.0, 0.0, self.power_ratio, self.power_ratio, 0.0, self.power_ratio]))
        # the [0, 0, 0, 0] corresponds to the fact that in the initial observation, before the first episode step
        # happens, there is no action
        # obs_t' for t'=t-(number_obs_saved+1) and act0_t', act1_t', act2_t', act3_t', average_power_t'-1_t',
        # max_power_t'-1_t', maxpos_t'-1_t', power_t' for t' = t,...,t-number_obs_saved
        # info
        self.info = {"episode_step": self.episode_steps, 
                     "act_1y_pos": self.actuators.motor_params[1]['pos'],
                     "act_1x_pos": self.actuators.motor_params[2]['pos'],
                     "act_2y_pos": self.actuators.motor_params[3]['pos'], 
                     "act_2x_pos": self.actuators.motor_params[4]['pos'], 
                     "power": self.power_ratio}
        
        print(self.info)
        time_reset_end = time.time()
        # save how long reset took
        print(time_reset_end-time_reset_start, self.number_reset_movements)
        self.df = self.df._append({"episode": self.episode_number,
                                  "number_movements_reset": self.number_reset_movements,
                                  "time_reset":time_reset_end-time_reset_start}, ignore_index = True)
        self.df.to_csv(f"reset_time_{self.timestamp}.csv")
        self.goal = False
        self.fail = False

        self.reset_times = np.roll(self.reset_times, -1)
        self.reset_times[-1] = time.time()-start_time
        self.reset_time_rolling_average = np.mean(self.reset_times)

        print("\n >>>>> Reset finished! Took {:.3f}s <<<<< \n".format(self.reset_times[-1]))
        print(" >>>>> Reset time rolling average: {:.3f}s <<<<< \n".format(self.reset_time_rolling_average))
        return self.observation, self.info  # reward, done can't be included

    def close(self):
        # close powermeters
        for pd in self.pds:
            pd.__exit__()


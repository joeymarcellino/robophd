#import stable_baselines3 as sb
import numpy as np
import gym
from gym import spaces
import random
import time as time
import pylablib as pll


class EnvFibreonlymirrorsaveaverage(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, actxm1, actym1, actxm2, actym2, pd1, pd2, max_actioninsteps, min_power, powermultiplier, poweradder, minmirrorintervalsteps,
                 maxmirrorintervalsteps, minmirrorintervalinitial, maxmirrorintervalinitial, wait_time_pd, max_episode_steps=100):
        super(EnvFibreonlymirrorsaveaverage, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(7,), dtype=np.float64)
        # Actuator i of mirror j
        self.actxm1 = actxm1
        self.actxm2 = actxm2
        self.actym1 = actym1
        self.actym2 = actym2
        # photodetector
        self.pd1 = pd1
        self.pd2 = pd2
        # variables
        self.max_episode_steps = max_episode_steps
        self.max_power = max_power
        self.max_actioninsteps = max_actioninsteps
        self.minmirrorintervalsteps = minmirrorintervalsteps
        self.maxmirrorintervalsteps = maxmirrorintervalsteps
        self.minmirrorintervalinitial = minmirrorintervalinitial
        self.maxmirrorintervalinitial = maxmirrorintervalinitial
        self.wait_time_pd = wait_time_pd
        self.min_power = min_power
        self.powermultiplier = powermultiplier
        self.poweradder = poweradder

    def step(self, action):
        self.episode_steps += 1
        while self.pd1.get_power() < self.min_power:
            time.sleep(self.wait_time_pd)
        self.max_power = self.pd1.get_power() * self.powermultiplier + self.poweradder
        actxm1pos = self.actxm1.get_position()
        if action[0] + actxm1pos >= self.maxmirrorintervalsteps:
            action[0] = self.maxmirrorintervalsteps - actxm1pos
        if action[0] + actxm1pos <= self.minmirrorintervalsteps:
            action[0] = self.minmirrorintervalsteps - actxm1pos
        actxm2pos = self.actxm2.get_position()
        if action[1] + actxm2pos >= self.maxmirrorintervalsteps:
            action[1] = self.maxmirrorintervalsteps - actxm2pos
        if action[1] + actxm2pos <= self.minmirrorintervalsteps:
            action[1] = self.minmirrorintervalsteps - actxm2pos
        actym1pos = self.actym1.get_position()
        if action[2] + actym1pos >= self.maxmirrorintervalsteps:
            action[2] = self.maxmirrorintervalsteps - actym1pos
        if action[2] + actym1pos <= self.minmirrorintervalsteps:
            action[2] = self.minmirrorintervalsteps - actym1pos
        actym2pos = self.actym2.get_position()
        if action[3] + actym2pos >= self.maxmirrorintervalsteps:
            action[3] = self.maxmirrorintervalsteps - actym2pos
        if action[3] + actym2pos <= self.minmirrorintervalsteps:
            action[3] = self.minmirrorintervalsteps - actym2pos
        actioninsteps = np.around(action * self.max_actioninsteps)
        actionnormalized=actioninsteps/(self.max_actioninsteps)
        avgn = 0
        avgobs = self.observation[-1]
        # perform action
        self.actxm1.move_by(actioninsteps[0], scale=False)
        self.actxm2.move_by(actioninsteps[1], scale=False)
        self.actym1.move_by(actioninsteps[2], scale=False)
        self.actym2.move_by(actioninsteps[3], scale=False)
        # measure until move is done
        while self.actxm1.is_moving() == True:
            avgobs += self.pd2.get_power() / self.max_power
            avgn += 1
            time.sleep(self.wait_time_pd)
        while self.actxm2.is_moving() == True:
            avgobs += self.pd2.get_power() / self.max_power
            avgn += 1
            time.sleep(self.wait_time_pd)
        while self.actym1.is_moving() == True:
            avgobs += self.pd2.get_power() / self.max_power
            avgn += 1
            time.sleep(self.wait_time_pd)
        while self.actym2.is_moving() == True:
            avgobs += self.pd2.get_power() / self.max_power
            avgn += 1
            time.sleep(self.wait_time_pd)
        # make observation
        avgobs = avgobs / avgn
        obsprime = self.pd2.get_power()/ self.max_power
        self.observation = np.array([actionnormalized[0],actionnormalized[1],actionnormalized[2],actionnormalized[3], self.observation[-2], self.observation[-1], avgobs, obsprime])
        # calculate reward
        reward = avgobs - np.log(1 - avgobs) - 1  # what reward? #right now same as in interferobot paper

        # done:
        if self.max_episode_steps == self.episode_steps:
            self.done=True
            print(self.observation)
        done = self.done
        # info
        info = {}
        return self.observation, reward, done, info

    def reset(self):
        self.episode_steps = 0
        self.done=False
        # have to put mirrors to random state not too far of
        self.actxm1.move_to(random.randint(self.minmirrorintervalinitial, self.maxmirrorintervalinitial), scale=False)
        self.actxm2.move_to(random.randint(self.minmirrorintervalinitial, self.maxmirrorintervalinitial), scale=False)
        self.actym1.move_to(random.randint(self.minmirrorintervalinitial, self.maxmirrorintervalinitial), scale=False)
        self.actym2.move_to(random.randint(self.minmirrorintervalinitial, self.maxmirrorintervalinitial), scale=False)
        # observation:
        while self.pd1.get_power() < self.min_power:
            time.sleep(self.wait_time_pd)
        self.max_power = self.pd1.get_power() * self.powermultiplier + self.poweradder
        obs = self.pd2.get_power()/self.max_power
        self.observation = np.array([0, 0, 0, 0, obs, obs, obs, obs])
        return self.observation  # reward, done, info can't be included




'''

class EnvFibreonlymirrors(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, actxm1, actym1, actxm2, actym2, pd, max_actioninmm, mmtosteps, max_power, max_episode_steps=200):
        super(EnvFibreonlymirrors, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1,
                                            shape=(6,), dtype=np.float64)
        # Actuator i of mirror j
        self.actxm1 = actxm1
        self.actxm2 = actxm2
        self.actym1 = actym1
        self.actym2 = actym2
        # photodetector
        self.pd = pd
        # variables
        self.max_episode_steps = max_episode_steps
        self.max_power = max_power
        self.max_actioninmm=max_actioninmm
        self.mmtosteps=mmtosteps

    def step(self, action):
        self.episode_steps += 1
        actioninsteps = np.around(action * self.mmtosteps*self.max_actioninmm)
        actionnormalized=actioninsteps/(self.mmtosteps*self.max_actioninmm)
        # perform action
        self.actxm1.move_by(actioninsteps[0], scale=False)
        self.actxm2.move_by(actioninsteps[1], scale=False)
        self.actym1.move_by(actioninsteps[2], scale=False)
        self.actym2.move_by(actioninsteps[3], scale=False)
        # wait until move is done
        self.actxm1.wait_move()
        self.actxm2.wait_move()
        self.actym1.wait_move()
        self.actym2.wait_move()
        # make observation
        obsprime = self.pd.get_power()/ self.max_power
        self.observation = np.array([actionnormalized[0],actionnormalized[1],actionnormalized[2],actionnormalized[3], self.observation[-1], obsprime])
        # calculate reward
        reward = obsprime - np.log(1-obsprime) - 1  # what reward? #right now same as in interferobot paper

        # done:
        if self.max_episode_steps == self.episode_steps:
            self.done=True
            print(self.observation)
        done = self.done
        # info
        info = {}
        return self.observation, reward, done, info

    def reset(self):
        self.episode_steps = 0
        self.done=False
        # have to put mirrors to random state not too far of
        self.actxm1.move_by(random.randint(1, self.max_actioninmm * self.mmtosteps), scale=False)
        self.actxm2.move_by(random.randint(1, self.max_actioninmm * self.mmtosteps), scale=False)
        self.actym1.move_by(random.randint(1, self.max_actioninmm * self.mmtosteps), scale=False)
        self.actym2.move_by(random.randint(1, self.max_actioninmm * self.mmtosteps), scale=False)
        # observation:
        obs = self.pd.get_power()/self.max_power
        self.observation = np.array([0, 0, 0, 0, obs, obs])
        return self.observation  # reward, done, info can't be included
'''

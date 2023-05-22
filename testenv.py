from stable_baselines3.common.env_checker import check_env
import numpy as np
import pylablib as pll
import Environmentfibre
import Photodetector
import pyvisa
from pyvisa.highlevel import ResourceManager
from pylablib.devices import Thorlabs
import os

actxm1=Thorlabs.KinesisMotor("26004585")
actym1=Thorlabs.KinesisMotor("26004587")
actxm2=Thorlabs.KinesisMotor("26003852")
actym2=Thorlabs.KinesisMotor("26003794")

pd2 = Photodetector.Photodetector('USB0::0x1313::0x807B::1915386::0::INSTR')
pd1 =
max_actioninsteps=10**6
minmirrorintervalsteps = -10**6
maxmirrorintervalsteps = 5 * 10**6
minmirrorintervalinitial = -10**6
maxmirrorintervalinitial = 5 * 10**6
max_power=1.8*10**(-3)
wait_time_pd =

env=Environmentfibre.EnvFibreonlymirrorsaveaverage(actxm1, actym1, actxm2, actym2, pd1, pd2, max_actioninsteps, min_power, powermultiplier, poweradder, minmirrorintervalsteps,
                 maxmirrorintervalsteps, minmirrorintervalinitial, maxmirrorintervalinitial, wait_time_pd, max_episode_steps=100)


check_env(env)
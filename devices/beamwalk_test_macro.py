#%%

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
#
from powermeter_pmodad5 import PmodAd5
from liveplotter_heavy import LivePlotAgent
from steppermotor_ble import StepMo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Lab')))

from MovePowerUp import scuffed_beamwalking, simple_grad_ascent

#%%
# enable pd after reset using sudo chmod 777 /dev/ttyACM0
log_dir = "/home/robophd/Documents/github/robophd/devices/"
pds: PmodAd5 = PmodAd5(address = "/dev/ttyACM0")
actuators: StepMo = StepMo(log_dir = log_dir)
liveplotter: LivePlotAgent = LivePlotAgent()

#%%

'''
Beamwalking test.
steppers: list, which steppers to use
'''


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

pds.bin_no = 100

### data_func is a method that returns an array of arrays
liveplotter.new_liveplot(data_func=get_data, kill_func = None, **plot_args)
#%%
max_power = pds.get_measurement()[0][-1]
simple_grad_ascent(pds, actuators, move_increment=10)
power_history = scuffed_beamwalking(actuators,pds,goal_power=0.8,move_increment= 10,fringe_min=.5)
plt.plot(power_history)
plt.show()
actuators.close()

                



#%%

import numpy as np
import matplotlib.pyplot as plt
import time
#
from powermeter_pmodad5 import PmodAd5
from liveplotter_heavy import LivePlotAgent
from steppermotor_ble import StepMo

#%%
# enable pd after reset using sudo chmod 777 /dev/ttyACM0
log_dir = "/home/robophd/Documents/github/robophd/devices/"
pds: PmodAd5 = PmodAd5(address = "/dev/ttyACM0")
actuators: StepMo = StepMo(log_dir = log_dir)
liveplotter: LivePlotAgent = LivePlotAgent()

#%%

'''
Motor calibration test. Perform some number of random actions and revert, then record power drift in standard deviations with arbitrary units (assuming Gaussian mode shape of unknown width).
Note: start by manually aligning to max power
steppers: list, which steppers to use
max_actions: int, maximum number of random actions to perform (will do 1 to max_actions)
trials_per_run: int, how many trials to do per number of actions
max_step: int, maximum step size for random actions
returns: mean_drift_in_stds, std_drift_in_stds, drifts_all
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

steppers = [4]
max_actions = 10
trials_per_run = 10
max_step = 100
#%%
actuators.zero_positions()
print('Stepper positions re-zeroed to current positions')

mean_drift_in_stds = np.zeros(max_actions)
std_drift_in_stds = np.zeros(max_actions)
drifts_all = []
starts_all = []
ends_all = [] 

max_power = pds.get_measurement()[1][-1]

for actions in range(max_actions):
    drifts = np.zeros(trials_per_run)
    for trial in range(trials_per_run):
        start_power = pds.get_measurement()[1][-1]
        starts_all.append(start_power)
        if start_power >= max_power:
            start_pos_stds = 0
        else:
            start_pos_stds = np.sqrt(-np.log(start_power/max_power))
        actuators.random_action_and_revert(steppers=steppers,num_actions = actions+1,step_magnitude=max_step)
        time.sleep(.5)
        end_power = pds.get_measurement()[1][-1]
        ends_all.append(end_power)
        if end_power >= max_power:
            end_pos_stds = 0
        else: 
            end_pos_stds = np.sqrt(-np.log(end_power/max_power))
        drift = np.abs(start_pos_stds - end_pos_stds)
        drifts[trial] = drift
    drifts_all.append(drifts)
    mean_drift_in_stds[actions] = np.mean(drifts)
    std_drift_in_stds[actions] = np.std(drifts)


plt.errorbar([x+1 for x in range(len(mean_drift_in_stds))], mean_drift_in_stds, yerr=std_drift_in_stds)
plt.show()
#%%

                



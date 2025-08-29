#%%
import os 
import sys 
import numpy as np
devices_relative_path = "../"
file_abs_path = os.path.abspath(__file__)
devices_abs_path = os.path.join(os.path.dirname(file_abs_path), devices_relative_path)

if devices_abs_path not in sys.path:
    sys.path.insert(0, devices_abs_path)

from devices.powermeter_pmodad5 import PmodAd5 
from devices.liveplotter_heavy import LivePlotAgent

pds: PmodAd5 = PmodAd5(address = "/dev/ttyACM0")
plotter: LivePlotAgent = LivePlotAgent()

pds.bin_no = 100

def get_data():
    data = pds.get_measurement()
    #np.append(data,data[1]/data[0]*100) # add relative power in %
    print(data)
    return data

plot_args ={
            'refresh_interval': 0.01,
            'title': "Live Powermeter",
            'xlabel': "Time (0.1s per bin)",
            'ylabel': "Power (mW)",
            'no_plots': 3,
            'plot_labels': None,
        }

### data_func is a method that returns an array of arrays
plotter.new_liveplot(data_func=get_data, kill_func = None, **plot_args)

# %%


import visa
import pyvisa
from pyvisa.highlevel import ResourceManager
import numpy as np

rm = ResourceManager()

class Photodetector:

    def __init__(self, visaname):
        self.name = rm.open_resource(visaname)  # open powermeters

    def get_power(self):
        return np.float16(self.name.query('meas?'))  # return power measurement

    def clear(self):
        self.name.clear()  # needed at the start

    def close(self):
        self.name.close()  # close powermeters
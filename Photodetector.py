
import visa
import pyvisa
from pyvisa.highlevel import ResourceManager


rm = ResourceManager()

class Photodetector:

    def __init__(self, visaname):
        self.name = rm.open_resource(visaname)

    def get_power(self):
        return(float(self.name.query('meas?')))
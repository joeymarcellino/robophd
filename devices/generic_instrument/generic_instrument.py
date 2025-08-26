#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Angelo Gelmini
"""

import logging
from pyvisa import ResourceManager
import numpy as np
import threading
import time
# TODO : add check for successful connection

class GenericInstrument:

    def __init__(self, address, name, scaling = 1., special_init = None):
        self._name = name
        self._address = address
        self.scaling = scaling
        if self._name is None:
            self.connected = False
            logging.info(f"Couldn't connect to {self._name}.")
        else:
            self.connected = True


        self.rm = ResourceManager()
        # print(self.rm.list_resources())
        if special_init is None:
            self.client = self.rm.open_resource(self._address, read_termination='\n', write_termination='\n')
        else:
            self.client = self.rm.open_resource(self._address, **special_init)
            logging.info(f"Connected to {self._name}.")
            print(f"Connected to {self._name}!!!!!!\n")


        self.measuring = False
        self.measurement = np.zeros(100)
        self.updated = False
        ### when we want all the data over time
        self.accumulate = False

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return

#### this is a place holder method that is meant to be 
### replaced by child class read_data methods
    def read_data(self):
        return None
###############################


    def kill_measurement(self):
        self.measuring = False
        return self
    
    def get_measurement(self):
        return np.array([self.measurement])

### THIS METHOD IS FOR INSTRUMENTS THAT DO NOT STORE AN INTERNAL
### ARRAY IN SOME INTERNAL BUFFER, SO WE MAKE ONE OURSELVES
    def start_measurement(self, n, clock_s, dataframe = False):
        '''
        n is the number of points in output array
        clock_s is the time between each measurement in seconds
        dataframe is a boolean to indicate if read_data gives full frame or just one point
        '''
        if not self.measuring:
            self.measuring = True
            def data_thread():
                data = np.zeros(n)
                
                while self.measuring:
                    if not dataframe:
                        if self.accumulate:
                            data = np.insert(data, -1, values = self.read_data() * self.scaling)
                        else:
                            data = np.roll(data, -1)
                            data[-1] = self.read_data()

                        self.measurement = data
                        self.updated = True
            
                    else:
                        self.measurement = self.read_data(n)

                    time.sleep(clock_s)

                return self.measurement

            t = threading.Thread(target=data_thread)
            t.start()
        return 1 ### number of channels by default



    def general_close(self):
        if self.connected:
            logging.info(f"Disconnecting {self._name}")
            self.client.close()
            self.connected = False
            logging.info(f"{self._name} connection closed")
        else:
            logging.info(f"{self._name} was already disconnected")

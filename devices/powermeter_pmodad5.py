#!/usr/bin/env python3

import serial
import serial.tools.list_ports
import numpy as np
import time 
import threading


class PmodAd5():

    def __init__(self, address = 'COM11', name = 'PmodAD5', **kwargs):
        
        self.buffer_reading = True 
        self.buffer = []
        self.buffer_length = 1000

        self.name = name
        self.address = address
        self.baudrate = 115200
        self.client = serial.Serial(self.address, self.baudrate, timeout = 1)
        print('\nInitialised PmodAD5 Arduino dual-channel powermeter.')
        print('Please note this driver uses PySerial and NOT PyVISA.')
        print('Dataframe is in [channel1 voltage, channel2 voltage, ch2/ch1 ratio] format!!!\n')
        self.measuring = False
        self.bin_no = 10
        self.accumulate = False
        self.dataframe = []
        # self.client.readline() ### flush first line
        threading.Thread(target=self.__start_buffer__, daemon = True).start()


    def __enter__(self):
        return self
    
    def __exit__(self):
        self.buffer_reading = False 
        time.sleep(1)
        self.client.close()
        return

    ### This is to standardise with the generic instruments start_measurement method
    ### It is not very meaningful by itself
    def start_measurement(self, bins, clock):
        self.bin_no = bins 
        self.measuring = True 
        return 2 ### number of channels 
    ### This is to standardise with the generic instruments kill_measurement method
    ### It is not very meaningful by itself
    def kill_measurement(self):
        return 

    ### We want to get time-synced dataframes like 
    # [['ch1', value], ['ch2', value], ['ratio', value]]
    def __start_buffer__(self):
        while self.buffer_reading:
            incoming = self.client.readlines(50)
            dataframe = [x.decode('utf-8')[:-2].split(':') for x in incoming]
            self.dataframe = dataframe
            if 'CH1' in dataframe[0][0]:
                append_sequence = [0, 1, 2]
            elif 'CH2' in dataframe[0][0]:
                append_sequence = [2, 0, 1]
            else:
                append_sequence = [1, 2, 0]

            lengths = [len(x) for x in dataframe] ### this might be inefficient 
            if lengths == [2, 2, 2]:
                try:
                    datarow = []
                    for index in append_sequence:
                        datarow.append(float(dataframe[index][1]))
                    self.buffer.append(datarow)## second element is the value, first is the channel name
                    if len(self.buffer) > self.buffer_length:
                        self.buffer = self.buffer[-self.buffer_length:]
                    if self.accumulate:
                        self.bin_no += 1
                except ValueError:
                    pass 
        return 


    ### we allow a dynamic buffer size to avoid another for loop in the datalogger script
    ### using this powermeter driver
    def read_dataframe(self, n):
        if n <= self.buffer_length:
            return np.array(self.buffer[-n:])
        
        else: 
            self.buffer_length = n 
            return np.array(self.buffer)
    
    ### returns a dataframe of format [[ch1], [ch2]]
    ### This is to standardise with the generic instruments get_measurement method
    '''
    def get_measurement(self):
        subframe = self.read_dataframe(self.bin_no)[:,:2]
        return subframe.T
    '''
    def get_measurement(self):
        arr = self.read_dataframe(self.bin_no)
        if arr.ndim < 2 or arr.shape[0] == 0:
            return np.zeros((2, 1))  # or whatever dummy shape you want
        subframe = arr[:, :2]
        return subframe.T

    def get_ratio(self):
        return self.read_dataframe(self.bin_no)[:,2]


if __name__ == '__main__':
    with PmodAd5('COM11', name = 'PmodAD5') as pa:
        print('\n\n################# ENTERING PmodAd5 AS  >>>>> pa <<<<<< ####################\n\n')
        import code; code.interact(local=locals())
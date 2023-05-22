# -*- coding: utf-8 -*-
from pylablib.devices import Thorlabs
import pylablib as pll
import serial
import visa
import pyvisa
from pyvisa.highlevel import ResourceManager
print(Thorlabs.list_kinesis_devices())
stage = Thorlabs.KinesisMotor("26003794")
print(stage.get_position())
stage.move_to(1000)
#stage.move_by(10000000)
stage.wait_move()
#stage.home()
#stage.wait_for_home()
stage.close()

print(pll.list_backend_resources("visa")) #maybe try pyserial
rm = ResourceManager()
print(rm.list_resources())
my_instrument = rm.open_resource('USB0::0x1313::0x807B::1915386::0::INSTR')
print(my_instrument.query('*IDN?'))
print(my_instrument.query('meas?'))



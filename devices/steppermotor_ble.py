import asyncio
from bleak import BleakClient, BleakScanner
import sys
import os
import threading
import time
import numpy as np
# Import the base BLE client class
# Ensure the path is correct based on your project structure
from arduino_dependencies.ble_client import BLE_Client
# These UUIDs must match the ones in your Arduino sketch
DEVICE_NAME = "StepperMotorBoard1"
COMMAND_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
STATUS_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

NUM_STEPPERS = 4
STEPS_PER_REVOLUTION = 1800 * 2
TIME_PER_STEP_S = 0.001 * 2 # we times 2 as an upper bound (normal)
#TIME_PER_STEP_S = .01 # too much for normal operation, testing charge drainage

class StepMo(BLE_Client):
    """
    This class pairs with the roboPhD project using an arduino uno r4 wifi board 
    and a quad-driver stepper motor board to control up to 4 tiny stepper motors. 

    Please look at the arduino sketch in the arduino_dependencies folder for more details about the pinout on the arduino board.

    self.current_position is a LIST of indexed motor positions
    self.motor_params is a DICT of motor parameters indexed by stepper number
    """
    def __init__(self, device_name=DEVICE_NAME, command_uuid=COMMAND_UUID, timeout=10, log_dir = "/"):

        super().__init__(device_name, command_uuid, timeout)
        self.log_dir = log_dir
        self.current_position = self._load_position()
        self._save_position()
        os.makedirs(self.log_dir, exist_ok=True)

        self.handshake()
        if self.connected:
            print("Connected to Stepper Motor Board")
            self.motor_params = {
                i: {'is_moving': False, 'last_direction': 0, 'steps': 0, 'backlash_steps': 0, 'pos': self.current_position[i - 1]} for i in range(1, NUM_STEPPERS + 1)
            }
            # self.say_hello()
        else:
            print("Failed to connect to Stepper Motor Board")
            sys.exit(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._save_position()
        self.disconnect()
        return 

    def close(self):
        self.__exit__(None, None, None)
        return 

    def _save_position(self):
        # save position to file
        with open(os.path.join(self.log_dir, f"stepper_position.txt"), "w") as f:
            f.write(",".join(map(str, self.current_position)))

    def _load_position(self):
        # load position from file
        try:
            with open(os.path.join(self.log_dir, f"stepper_position.txt"), "r") as f:
                return np.array([int(x) for x in f.read().strip().split(",")])
        except FileNotFoundError:
            return np.array([0,0,0,0])

    def set_position(self, stepper_number, position):
        # set stepper position and log it
        self.current_position[stepper_number - 1] = position
        self._save_position()

    def move_stepper(self, stepper_num, direction, steps):
        """Convenience method to move a specific stepper motor.
        we standardise the command format as 'stepper{num}_{direction}_{steps}'
        
        stepper_num: int, the stepper motor number (1 to NUM_STEPPERS)
        direction: int, 0 is backward, 1 is forward
        steps: int, number of steps to move
        
        we also break up the number of steps to allow interrupts and avoid sending 
        the arduino too big of a movement command at once.
        """
        if stepper_num not in range(1, NUM_STEPPERS + 1):
            raise ValueError("Stepper number must be between 1 and {}, but got {}".format(NUM_STEPPERS, stepper_num))
        if direction not in [0, 1]:
            raise ValueError("Direction must be 0 (backward) or 1 (forward), but got {}".format(direction))
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Steps must be a positive integer, but got {}".format(steps))

        step_chunks = 200
        time_chunks = step_chunks * TIME_PER_STEP_S
        self.motor_params[stepper_num]['is_moving'] = True

        if steps < step_chunks:
            self.send_command(f"stepper{stepper_num}_{direction}_{steps}")
            print(f"Moving stepper {stepper_num} {'forward' if direction == 1 else 'backward'} by {steps} steps.")

            action_sign = 1 if direction == 1 else -1
            self.current_position[stepper_num - 1] += steps * action_sign
            self.motor_params[stepper_num].update({'last_direction': direction, 'steps': steps, 'pos': self.current_position[stepper_num - 1]})
            self._save_position()            

            time.sleep(steps * TIME_PER_STEP_S)
            self.motor_params[stepper_num]['is_moving'] = False
        else:
            # Break the steps into chunks to avoid overwhelming the Arduino
            for i in range(0, steps, step_chunks):
                chunk = min(step_chunks, steps - i)
                self.send_command(f"stepper{stepper_num}_{direction}_{chunk}")
                print(f"Moving stepper {stepper_num} {'forward' if direction == 1 else 'backward'} by {chunk} steps.")

                action_sign = 1 if direction == 1 else -1
                self.current_position[stepper_num - 1] += chunk * action_sign
                self.motor_params[stepper_num].update({'last_direction': direction, 'steps': steps, 'pos': self.current_position[stepper_num - 1]})
                self._save_position()   

                time.sleep(time_chunks)
            self.motor_params[stepper_num]['is_moving'] = False

        return

    def say_hello(self):
        """A simple method to test the connection."""

        for i in range(NUM_STEPPERS):
            for q in range(4):
                self.move_stepper(i+1, 1, 100)
                time.sleep(0.2)
                self.move_stepper(i+1, 0, 100)
                time.sleep(0.2)
        print("Hello from StepMo!")
        return 

    def test_motor(self, stepper_num, steps=100, cycles=10):
        """Test a specific stepper motor by moving it a certain number of steps back and forth."""
        if stepper_num not in range(1, NUM_STEPPERS + 1):
            raise ValueError("Stepper number must be between 1 and {}".format(NUM_STEPPERS))
        
        for i in range(cycles):
            time.sleep(0.5)
            self.move_stepper(stepper_num, 1, steps)
            time.sleep(0.5)
            self.move_stepper(stepper_num, 0, steps)

        return 


if __name__ == '__main__':
    with StepMo(log_dir = "/home/robophd/Documents/github/robophd/devices/") as sm:
        print('\n\n################# ENTERING StepMo AS  >>>>> sm <<<<<< ####################\n\n')
        import code; code.interact(local=locals())

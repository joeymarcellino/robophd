import asyncio
from token import NUMBER
from bleak import BleakClient, BleakScanner
import sys
import os
import threading
import time
import numpy as np
import random 
# Import the base BLE client class
# Ensure the path is correct based on your project structure
from arduino_dependencies.ble_client import BLE_Client
# These UUIDs must match the ones in your Arduino sketch
DEVICE_NAME = "StepperMotorBoard1"
COMMAND_UUID = "19B10001-E8F2-537E-4F6C-D104768A1214"
STATUS_UUID = "19B10002-E8F2-537E-4F6C-D104768A1214"

NUM_STEPPERS = 4
STEPS_PER_REVOLUTION = 4096 
TIME_PER_STEP_S = 0.001 * 1.7 # we times 2 as an upper bound (normal)
MOVEMENT_LIMIT = 10000
MINIMUM_STEPS = 5
#TIME_PER_STEP_S = .01 # too much for normal operation, testing charge drainage



### defined as [1to1, 0to0]
FRONTLASH = {1: [0, 0], 
            2: [0, 0], 
            3: [0, 0], 
            4: [0, 0]}

### defined as [1to0, 0to1]
BACKLASH = {1: [0, 0],
            2: [0, 0], 
            3: [0, 0], 
            4: [0, 0]}

BACKLASH = {1: [0, 0],
            2: [0, 0], 
            3: [0, 0], 
            4: [0, 0]}

LASH = {1: [3,3], 
            2: [0,0], 
            3: [0,0], 
            4: [0,0]}

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
                i: {'is_moving': False, 'last_direction': 1, 'steps': 0, 'backlash': BACKLASH[i], 'frontlash': FRONTLASH[i], 'lash': LASH[i], 'pos': self.current_position[i - 1]} for i in range(1, NUM_STEPPERS + 1)
            }
            # self.say_hello()
            self.move_stepper = self.mvstp  # set default movement method

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

    def zero_positions(self, stepper_numbers = [1, 2, 3, 4]):
        for stepper_num in stepper_numbers:
            if stepper_num in range(1, NUM_STEPPERS + 1):
                self.current_position[stepper_num - 1] = 0
                self.motor_params[stepper_num]['pos'] = 0
            else:
                raise ValueError("Stepper number must be between 1 and {}, but got {}".format(NUM_STEPPERS, stepper_num))
        self._save_position()
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

    def home_steppers(self, stepper_numbers = [1, 2, 3, 4]):
        for stepper_num in stepper_numbers:
            if stepper_num in range(1, NUM_STEPPERS + 1):
                current_pos = self.current_position[stepper_num - 1]
                direction = 1 if current_pos < 0 else 0
                self.move_stepper(stepper_num, direction, int(np.abs(current_pos)))
            else:
                raise ValueError("Stepper number must be between 1 and {}, but got {}".format(NUM_STEPPERS, stepper_num))
        return

    def set_position(self, stepper_number, position):
        # set stepper position and log it
        self.current_position[stepper_number - 1] = position
        self._save_position()


    def mvstp(self, stepper_num, direction, steps, verbose = False):
        """Move stepper with a different backlash compensation method.
        This method oversteps by a fixed amount and then corrects back to the desired position.
        
        """
        overstep = 200
        if steps < MINIMUM_STEPS:
            steps = 0

        if steps == 0: 
            self.motor_params[stepper_num]['is_moving'] = False
            self._save_position()            
            return 

        if steps < MOVEMENT_LIMIT:
            

            last_direction = self.motor_params[stepper_num]['last_direction']
            if last_direction != direction and steps > 0:

                self.send_command(f"stepper{stepper_num}_{direction}_{steps+overstep}")
                if verbose:
                    print(f"Moving stepper {stepper_num} {'forward' if direction == 1 else 'backward'} by {steps} steps.")
                ## go backwards the overstep amount to compensate for overstepping
                lash = self.motor_params[stepper_num]["lash"][1] if direction == 1 else self.motor_params[stepper_num]["lash"][0] 
                self.send_command(f"stepper{stepper_num}_{0 if last_direction == 0 else 1}_{overstep + lash}")

            elif last_direction == direction and steps > 0:
                self.send_command(f"stepper{stepper_num}_{direction}_{steps}")
                if verbose:
                    print(f"Moving stepper {stepper_num} {'forward' if direction == 1 else 'backward'} by {steps} steps.")
                lash = 0

            action_sign = 1 if direction == 1 else -1
            self.current_position[stepper_num - 1] += steps * action_sign
            self.motor_params[stepper_num].update({'last_direction': last_direction, 'steps': steps, 'pos': self.current_position[stepper_num - 1]})
            self._save_position()            

            time.sleep((steps+overstep+lash) * TIME_PER_STEP_S)
            self.motor_params[stepper_num]['is_moving'] = False
        
        else:
            print(f"MOVEMENT TOO LARGE {steps}, MAKE IT LESS THAN {MOVEMENT_LIMIT}")

        return


    def move_stepper(self, stepper_num, direction, steps, verbose = False):
        """
        Move stepper with set backlash compensation (don't intentionally overshoot).
        Convenience method to move a specific stepper motor.
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

        self.motor_params[stepper_num]['is_moving'] = True
        last_direction = self.motor_params[stepper_num]['last_direction']

        if steps < MINIMUM_STEPS:
            steps = 0

        if last_direction != direction and steps > 0:
            if last_direction - direction > 0:
                lash = self.motor_params[stepper_num]['backlash'][0]
            else: 
                lash = self.motor_params[stepper_num]['backlash'][1]

        elif last_direction == direction and steps > 0:
            if direction == 1:
                lash = self.motor_params[stepper_num]['frontlash'][0]
            else:
                lash = self.motor_params[stepper_num]['frontlash'][1]

        if steps == 0: 
            self.motor_params[stepper_num]['is_moving'] = False
            self._save_position()            
            return 

        if steps < MOVEMENT_LIMIT:
            self.send_command(f"stepper{stepper_num}_{direction}_{steps+lash}")
            if verbose:
                print(f"Moving stepper {stepper_num} {'forward' if direction == 1 else 'backward'} by {steps} steps.")

            action_sign = 1 if direction == 1 else -1
            self.current_position[stepper_num - 1] += steps * action_sign
            self.motor_params[stepper_num].update({'last_direction': direction, 'steps': steps, 'pos': self.current_position[stepper_num - 1]})
            self._save_position()            

            time.sleep((steps+lash) * TIME_PER_STEP_S)
            self.motor_params[stepper_num]['is_moving'] = False
        
        else:
            print(f"MOVEMENT TOO LARGE {steps}, MAKE IT LESS THAN {MOVEMENT_LIMIT}")

        return
    
    def backlash_test_3up3down(self, stepper_num, step_magnitude):

        self.mvstp(stepper_num, 1, step_magnitude)
        time.sleep(0.1)
        self.mvstp(stepper_num, 0, step_magnitude)
        time.sleep(0.1)
        self.mvstp(stepper_num, 0, step_magnitude)
        time.sleep(0.1)

        self.mvstp(stepper_num, 1, step_magnitude)
        time.sleep(0.1)

        self.mvstp  (stepper_num, 1, step_magnitude)
        time.sleep(0.1)

        self.mvstp(stepper_num, 0, step_magnitude)

        return

    def frontlash_test(self, stepper_num, step_magnitude, num_actions = 3):



        for i in range (num_actions):
            self.move_stepper(stepper_num, 0, step_magnitude)
        for i in range (num_actions):
            self.move_stepper(stepper_num, 1, step_magnitude)
        return 
    
    def random_action_and_revert(self, steppers = [1,2,3,4],num_actions = 5, step_magnitude = 100):

        print(f"Moving stepper motors randomly...in {num_actions} actions of max {step_magnitude} steps each")
        net_displacement = np.zeros(len(steppers))
        for i in range(len(steppers)):
            displacement = 0
            for q in range(num_actions):
                direction = random.choice([0, 1])
                sign = 1 if direction == 1 else -1
                steps = random.randint(0, step_magnitude)
                if steps < 5: 
                    steps = 0
                print(f"Moving stepper {steppers[i]} {'forward' if direction == 1 else 'backward'} by {steps} steps.")
                self.move_stepper(steppers[i], direction, steps)
                displacement += steps * sign

            net_displacement[i] = displacement

        time.sleep(0.5)
        print("Reverting all positions!!")
        for i in range(len(steppers)):
            if net_displacement[i] != 0:
                print(f"Reverting stepper {steppers[i]} {'forward' if net_displacement[i] < 0 else 'backward'} by {int(np.abs(net_displacement[i]))} steps.")
                direction = 1 if net_displacement[i] < 0 else 0
                self.move_stepper(steppers[i], direction, int(np.abs(net_displacement[i])))

        print("Done.")

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

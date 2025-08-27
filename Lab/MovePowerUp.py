import numpy as np
import random
import warnings
import time

def move_power_up(pds, actuators, max_power, start_dir, neutral_positions, min_power_after_reset,
                  max_power_to_neutral, number_of_random_steps_low_power, max_random_reset_step_low_power,
                  min_power_stop_random_steps, reset_step_size, min_ref_power, wait_time_pd,
                  powermultiplier, poweradder):
    # with each act keep on moving in same direction while this moving helps (power_dif>0)
    # when this changes, change direction
    number_move_power_up_movements = 0
    # test if reference powermeter has power, otherwise wait until it has
    how_long_ref_power_under_min_ref_power = 0
    while pds.get_measurement()[0][-1] < min_ref_power:
        time.sleep(wait_time_pd)
        how_long_ref_power_under_min_ref_power += 1
        if how_long_ref_power_under_min_ref_power > 10:
            warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
    max_power = (pds.get_measurement()[0][-1]) * powermultiplier + poweradder  # get max possible power
    power_new = pds.get_measurement()[1][-1] / max_power
    current_dir = start_dir
    while power_new < min_power_after_reset:
        # test if reference powermeter has power, otherwise wait until it has
        how_long_ref_power_under_min_ref_power = 0
        while pds.get_measurement()[0][-1] < min_ref_power:
            time.sleep(wait_time_pd)
            how_long_ref_power_under_min_ref_power += 1
            if how_long_ref_power_under_min_ref_power > 10:
                warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
        max_power = (pds.get_measurement()[0][-1]) * powermultiplier + poweradder  # get max possible power
        # move to neutral positions and do some random steps for very small power
        if power_new < max_power_to_neutral:
            number_moves_to_neutral, power_new = to_neutral_positions_random_steps(pds, actuators, max_power, neutral_positions,
                                              max_power_to_neutral, number_of_random_steps_low_power,
                                              max_random_reset_step_low_power,
                                              min_power_stop_random_steps)
            number_move_power_up_movements += number_moves_to_neutral
        else:
            # move in current direction while power gets better (each actuator individually)
            shuffled_list = random.sample(range(4), k=4)
            for i in shuffled_list:
                print(f'Current actuator: {i}')
                power_old = power_new
                rand = np.random.uniform(low=0.5, high=2.0)  # add some randomness to step size
                movement = int(current_dir[i]*reset_step_size*rand)
                action_sign = [1 if np.sign(movement) >= 0 else 0]
                actuators.move_stepper(i+1, action_sign, np.abs(movement))
                number_move_power_up_movements += 1
                power_new = pds.get_measurement()[1][-1] / max_power
                print(f'Power after moving: {power_new}')
                p_diff = power_new - power_old
                if p_diff < -0.002:  # if power gets worse, reverse last action and change direction.
                    power_old = power_new
                    movement = -(int(current_dir[i] * reset_step_size * rand))
                    action_sign = [1 if np.sign(movement) >= 0 else 0]
                    actuators.move_stepper(i+1, action_sign, np.abs(movement))
                    number_move_power_up_movements += 1
                    # print('Reverse last action, then next actuator') # Commented 20/04
                    power_new = pds.get_measurement()[1][-1] / max_power
                    # p_diff = power_new - power_old
                    # print(f'Power after reversing: {power_new}') # Commented 20/04
                    # current_dir[i] = -current_dir[i]
                while p_diff >= 0 and power_new < min_power_after_reset:
                    # 2nd condition added so that loop stops when threshold power is reached
                    power_old = power_new
                    # use >= so that it keeps on moving in this direction if one movement was so small
                    # that it had no impact at all.
                    rand = np.random.uniform(low=0.5, high=2.0)  # add some randomness to step size
                    movement = int(current_dir[i] * reset_step_size * rand)
                    action_sign = [1 if np.sign(movement) >= 0 else 0]
                    actuators.move_stepper(i+1, action_sign, np.abs(movement))
                    number_move_power_up_movements += 1
                    power_new = pds.get_measurement()[1][-1] / max_power
                    p_diff = power_new - power_old
                    if p_diff < -0.002:  # if power gets worse, reverse last action and change direction
                        movement = -(int(current_dir[i] * reset_step_size * rand))
                        action_sign = [1 if np.sign(movement) >= 0 else 0]
                        actuators.move_stepper(i+1, action_sign, np.abs(movement))
                        number_move_power_up_movements += 1
                        power_new = pds.get_measurement()[1][-1] / max_power
                if power_new >= min_power_after_reset:
                    print('Threshold reached. Break in for loop.')
                    break
            current_dir = (-1)*current_dir  # change direction
    return number_move_power_up_movements

def to_neutral_positions_random_steps(pds, actuators, max_power, neutral_positions,
                  max_power_to_neutral, number_of_random_steps_low_power, max_random_reset_step_low_power,
                  min_power_stop_random_steps):
    # move to neutral position
    number_movements = 0
    print('Reverse to neutral positions.')
    for i in range(4):
        current_pos = actuators.motor_params[i+1]['pos']
        action_sign = np.sign(neutral_positions[i] - current_pos)
        actuators.move_stepper(i+1, action_sign, np.abs(neutral_positions[i] - current_pos))
        number_movements += 1
    power_new = pds.get_measurement()[1][-1] / max_power  # ADD!
    # do random steps if power (still) small
    if power_new < max_power_to_neutral:
        print("random steps")
        for j in range(number_of_random_steps_low_power):
            for i in range(4):
                add_random_steps = random.randint(- max_random_reset_step_low_power,
                                                  max_random_reset_step_low_power)
                action_sign = [1 if np.sign(add_random_steps) >= 0 else 0]
                actuators.move_stepper(i+1, action_sign, np.abs(add_random_steps))
                number_movements += 1
            power_new = pds.get_measurement()[1][-1] / max_power
            print(f'round {j}, current power: {power_new}.')
            if power_new > min_power_stop_random_steps:
                print(f'p = {power_new} > min_power_stop_random_steps.')
                break
    return number_movements, power_new


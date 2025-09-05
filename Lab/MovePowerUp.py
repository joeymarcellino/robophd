import numpy as np
import random
import warnings
import time

def grad_ascent(pds, actuators, max_power, actioninsteps, neutral_positions,      min_power_after_reset,
                  max_power_to_neutral, number_of_random_actions_low_power, neutral_flailing_step_magnitude,
                  min_power_stop_random_actions_neutral_failure, grad_ascent_step_size, min_ref_power, wait_time_pd,
                  ref_pd_slope, ref_pd_intercept):
    # with each act keep on moving in same direction while this moving helps (power_dif>0)
    # when this changes, change direction
    sgn_last_action = np.sign(actioninsteps) ## actioninsteps being the last action taken in motor steps
    start_dir = (-1) * sgn_last_action
    if np.array_equal(start_dir, np.array([0, 0, 0, 0])):
        start_dir = np.array([(2*random.randint(0, 1) - 1) for _ in range(4)])

    number_grad_ascent_movements = 0
    # test if reference powermeter has power, otherwise wait until it has
    how_long_ref_power_under_min_ref_power = 0
    while pds.get_measurement()[0][-1] < min_ref_power:
        time.sleep(wait_time_pd)
        how_long_ref_power_under_min_ref_power += 1
        if how_long_ref_power_under_min_ref_power > 10:
            warnings.warn(f"no reference power for {how_long_ref_power_under_min_ref_power} steps")
    max_power = (pds.get_measurement()[0][-1]) * ref_pd_slope + ref_pd_intercept  # get max possible power
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
        max_power = (pds.get_measurement()[0][-1]) * ref_pd_slope + ref_pd_intercept  # get max possible power
        # move to neutral positions and do some random steps for very small power
        if power_new < max_power_to_neutral:
            print('power very low, going to neutral positions and doing random steps')
            number_moves_to_neutral, power_new = to_neutral_positions_random_steps(pds, actuators, max_power, neutral_positions,
                                              max_power_to_neutral, number_of_random_actions_low_power,
                                              neutral_flailing_step_magnitude,
                                              min_power_stop_random_actions_neutral_failure)
            number_grad_ascent_movements += number_moves_to_neutral
        else:
            print('trying gradient ascent')
            # move in current direction while power gets better (each actuator individually)
            shuffled_list = random.sample(range(4), k=4)
            for i in shuffled_list:
                #print(f'Current actuator: {i}')
                power_old = power_new
                rand = np.random.uniform(low=0.5, high=2.0)  # add some randomness to step size
                movement = int(current_dir[i]*grad_ascent_step_size*rand)
                direction = 1 if np.sign(movement) >= 0 else 0
                actuators.move_stepper(i+1, direction, int(np.abs(movement)))
                number_grad_ascent_movements += 1
                power_new = pds.get_measurement()[1][-1] / max_power
                # print(f'Power after moving: {power_new}')
                p_diff = power_new - power_old
                if p_diff < -0.002:  # if power gets worse, reverse last action and change direction.
                    power_old = power_new
                    movement = -(int(current_dir[i] * grad_ascent_step_size * rand))
                    direction = 1 if np.sign(movement) >= 0 else 0
                    actuators.move_stepper(i+1, direction, int(np.abs(movement)))
                    number_grad_ascent_movements += 1
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
                    movement = int(current_dir[i] * grad_ascent_step_size * rand)
                    direction = 1 if np.sign(movement) >= 0 else 0
                    actuators.move_stepper(i+1, direction, int(np.abs(movement)))
                    number_grad_ascent_movements += 1
                    power_new = pds.get_measurement()[1][-1] / max_power
                    p_diff = power_new - power_old
                    if p_diff < -0.002:  # if power gets worse, reverse last action and change direction
                        movement = -(int(current_dir[i] * grad_ascent_step_size * rand))
                        direction = 1 if np.sign(movement) >= 0 else 0
                        actuators.move_stepper(i+1, direction, int(np.abs(movement)))
                        number_grad_ascent_movements += 1
                        power_new = pds.get_measurement()[1][-1] / max_power
                if power_new >= min_power_after_reset:
                    print('Threshold reached. Break in for loop.')
                    break
            current_dir = (-1)*current_dir  # change direction
    return number_grad_ascent_movements, power_new

def to_neutral_positions_random_steps(pds, actuators, max_power, neutral_positions,
                  max_power_to_neutral, number_of_random_actions_low_power, neutral_flailing_step_magnitude,
                  min_power_stop_random_actions_neutral_failure):
    # move to neutral position
    number_movements = 0
    print('Reversing to neutral positions now.')
    for i in range(4):
        current_pos = actuators.current_position[i]
        direction = 0 if neutral_positions[i] - current_pos < 0 else 1
        actuators.move_stepper(i+1, direction, int(np.abs(neutral_positions[i] - current_pos)))
        number_movements += 1
    power_new = pds.get_measurement()[1][-1] / max_power  # ADD!
    print(f"done. Power ratio is {power_new:.4f}")
    # if power small, try moving each motor a fair amount in both directions, checking constantly for power
    if power_new < max_power_to_neutral:
        for i in range(4):
            for direction in [0,1]:
                for j in range(100):
                    actuators.move_stepper(i+1, direction, 10)
                    power_new = pds.get_measurement()[1][-1] / max_power
                    if power_new > min_power_stop_random_actions_neutral_failure:
                        print(f'p = {power_new} > min_power_stop_random_actions_neutral_failure.')
                        return number_movements, power_new
                actuators.move_stepper(i+1,-1*direction + 1, 1000)
    # do random steps if power (still) small, checking for power as you go
    if power_new < max_power_to_neutral:
        print("flailing neutral failure!!!")
        for j in range(number_of_random_actions_low_power):
            print(f'round {j}, current power: {power_new}.')
            for i in range(4):
                add_random_steps = random.randint(- neutral_flailing_step_magnitude,
                                                  neutral_flailing_step_magnitude)
                direction = 1 if np.sign(add_random_steps) >= 0 else 0
                for k in range(add_random_steps // 10):
                    actuators.move_stepper(i+1, direction, 10)
                    power_new = pds.get_measurement()[1][-1] / max_power
                    if power_new > min_power_stop_random_actions_neutral_failure:
                        print(f'p = {power_new} > min_power_stop_random_actions_neutral_failure.')
                        return number_movements, power_new
                number_movements += 1
    return number_movements, power_new


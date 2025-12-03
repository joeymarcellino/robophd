import numpy as np
import random
import warnings
import time

def grad_ascent(pds, actuators, max_power, actioninsteps, neutral_positions,      min_power_after_reset,
                  max_power_to_neutral, number_of_random_actions_low_power, neutral_flailing_step_magnitude,
                  min_power_stop_random_actions_neutral_failure, grad_ascent_step_size, min_ref_power, wait_time_pd,
                  time_wait_pd,
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
                                              min_power_stop_random_actions_neutral_failure, time_wait_pd)
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
                time.sleep(time_wait_pd)
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
                    time.sleep(time_wait_pd)
                    power_new = pds.get_measurement()[1][-1] / max_power
                    p_diff = power_new - power_old
                    if p_diff < -0.002:  # if power gets worse, reverse last action and change direction
                        movement = -(int(current_dir[i] * grad_ascent_step_size * rand))
                        direction = 1 if np.sign(movement) >= 0 else 0
                        actuators.move_stepper(i+1, direction, int(np.abs(movement)))
                        number_grad_ascent_movements += 1
                        time.sleep(time_wait_pd)
                        power_new = pds.get_measurement()[1][-1] / max_power
                if power_new >= min_power_after_reset:
                    print('Threshold reached. Break in for loop.')
                    break
            current_dir = (-1)*current_dir  # change direction
    return number_grad_ascent_movements, power_new

def to_neutral_positions_random_steps(pds, actuators, max_power, neutral_positions,
                  max_power_to_neutral, number_of_random_actions_low_power, neutral_flailing_step_magnitude,
                  min_power_stop_random_actions_neutral_failure, time_wait_pd):
    # move to neutral position
    number_movements = 0
    print('Reversing to neutral positions now.')
    for i in range(4):
        current_pos = actuators.current_position[i]
        direction = 0 if neutral_positions[i] - current_pos < 0 else 1
        actuators.move_stepper(i+1, direction, int(np.abs(neutral_positions[i] - current_pos)))
        number_movements += 1
    time.sleep(time_wait_pd)
    power_new = pds.get_measurement()[1][-1] / max_power  # ADD!
    print(f"done. Power ratio is {power_new:.4f}")
    # if power small, try moving each motor a fair amount in both directions, checking constantly for power
    if power_new < max_power_to_neutral:
        for i in range(4):
            for direction in [0,1]:
                for j in range(10):
                    actuators.move_stepper(i+1, direction, 50)
                    time.sleep(time_wait_pd)
                    power_new = pds.get_measurement()[1][-1] / max_power
                    if power_new > min_power_stop_random_actions_neutral_failure:
                        print(f'p = {power_new} > min_power_stop_random_actions_neutral_failure.')
                        return number_movements, power_new
                actuators.move_stepper(i+1,-1*direction + 1, 500)
    # do random steps if power (still) small, checking for power as you go
    if power_new < max_power_to_neutral:
        print("flailing neutral failure!!!")
        for j in range(number_of_random_actions_low_power):
            print(f'round {j}, current power: {power_new}.')
            for i in range(4):
                add_random_steps = random.randint(- neutral_flailing_step_magnitude,
                                                  neutral_flailing_step_magnitude)
                direction = 1 if np.sign(add_random_steps) >= 0 else 0
                for k in range(add_random_steps // 50):
                    actuators.move_stepper(i+1, direction, 50)
                    time.sleep(time_wait_pd)
                    power_new = pds.get_measurement()[1][-1] / max_power
                    if power_new > min_power_stop_random_actions_neutral_failure:
                        print(f'p = {power_new} > min_power_stop_random_actions_neutral_failure.')
                        return number_movements, power_new
                number_movements += 1
    return number_movements, power_new

def to_max_quantized(pds,actuators,i,direction,percent_change=.05,move_increment=10,power_history=[]):
    p0 = pds.get_measurement()[1][-1]
    p_ref = pds.get_measurement()[0][-1]
    if p0/p_ref < 0.01:
        print('Power too low, aborting')
        return
    p1 = p0
    moving = True
    while moving:
        while np.abs(p1 - p0)/p0 < percent_change: # move until the power changes by at least percent_change
            actuators.move_stepper(i,direction,move_increment)
            time.sleep(0.5)
            p1 = pds.get_measurement()[1][-1]
            power_history.append(p1)
        if p1 < p0:
            moving = False
        else: 
            p0 = p1
    return p0, p1, power_history

def simple_grad_ascent(pds, actuators, move_increment=10):
    print('Starting simple gradient ascent...')
    for i in [1,2,3,4]:
        direction = 0
        start_power = pds.get_measurement()[1][-1]
        p_ref = pds.get_measurement()[0][-1]
        if start_power/p_ref < 0.01:
            print('Power too low, aborting')
            return

        p0 = start_power
        p1 = p0

        while np.abs(p1 - p0)/p0 < .05: # move one direction until the power changes by at least %
            actuators.move_stepper(i,direction,move_increment)
            time.sleep(0.5)
            p1 = pds.get_measurement()[1][-1]
        
        if p1 < p0: # if it went down, reverse direction
            direction = 1
            print('Power went down, reversing direction.')
            p0 = p1

        p0, p1, _ = to_max_quantized(pds,actuators,i,direction,percent_change=0.02,move_increment=move_increment)

    return

# implement crude beamwalking procedure to retrieve power for auto-aligner training

def walk_one_fringe(actuators,pds,stepper1,stepper2,direction1,direction2,move_increment=10,fringe_min=0.5,power_history=[]):
    p0 = pds.get_measurement()[1][-1]
    p_ref = pds.get_measurement()[0][-1]
    if p0/p_ref < 0.01:
        print('Power too low, aborting')
        return power_history
    p1 = p0
    power_history.append(p0)
    
    # move stepper1 until power drops by fringe_min
    while p1/p0 > fringe_min:
        actuators.move_stepper(stepper1, direction1, move_increment)
        time.sleep(0.5)
        p1 = pds.get_measurement()[1][-1]
        power_history.append(p1)
    
    p0, p1, power_history = to_max_quantized(pds,actuators,stepper2,direction2,percent_change=0.02,move_increment=move_increment,power_history=power_history)

    return power_history        

def scuffed_beamwalking(actuators,pds,goal_power=0.8,move_increment= 10):
    # assume we've just done simple grad ascent
    print('Starting scuffed beamwalking...')
    power_history = []
    
    p0 = pds.get_measurement()[1][-1]
    p_ref = pds.get_measurement()[0][-1]
    if p0/p_ref < 0.01:
        print('Power too low, aborting')
        return power_history
    p1 = p0
    power_history.append(p0)

    x1_direction = 0
    x2_direction = 0
    y1_direction = 0
    y2_direction = 0

    x1 = 1
    x2 = 3
    y1 = 2
    y2 = 4

    # turn x1 until power drops by 25%
    while p1/p0 > 0.75:
        actuators.move_stepper(x1, x1_direction, move_increment)
        time.sleep(0.5)
        p1 = pds.get_measurement()[1][-1]
        power_history.append(p1)

    start_power = p1

    # find corresponding x2 direction
    p0 = p1
    while abs(p1 - p0)/p0 < 0.05:
        actuators.move_stepper(x2, x2_direction, move_increment)
        time.sleep(0.5)
        p1 = pds.get_measurement()[1][-1]
        power_history.append(p1)
    if p1 < p0:
        x2_direction = 1
    p0, p1, power_history = to_max_quantized(pds,actuators,x2,x2_direction,percent_change=0.02,move_increment=move_increment,power_history=power_history)

    # check if new power is better than start power, reverse directions if not
    if p0 < start_power:
        x1_direction = 1 - x1_direction
        x2_direction = 1 - x2_direction
    
    # turn y1 until power drops by 25%
    while p1/p0 > 0.75:
        actuators.move_stepper(y1, y1_direction, move_increment)
        time.sleep(0.5)
        p1 = pds.get_measurement()[1][-1]
        power_history.append(p1)

    start_power = p1

    # find corresponding y2 direction
    p0 = p1
    while abs(p1 - p0)/p0 < 0.05:
        actuators.move_stepper(y2, y2_direction, move_increment)
        time.sleep(0.5)
        p1 = pds.get_measurement()[1][-1]
        power_history.append(p1)
    if p1 < p0:
        y2_direction = 1
    p0, p1, power_history = to_max_quantized(pds,actuators,y2,y2_direction,percent_change=0.02,move_increment=move_increment,power_history=power_history)

    # check if new power is better than start power, reverse directions if not
    if p0 < start_power:
        y1_direction = 1 - y1_direction
        y2_direction = 1 - y2_direction

    if p0/p_ref >= goal_power:
        print('Power restored, no walking necessary')
        return power_history
    
    walking = True
    while walking:
        p0 = pds.get_measurement()[1][-1]
        power_history = walk_one_fringe(actuators,pds,x2,x1,x2_direction,x1_direction,move_increment=move_increment,fringe_min=.75,power_history=power_history)
        p1 = pds.get_measurement()[1][-1]
        if p1/p_ref >= goal_power:
            walking = False
            print('Goal power reached, stopping beamwalking.')
            return power_history
        elif p1/p_ref < 0.01:
            print('Power too low, aborting')
            return power_history
        power_history = walk_one_fringe(actuators,pds,x1,x2,x1_direction,x2_direction,move_increment=move_increment,fringe_min=.75,power_history=power_history)
        p1 = pds.get_measurement()[1][-1]
        if p1/p_ref >= goal_power:
            walking = False
            print('Goal power reached, stopping beamwalking.')
            return power_history
        elif p1/p_ref < 0.01:
            print('Power too low, aborting')
            return power_history
    
        if p1 < p0:
            print('Power went down after x walk, reversing directions.')
            x1_direction = 1 - x1_direction
            x2_direction = 1 - x2_direction
        
        p0 = pds.get_measurement()[1][-1]
        power_history = walk_one_fringe(actuators,pds,y2,y1,y2_direction,y1_direction,move_increment=move_increment,fringe_min=.75,power_history=power_history)
        p1 = pds.get_measurement()[1][-1]
        if p1/p_ref >= goal_power:
            walking = False
            print('Goal power reached, stopping beamwalking.')
            return power_history
        elif p1/p_ref < 0.01:
            print('Power too low, aborting')
            return power_history
        power_history = walk_one_fringe(actuators,pds,y1,y2,y1_direction,y2_direction,move_increment=move_increment,fringe_min=.75,power_history=power_history)
        p1 = pds.get_measurement()[1][-1]
        if p1/p_ref >= goal_power:
            walking = False
            print('Goal power reached, stopping beamwalking.')
            return power_history
        elif p1/p_ref < 0.01:
            print('Power too low, aborting')
            return power_history
    
        if p1 < p0:
            print('Power went down after y walk, reversing directions.')
            y1_direction = 1 - y1_direction
            y2_direction = 1 - y2_direction
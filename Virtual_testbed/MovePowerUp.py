import numpy as np
import random
import pandas as pd

params = pd.read_csv('param_fit.csv')['value']
min_max = pd.read_csv('min_max_positions.csv')['value']
min_xm1 = min_max[4]
min_ym1 = min_max[5]
min_xm2 = min_max[6]
min_ym2 = min_max[7]
max_xm1 = min_max[0]
max_ym1 = min_max[1]
max_xm2 = min_max[2]
max_ym2 = min_max[3]

def fit(x1, y1, x2, y2):
    return (np.exp(-(x1-params[0])**2/(2*params[1]**2))
     *np.exp(-(y1-params[2])**2/(2*params[3]**2))
     *np.exp(-(x2-params[4])**2/(2*params[5]**2))
     * np.exp(-(y2 - params[6]) ** 2 / (2 * params[7] ** 2))
     )

def move_power_up(actuator_positions, start_dir, startvalues_mean, initial_radius,
                      min_power_after_reset, max_power_to_neutral, number_of_random_steps_low_power,
                      max_random_reset_step_low_power, min_actuators_grid_scan, max_actuators_grid_scan,
                      min_power_stop_random_steps, reset_step_size):
    # with each act keep on moving in same direction while this moving helps (power_dif>0)
    # when this changes, change direction
    # p_diff = power_new - power_old # not needed I think...
    actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
            max_actuators_grid_scan - min_actuators_grid_scan)
    power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                        actuator_positions_standardized[2], actuator_positions_standardized[3])
    current_dir = start_dir
    while power_new < min_power_after_reset:
        # move to neutral positions and do some random steps for very small power
        if power_new < max_power_to_neutral:
            power_new, actuator_positions, actuator_positions_standardized = to_neutral_positions_random_steps(
                startvalues_mean, initial_radius,
                       max_power_to_neutral, number_of_random_steps_low_power,
                      max_random_reset_step_low_power, min_actuators_grid_scan, max_actuators_grid_scan,
                      min_power_stop_random_steps)
        else:
            # move in current direction while power gets better (each actuator individually)
            shuffled_list = random.sample(range(4), k=4)
            for i in shuffled_list:
                power_old = power_new
                rand = np.random.uniform(low=0.5, high=2.0)  # add some randomness to step size
                actuator_positions[i] = actuator_positions[i] + int(current_dir[i]*reset_step_size*rand)
                actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
                        max_actuators_grid_scan - min_actuators_grid_scan)
                power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                                actuator_positions_standardized[2], actuator_positions_standardized[3])
                p_diff = power_new - power_old
                if p_diff < -0.002:  # reverse last action and change direction.
                    power_old = power_new
                    actuator_positions[i] = actuator_positions[i] - int(current_dir[i] * reset_step_size * rand)
                    actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
                            max_actuators_grid_scan - min_actuators_grid_scan)
                    power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                                    actuator_positions_standardized[2], actuator_positions_standardized[3])
                    p_diff = power_new - power_old
                    current_dir[i] = -current_dir[i]
                while p_diff >= 0 and power_new < min_power_after_reset:  # 2nd condition added so that loop is stoped when threshold power is reached
                    power_old = power_new
                    # use >= so that it keeps on moving in this direction if one movement was so small that it had no impact at all.
                    rand = np.random.uniform(low=0.5, high=2.0)  # add some randomness to step size
                    actuator_positions[i] = actuator_positions[i] + int(current_dir[i] * reset_step_size * rand)
                    actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
                            max_actuators_grid_scan - min_actuators_grid_scan)
                    power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                                    actuator_positions_standardized[2], actuator_positions_standardized[3])
                    p_diff = power_new - power_old
                    if p_diff < 0:  # reverse last action
                        actuator_positions[i] = actuator_positions[i] - int(current_dir[i] * reset_step_size * rand)
                        actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
                                max_actuators_grid_scan - min_actuators_grid_scan)
                        power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                                        actuator_positions_standardized[2], actuator_positions_standardized[3])
                if power_new >= min_power_after_reset:
                    break
            current_dir = (-1)*current_dir  # change direction
    return actuator_positions

def to_neutral_positions_random_steps(startvalues_mean, initial_radius,
                       max_power_to_neutral, number_of_random_steps_low_power,
                      max_random_reset_step_low_power, min_actuators_grid_scan, max_actuators_grid_scan,
                      min_power_stop_random_steps):
    # move to neutral position
    actuator_positions = startvalues_mean
    actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
            max_actuators_grid_scan - min_actuators_grid_scan)
    power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                    actuator_positions_standardized[2], actuator_positions_standardized[3])
    # do random steps if power (still) small (not really necessary here, but in Lab)
    if power_new < max_power_to_neutral:
        for j in range(number_of_random_steps_low_power):
            add_random_steps = np.array([random.randint(- max_random_reset_step_low_power,
                                                        max_random_reset_step_low_power) for _ in
                                         range(4)])  # to be tested what is a good random step!!
            actuator_positions = actuator_positions + add_random_steps
            actuator_positions_standardized = (actuator_positions - min_actuators_grid_scan) / (
                    max_actuators_grid_scan - min_actuators_grid_scan)
            power_new = fit(actuator_positions_standardized[0], actuator_positions_standardized[1],
                            actuator_positions_standardized[2], actuator_positions_standardized[3])
            if power_new > min_power_stop_random_steps:
                break
    return power_new, actuator_positions, actuator_positions_standardized



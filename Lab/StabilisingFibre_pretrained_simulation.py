import time

from stable_baselines3 import SAC
from EnvFibreGoal220424 import *
from CustomTensorboardCallback import *
import Photodetector
from pylablib.devices import Thorlabs
import safe_exit
from sb3_contrib import TQC


def main():
    actxm1 = Thorlabs.KinesisMotor("26004585")
    actym1 = Thorlabs.KinesisMotor("26004587")
    actxm2 = Thorlabs.KinesisMotor("26003852")
    actym2 = Thorlabs.KinesisMotor("26003794")
    actuators = [actxm1, actym1, actxm2, actym2]

    pd2 = Photodetector.Photodetector('USB0::0x1313::0x807B::1922851::0::INSTR')
    pd1 = Photodetector.Photodetector('USB0::0x1313::0x807B::1922850::0::INSTR')  # reference
    pd1.clear()  # this should solve the famous Visa Error.
    pd2.clear()

    pds = [pd1, pd2]
    max_actioninsteps = 6000
    reset_power_fail = 0.05  # 0.0009
    reset_power_goal = 0.9
    min_power_after_reset = 0.2  # 0.001
    max_power_after_reset = reset_power_goal
    max_episode_steps = 30

    # close power meters when program is stopped or an error occurs
    @safe_exit.register
    def cleanup():
        pd1.close()
        pd2.close()
        print("cleanup called")

    # reward parameters
    beta_step = 5
    beta_fail_1 = 5
    beta_fail_2 = 5
    beta_goal_1 = 5
    beta_goal_2 = 1
    alpha_goal = 0.5
    alpha_fail = 0.5
    alpha_step = 0.9
    prefactor_step = 10
    prefactor_goal = 100
    prefactor_fail = 100
    reward_fct_descriptor_2024_04_22 = (
        f"reward_2024_04_22_betas_{beta_step}_{beta_fail_1}_{beta_fail_2}"
        f"_{beta_goal_1}_{beta_goal_2}_prefactor_{prefactor_step}_"
        f"{prefactor_fail}_{prefactor_goal}_alphas_{alpha_step}_{alpha_fail}_{alpha_goal}")
    dir_names = f"goal/pretrained_on_simulation/start_with_1715764610_500000"  # subdirectory to save models, logs,...

    def reward_fct_2024_04_22(avg_power, max_power, power, reset_power_fail, max_episode_steps,
                                  reset_power_goal, min_power_after_reset, current_step):
        if power > reset_power_goal:
            reward = prefactor_goal * (
                    ((1 - alpha_goal) * np.exp(-beta_goal_1 * current_step / max_episode_steps))
                    + alpha_goal * np.exp(beta_goal_2 * power / reset_power_goal))
        elif power < reset_power_fail:
            reward = - prefactor_fail * (
                    (1 - alpha_fail) * np.exp(-beta_fail_1 * current_step / max_episode_steps)
                    + alpha_fail * np.exp(-beta_fail_2 * power / reset_power_fail))
        else:
            reward = prefactor_step / max_episode_steps * ((1 - alpha_step) * np.exp(
                beta_step * (power - reset_power_goal)) + alpha_step * (power - min_power_after_reset))
        return reward
    # load model pretrained on virtual testbed
    num = 500000
    timestamp = None
    env = Env_fiber_move_by_grad_reset(actuators, pds, max_actioninsteps, reset_power_fail, reset_power_goal,
                 reward_fct_2024_04_22, reward_fct_descriptor_2024_04_22, min_power_after_reset, max_power_after_reset,
                                       timestamp=timestamp, dir_names=dir_names, save_replay=True,
                                       max_episode_steps=max_episode_steps)
    env.reset()
    models_dir = "./simulation/models/A/1715764610"
    model_path = models_dir+"/"+str(num)
    replay_dir = env.replay_dir
    replay_path = replay_dir+"/"+str(num)
    log_path = env.logdir
    model = TQC.load(model_path, tensorboard_log=log_path)
    model.set_env(env)
    #model.load_replay_buffer(old_replay_path, truncate_last_traj=True)

    # start training (for 200000 training steps)

    TIMESTEPS = 1000
    for i in range(200):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="TQC",
                    callback=CustomTensorboardCallback(env))
        model.save(f"{env.models_dir}/{num + TIMESTEPS * (i+1)}")
        model.save_replay_buffer(f"{env.replay_dir}/{num + TIMESTEPS * (i+1)}")

    env.close()


if __name__ == '__main__':
    main()

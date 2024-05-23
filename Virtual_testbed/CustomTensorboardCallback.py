from stable_baselines3.common.callbacks import BaseCallback


class CustomTensorboardCallback(BaseCallback):
    """
    Custom callback for plotting  power, actuator positions additionally in tensorboard
    """

    def __init__(self, env, verbose=0):
        super().__init__(verbose)
        self.env = env

    def _on_step(self) -> bool:
        # Log power, actuator positions
        power = self.env.info["power"]
        self.logger.record("power", power)
        act_1x_pos = self.env.info["act_1x_pos"]
        self.logger.record("act_1x_pos", act_1x_pos)
        act_1y_pos = self.env.info["act_1y_pos"]
        self.logger.record("act_1y_pos", act_1y_pos)
        act_2x_pos = self.env.info["act_2x_pos"]
        self.logger.record("act_2x_pos", act_2x_pos)
        act_2y_pos = self.env.info["act_2y_pos"]
        self.logger.record("act_2y_pos", act_2y_pos)
        return True
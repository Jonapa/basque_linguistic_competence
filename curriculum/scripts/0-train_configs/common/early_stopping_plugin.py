from axolotl.integrations.base import BasePlugin
from transformers import TrainerCallback

class EarlyStopEpochPlugin(BasePlugin):
    def add_callbacks_post_trainer(self, cfg, trainer):
        stop_epoch = cfg.get("early_stop_epoch", 1)
        return [EarlyStopCallback(stop_epoch)]

class EarlyStopCallback(TrainerCallback):
    def __init__(self, stop_epoch: int):
        self.stop_epoch = float(stop_epoch)

    def on_epoch_end(self, args, state, control, **kwargs):
        # state.epoch can be float; also can be None in some step-based runs
        if state.epoch is not None and state.epoch >= self.stop_epoch:
            print(f"Stopping training at epoch {state.epoch}")
            control.should_training_stop = True
        return control
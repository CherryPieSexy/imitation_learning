from tensorboardX import SummaryWriter


# TODO: add logging here
class OnPolicyTrainer:
    """
    Simple on-policy trainer.
    """
    def __init__(self):
        # here should be (at least):
        #     algorithm/agent instance (A2C or PPO or something)
        #     train vec env instance
        #     test single (or vec? vec will be better) env instance
        #     'log_dir' str to save checkpoints into, or may be 'checkpoints_dir'
        self._writer = SummaryWriter  # instantiate this
        # TODO: it will be great if we write episode reward
        #  for each train env with this writer. I see two options here:
        #  1) Track environment.done in gather rollout and write logs by trainer
        #  2) Create separate 'RewardTrackingWrapper' environment wrapper
        #  2-nd option is better

    def _gather_rollout(self):
        pass

    def _write_logs(self, tag, values, step):
        for key, value in values.item():
            self._writer.add_scalar(tag + key, value, step)

    def _train_step(self):
        # gather rollout -> train on it -> write training logs
        pass

    def _save_checkpoint(self, checkpoint_name):
        pass

    def _test_agent(self):
        pass

    def train(self, n_epoch, n_steps, rollout_len, n_tests):
        """
        Run training for 'n_epoch', each epoch takes 'n_steps' training steps
        on rollouts of len 'rollout_len'.
        At the end of each epoch run 'n_tests' tests and saves checkpoint

        :param n_epoch:
        :param n_steps:
        :param rollout_len:
        :param n_tests:
        :return:
        """
        pass

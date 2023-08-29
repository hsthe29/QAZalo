import tensorflow as tf


class WarmupLinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, lr, warmup_steps, total_steps, min_lr=0.3):
        super(WarmupLinearSchedule, self).__init__()
        self.lr = lr
        self.warmup_steps = float(warmup_steps)
        self.cooldown_steps = float(total_steps - warmup_steps)
        self.total_steps = total_steps
        self.min_lr = lr*min_lr

    def __call__(self, step):
        if step < self.warmup_steps:
            return max(self.min_lr/2, self.lr * float(step) / max(1.0, self.warmup_steps))
        return max(self.min_lr, self.lr * float(self.total_steps - step) / max(1.0, self.cooldown_steps))

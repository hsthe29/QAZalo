import tensorflow as tf


class WarmupLinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, lr, warmup_steps, total_steps, min_lr=0.3):
        super(WarmupLinearSchedule, self).__init__()
        self.lr = lr
        self.warmup_steps = tf.cast(warmup_steps, dtype=tf.float32)
        self.cooldown_steps = tf.cast(total_steps - warmup_steps, dtype=tf.float32)
        self.total_steps = tf.cast(total_steps, dtype=tf.float32)
        self.min_lr = lr*min_lr

    def __call__(self, step):
        cond = step < self.warmup_steps
        true_fn = lambda: tf.math.maximum(self.min_lr/2, self.lr * step / self.warmup_steps)
        false_fn = lambda: tf.math.maximum(self.min_lr, self.lr * (self.total_steps - step) / self.cooldown_steps)
        return tf.cond(cond, true_fn, false_fn)  # Use tf.cond() explicitly.

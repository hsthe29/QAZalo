import tensorflow as tf


class WarmupLinearSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, init_lr, num_train_steps, num_warmup_steps):
        super(WarmupLinearSchedule, self).__init__()
        self.init_lr = init_lr
        self.lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            init_lr,
            num_train_steps - num_warmup_steps,
            0.0,
            power=1)
        self.num_train_steps = tf.cast(num_train_steps, dtype=tf.float32)
        self.num_warmup_steps = tf.cast(num_warmup_steps, dtype=tf.float32)

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        is_warmup = tf.cast(step < self.num_warmup_steps, tf.float32)

        decay_step = (1.0 - is_warmup) * (step - self.num_warmup_steps) + is_warmup * step

        learning_rate = self.lr_fn(decay_step)

        warmup_percent_done = step / self.num_warmup_steps
        warmup_learning_rate = self.init_lr * warmup_percent_done
        final_learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)
        return final_learning_rate

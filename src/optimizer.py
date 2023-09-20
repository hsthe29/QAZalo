import tensorflow as tf
from schedule import WarmupLinearSchedule


def create_optimizer(init_lr,
                     num_train_steps,
                     num_warmup_steps,
                     weight_decay_rate=0.001,
                     epsilon=1e-7):
    """Creates an optimizer training op."""
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    lr_schedule = WarmupLinearSchedule(init_lr, num_train_steps, num_warmup_steps)

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=weight_decay_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=epsilon
    )
    optimizer.exclude_from_weight_decay(var_names=["layer_norm", "bias"])

    return optimizer

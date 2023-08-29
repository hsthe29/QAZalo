import tensorflow as tf


def train(model, dataset):

    for i in range(0, 10):
        # Forward
        with tf.GradientTape() as tape:
            loss = model(0, training=True)
            pass
        # Backward
        variables = model.trainable_variables
        gradient = tape.gradient(loss, variables)
        model.optimizer.apply_gradient(gradient)


if __name__ == "__main__":
    pass

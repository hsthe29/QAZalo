import tensorflow as tf
from transformers import AdamWeightDecay, WarmUp
from schedule import WarmupLinearSchedule
from metrics import F1_Score


def create_optimizer(init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer with learning rate schedule."""
    # Implements linear decay of the learning rate.
    lr_schedule = WarmupLinearSchedule(init_lr, num_warmup_steps, num_train_steps)

    optimizer = AdamWeightDecay(
        learning_rate=lr_schedule,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["layer_norm", "bias"],
    )

    return optimizer


@tf.function
def train(model, max_epochs, train_dataset, val_dataset=None):
    steps_per_epoch = train_dataset.cardinality()
    total_steps = max_epochs * steps_per_epoch
    optimizer = create_optimizer(init_lr=1e-3, num_train_steps=total_steps, num_warmup_steps=2000)
    f1_fn = F1_Score()
    accuracy_fn = tf.keras.metrics.Accuracy()

    for ep in range(max_epochs):
        ground_truths = []
        predictions = []
        total_loss, logging_loss = 0.0, 0.0
        tf.print(f"#Epoch: {ep + 1}/{max_epochs}:")
        for step, (input_ids, attention_masks, labels) in enumerate(train_dataset):
            # Forward
            with tf.GradientTape() as tape:
                loss_value, predicts = model.train_loss((input_ids, attention_masks), labels, training=True)

            ground_truths.append(labels)
            predictions.append(predicts)
            total_loss += loss_value
            # backward
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Log every 10 batches.
            if step % 10 == 0:
                tf.print(f"Batch training result at step {step + 1}/{steps_per_epoch}:")

                f1_score = f1_fn(labels, predicts)
                accuracy = accuracy_fn(labels, predicts)
                tf.print(f"    Loss: {round(float(loss_value), 4)}")
                tf.print(f"    F1: {round(float(f1_score), 4)}")
                tf.print(f"    Acc: {round(float(accuracy), 4)}")

        ground_truths = tf.squeeze(tf.concat(ground_truths, axis=0))
        predictions = tf.squeeze(tf.concat(predictions, axis=0))
        f1_score = f1_fn(ground_truths, predictions)
        accuracy = accuracy_fn(ground_truths, predictions)

        tf.print("Epoch summary:")
        tf.print(f"    Mean training loss: {round(float(total_loss) / steps_per_epoch, 4)}")
        tf.print(f"    F1: {round(float(f1_score), 4)}")
        tf.print(f"    Acc: {round(float(accuracy), 4)}")

        output_validation = evaluate(val_dataset, model)
        tf.print("Evaluation result:")
        tf.print(f"    Loss: {output_validation['loss']}")
        tf.print(f"    F1: {output_validation['f1']}")
        tf.print(f"    Acc: {output_validation['accuracy']}")


@tf.function
def evaluate(eval_dataset, model):
    # Eval!
    tf.print("--- Running evaluation ---")
    steps = eval_dataset.cardinality()
    total_loss = 0.0
    ground_truths = []
    predictions = []

    f1_fn = F1_Score()
    accuracy_fn = tf.keras.metrics.Accuracy()

    for step, (input_ids, attention_masks, labels) in enumerate(eval_dataset):
        loss_value, predicts = model.train_loss((input_ids, attention_masks), labels, training=False)
        ground_truths.append(labels)
        predictions.append(predicts)
        total_loss += loss_value

    ground_truths = tf.squeeze(tf.concat(ground_truths, axis=0))
    predictions = tf.squeeze(tf.concat(predictions, axis=0))
    f1_score = f1_fn(ground_truths, predictions)
    accuracy = accuracy_fn(ground_truths, predictions)

    output_validation = {
        "loss": round(total_loss / steps, 4),
        "accuracy": round(accuracy, 4),
        "f1": round(f1_score, 4)
    }

    return output_validation

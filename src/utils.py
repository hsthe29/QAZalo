import tensorflow as tf
from transformers import AdamWeightDecay, WarmUp
from schedule import WarmupLinearSchedule
from metrics import F1_Score
import sys


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
def __train_step(model, optimizer, inputs):
    x, y = inputs
    with tf.GradientTape() as tape:
        loss_value, predicts = model.calculate_loss(x, y, training=True)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value, predicts


def train(model,
          max_epochs,
          train_dataset,
          val_dataset=None,
          save_per_epochs=2,
          fix_epochs=10,
          save_model_paths="../save/bert-qa"):
    steps_per_epoch = train_dataset.cardinality()
    total_steps = max_epochs * steps_per_epoch
    optimizer = create_optimizer(init_lr=1e-3, num_train_steps=total_steps, num_warmup_steps=2000)
    f1_fn = F1_Score()
    accuracy_fn = tf.keras.metrics.Accuracy()

    tf.print("#*#*#*# Training #*#*#*#", output_stream=sys.stdout)

    for ep in range(max_epochs):
        if ep == fix_epochs:
            model.set_trainable(True)

        ground_truths = []
        predictions = []
        total_loss, logging_loss = 0.0, 0.0
        tf.print(f"    #Epoch: {ep + 1}/{max_epochs}:", output_stream=sys.stdout)
        for step, (input_ids, attention_masks, labels) in enumerate(train_dataset):

            loss_value, predicts = __train_step(model, optimizer, ((input_ids, attention_masks), labels))
            ground_truths.append(labels)
            predictions.append(predicts)
            total_loss += loss_value

            # Log every 10 batches.
            if step % 10 == 0:
                tf.print(f"    Batch training result at step {step + 1}/{steps_per_epoch}:", output_stream=sys.stdout)

                f1_score = f1_fn(labels, predicts)
                accuracy = accuracy_fn(labels, predicts)
                tf.print(f"      -> Loss: {loss_value}", output_stream=sys.stdout)
                tf.print(f"      -> F1: {f1_score}", output_stream=sys.stdout)
                tf.print(f"      -> Acc: {accuracy}", output_stream=sys.stdout)

        ground_truths = tf.squeeze(tf.concat(ground_truths, axis=0))
        predictions = tf.squeeze(tf.concat(predictions, axis=0))
        f1_score = f1_fn(ground_truths, predictions)
        accuracy = accuracy_fn(ground_truths, predictions)

        tf.print("    Epoch summary:", output_stream=sys.stdout)
        tf.print(f"      -> Mean training loss: {total_loss / tf.cast(steps_per_epoch, dtype=tf.float32)}",
                 output_stream=sys.stdout)
        tf.print(f"      -> F1: {f1_score}", output_stream=sys.stdout)
        tf.print(f"      -> Acc: {accuracy}", output_stream=sys.stdout)

        if (ep + 1) % save_per_epochs == 0:
            tf.print(f"--- Saving model to \"{save_model_paths}\" ---", output_stream=sys.stdout)
            model.save(save_model_paths)

        if val_dataset is not None:
            output_validation = evaluate(val_dataset, model)
            tf.print("    Evaluation result:", output_stream=sys.stdout)
            tf.print(f"      -> Loss: {output_validation['loss']}", output_stream=sys.stdout)
            tf.print(f"      -> F1: {output_validation['f1']}", output_stream=sys.stdout)
            tf.print(f"      -> Acc: {output_validation['accuracy']}", output_stream=sys.stdout)



@tf.function
def __test_step(model, inputs):
    x, y = inputs
    loss_value, predicts = model.calculate_loss(x, y, training=False)
    return loss_value, predicts


def evaluate(eval_dataset, model):
    # Eval!
    tf.print("    --- Running evaluation ---", output_stream=sys.stdout)
    steps = eval_dataset.cardinality()
    total_loss = 0.0
    ground_truths = []
    predictions = []

    f1_fn = F1_Score()
    accuracy_fn = tf.keras.metrics.Accuracy()

    for step, (input_ids, attention_masks, labels) in enumerate(eval_dataset):
        loss_value, predicts = __test_step(model, ((input_ids, attention_masks), labels))
        ground_truths.append(labels)
        predictions.append(predicts)
        total_loss += loss_value

    ground_truths = tf.squeeze(tf.concat(ground_truths, axis=0))
    predictions = tf.squeeze(tf.concat(predictions, axis=0))
    f1_score = f1_fn(ground_truths, predictions)
    accuracy = accuracy_fn(ground_truths, predictions)

    output_validation = {
        "loss": total_loss / tf.cast(steps, dtype=tf.float32),
        "accuracy": accuracy,
        "f1": f1_score
    }

    return output_validation

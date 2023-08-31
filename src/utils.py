import tensorflow as tf
from transformers import AdamWeightDecay, WarmUp
import numpy as np
from bert_model.bert import BERT_QA
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
    variables = model.trainable_variables
    grads = tape.gradient(loss_value, variables)
    optimizer.apply_gradients(zip(grads, variables))
    return loss_value, predicts


def train(model,
          max_epochs,
          train_dataset,
          val_dataset=None,
          print_steps=5,
          save_per_epochs=2,
          fix_epochs=10,
          weight_paths="../save/weights/bert-qa"
          ):
    steps_per_epoch = train_dataset.cardinality()
    total_steps = (max_epochs + 2) * steps_per_epoch
    optimizer = create_optimizer(init_lr=1e-3, num_train_steps=total_steps, num_warmup_steps=200)
    f1_fn = F1_Score()
    accuracy_fn = tf.keras.metrics.Accuracy()
    history = {"epochs": max_epochs, "loss": [], "f1_score": [], "accuracy": []}

    if val_dataset is not None:
        history["val_loss"] = []
        history["val_f1_score"] = []
        history["val_accuracy"] = []

    tf.print("#*#*#*# Training #*#*#*#", output_stream=sys.stdout)

    for ep in range(max_epochs):
        if ep == fix_epochs:
            model.set_trainable(True)

        ground_truths = []
        predictions = []
        total_loss, logging_loss = 0.0, 0.0
        tf.print(f"#Epoch: {ep + 1}/{max_epochs}:", output_stream=sys.stdout)
        for step, (*inputs, labels) in enumerate(train_dataset):

            loss_value, predicts = __train_step(model, optimizer, (inputs, labels))
            ground_truths.append(labels)
            predictions.append(predicts)
            total_loss += loss_value

            # Log every 10 batches.
            if (step + 1) % print_steps == 0:
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
        mean_loss = total_loss / tf.cast(steps_per_epoch, dtype=tf.float32)

        history["loss"].append(mean_loss)
        history["f1_score"].append(f1_score)
        history["accuracy"].append(accuracy)

        tf.print("    Epoch summary:", output_stream=sys.stdout)
        tf.print(f"      -> Mean training loss: {mean_loss}",
                 output_stream=sys.stdout)
        tf.print(f"      -> F1: {f1_score}", output_stream=sys.stdout)
        tf.print(f"      -> Acc: {accuracy}", output_stream=sys.stdout)

        if (ep + 1) % save_per_epochs == 0:
            tf.print(f"--- Saving model to \"{weight_paths}\" ---", output_stream=sys.stdout)
            model.save_weights(weight_paths, save_format="tf")

        if val_dataset is not None:
            output_validation = evaluate(val_dataset, model)

            history["val_loss"].append(output_validation['loss'])
            history["val_f1_score"].append(output_validation['f1'])
            history["val_accuracy"].append(output_validation['accuracy'])

            tf.print("    Evaluation result:", output_stream=sys.stdout)
            tf.print(f"      -> Loss: {output_validation['loss']}", output_stream=sys.stdout)
            tf.print(f"      -> F1: {output_validation['f1']}", output_stream=sys.stdout)
            tf.print(f"      -> Acc: {output_validation['accuracy']}", output_stream=sys.stdout)

    history["loss"] = np.array(list(map(lambda x: x.numpy(), history["loss"])))
    history["f1_score"] = np.array(list(map(lambda x: x.numpy(), history["f1_score"])))
    history["accuracy"] = np.array(list(map(lambda x: x.numpy(), history["accuracy"])))

    if val_dataset is not None:
        history["val_loss"] = np.array(list(map(lambda x: x.numpy(), history["val_loss"])))
        history["val_f1_score"] = np.array(list(map(lambda x: x.numpy(), history["val_f1_score"])))
        history["val_accuracy"] = np.array(list(map(lambda x: x.numpy(), history["val_accuracy"])))

    return history


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

    for step, (*inputs, labels) in enumerate(eval_dataset):
        loss_value, predicts = __test_step(model, (inputs, labels))
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


def load_pretrained_bert_qa(weight_path):
    with open("../save/configs/bert-qa.json", "r") as f:
        config = f.read()
        model = tf.keras.models.model_from_json(config, custom_objects={"BERT_QA": BERT_QA})
    model.load_weights(weight_path)
    return model

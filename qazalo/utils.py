import sys
import tensorflow as tf
import pandas as pd
from .model import BertClassifier
from .optimizer import create_optimizer
from .metrics import create_metrics


def train(data,
          flags,
          strategy=None):
    if hasattr(data, '__len__'):
        if len(data) == 1:
            train_ds = data[0]
            val_ds = None
        elif len(data) == 2:
            train_ds, val_ds = data
        else:
            raise ValueError("There is no data or too many data passed!")
    else:
        train_ds = data
        val_ds = None

    num_train_steps = train_ds.cardinality().numpy() * flags.EPOCHS
    num_warmup_steps = int(num_train_steps * flags.warmup_proportion)

    if strategy:
        with strategy.scope():
            optimizer = create_optimizer(flags.init_lr,
                                         num_train_steps,
                                         num_warmup_steps,
                                         weight_decay_rate=flags.weight_decay
                                         )
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=flags.from_logits)
            metrics = create_metrics(from_logits=flags.from_logits)
            if flags.from_scratch:
                model = BertClassifier(num_classes=flags.num_classes,
                                       use_pooler=flags.use_pooler,
                                       to_logits=flags.from_logits)
            else:
                pretrained_name = f"zqa-{flags.num_classes}-P{int(flags.use_pooler)}-L{int(flags.from_logits)}"
                model = BertClassifier.from_pretrained(pretrained_name)
            model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics.values()))
    else:
        optimizer = create_optimizer(flags.init_lr,
                                     num_train_steps,
                                     num_warmup_steps,
                                     weight_decay_rate=flags.weight_decay
                                     )
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=flags.from_logits)
        metrics = create_metrics(from_logits=flags.from_logits)
        if flags.from_scratch:
            model = BertClassifier(num_classes=flags.num_classes,
                                   use_pooler=flags.use_pooler,
                                   to_logits=flags.from_logits)
        else:
            pretrained_name = f"zqa-{flags.num_classes}-P{int(flags.use_pooler)}-L{int(flags.from_logits)}"
            model = BertClassifier.from_pretrained(pretrained_name)
        model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics.values()))

    # Metrics

    logger = tf.keras.callbacks.CSVLogger(flags.log_dir + "train_log.csv", separator=',', append=False)
    ckpt_name = f"zqa-{flags.num_classes}-P{int(flags.use_pooler)}-L{int(flags.from_logits)}"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(flags.save_dir + ckpt_name,
                                                    monitor=flags.monitor,
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='max')
    update_freq = flags.update_freq
    if update_freq != "epoch" and update_freq != "batch":
        update_freq = int(update_freq)
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=flags.log_dir + "tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq=update_freq,
        profile_batch=0
    )

    tf.print("###### START TRAINING ######", output_stream=sys.stdout)
    history_callback = model.fit(train_ds,
                                 epochs=flags.EPOCHS,
                                 validation_data=val_ds,
                                 callbacks=[checkpoint, logger, tensorboard])
    tf.print("###### DONE ######", output_stream=sys.stdout)
    return history_callback


def evaluate(val_ds, flags, strategy=None):
    if strategy:
        with strategy.scope():
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=flags.from_logits)
            metrics = create_metrics(from_logits=flags.from_logits)
            model = BertClassifier.from_pretrained(flags.pretrained_name)
            model.compile(loss=loss, metrics=list(metrics.values()))
    else:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=flags.from_logits)
        metrics = create_metrics(from_logits=flags.from_logits)
        model = BertClassifier.from_pretrained(flags.pretrained_name)
        model.compile(loss=loss, metrics=list(metrics.values()))

    logger_path = flags.log_dir + "validation_log.csv"

    tf.print("###### START EVALUATING ######", output_stream=sys.stdout)
    evaluate_result = model.evaluate(val_ds, return_dict=True)
    tf.print("###### DONE ######", output_stream=sys.stdout)
    tf.print(f"Result: {evaluate_result}", output_stream=sys.stdout)
    df = pd.DataFrame([evaluate_result])
    df.to_csv(logger_path, index=False)
    tf.print(f"Saved result to {logger_path}")


def predict(test_ds, test_id, flags, strategy=None):
    if strategy:
        with strategy.scope():
            model = BertClassifier.from_pretrained(flags.pretrained_name)
    else:
        model = BertClassifier.from_pretrained(flags.pretrained_name)

    tf.print("###### START PREDICTION ######", output_stream=sys.stdout)
    logits = model.predict(test_ds)
    tf.print("###### DONE ######", output_stream=sys.stdout)
    y_pred = tf.where(logits >= 0.0, 1, 0)
    y_pred = tf.reshape(y_pred, shape=(-1,)).numpy()

    tf.print(f"Writing prediction to {flags.output}", output_stream=sys.stdout)

    with open(flags.output, "w") as f:
        for t_id, pred in zip(test_id, y_pred):
            f.write(f"{t_id}\t{bool(pred)}\n")

    tf.print("Done", output_stream=sys.stdout)



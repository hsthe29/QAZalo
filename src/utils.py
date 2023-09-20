import sys
import tensorflow as tf
from optimizer import create_optimizer
from metrics import create_metrics
import pandas as pd


def train(model,
          data,
          flags):
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
    optimizer = create_optimizer(flags.init_lr,
                                 num_train_steps,
                                 num_warmup_steps,
                                 weight_decay_rate=flags.weight_decay
                                 )
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=flags.from_logits)
    metrics = create_metrics(from_logits=flags.from_logits)
    model.compile(loss=loss, optimizer=optimizer, metrics=list(metrics.values()))

    logger = tf.keras.callbacks.CSVLogger(flags.log_dir + "train_logs.csv", separator=',', append=False)
    ckpt_name = f"zqa-{flags.num_classes}-P{int(flags.use_pooler)}-L{int(flags.from_logits)}"

    checkpoint = tf.keras.callbacks.ModelCheckpoint(flags.save_dir + ckpt_name,
                                                    monitor='val_f1',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    save_weights_only=True,
                                                    mode='max')
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=flags.log_dir + "tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        write_steps_per_second=False,
        update_freq='epoch',
        profile_batch=0
    )

    tf.print("###### START TRAINING ######", output_stream=sys.stdout)
    history_callback = model.fit(train_ds,
                                 epochs=flags.EPOCHS,
                                 validation_data=val_ds,
                                 callbacks=[checkpoint, logger, tensorboard])
    tf.print("###### DONE ######", output_stream=sys.stdout)


def evaluate(model, val_ds, flags):
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=flags.from_logits)
    metrics = create_metrics(from_logits=flags.from_logits)
    model.compile(loss=loss, metrics=list(metrics.values()))

    logger_path = flags.log_dir + "validation_logs.csv"

    tf.print("###### START EVALUATING ######", output_stream=sys.stdout)
    evaluate_result = model.evaluate(val_ds, return_dict=True)
    tf.print("###### DONE ######", output_stream=sys.stdout)
    tf.print(f"Result: {evaluate_result}", output_stream=sys.stdout)
    df = pd.DataFrame([evaluate_result])
    df.to_csv(logger_path, index=False)


def predict(model, test_ds, flags):
    logits = model.predict(test_ds)
    y_pred = tf.where(logits >= 0.0, 1, 0)
    y_pred = tf.reshape(y_pred, shape=(-1,)).numpy()
    with open(flags.output_dir + "submission.txt", "w") as f:
        for pred_label in y_pred:
            f.write(str(pred_label) + "\n")



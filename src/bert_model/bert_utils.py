from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf


def load_pretrained_bert():
    return TFAutoModel.from_pretrained("vinai/phobert-base", output_hidden_states=True)


def load_tokenizer():
    return AutoTokenizer.from_pretrained("vinai/phobert-base-v2")


def make_classifier(num_classes, use_pooler, dropout_rate, hidden_units=768):
    if use_pooler:
        return tf.keras.Sequential([
            tf.keras.Input(shape=hidden_units),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes)
        ])
    else:
        if hidden_units is None:
            raise ValueError("Without using pooler, hidden_units must not be None")
        return tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units*4,)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(hidden_units, activation="swish"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes)
        ])

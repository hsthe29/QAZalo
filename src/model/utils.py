import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


def load_pretrained_bert():
    return TFBertModel.from_pretrained("bert-base-multilingual-cased", output_hidden_states=True)


def load_tokenizer():
    return BertTokenizer.from_pretrained("bert-base-multilingual-cased")


def build_classifier(num_classes, use_pooler, dropout_rate, hidden_units=768):
    if use_pooler:
        return tf.keras.Sequential([
            tf.keras.Input(shape=hidden_units),
            tf.keras.layers.Dense(num_classes)
        ])
    else:
        if hidden_units is None:
            raise ValueError("Without using pooler, hidden_units must not be None")
        return tf.keras.Sequential([
            tf.keras.Input(shape=(hidden_units*4,)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(num_classes)
        ])


def retrieve(name):
    config_path = "config/" + name + ".json"
    weights_path = "checkpoint/" + name
    return config_path, weights_path

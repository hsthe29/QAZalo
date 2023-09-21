import tensorflow as tf
from tensorflow import keras
from .utils import load_pretrained_bert, build_classifier, retrieve
import json


class BertClassifier(keras.Model):
    def __init__(self, num_classes, use_pooler=False, to_logits=True, dropout_rate=0.1, hidden_units=768,
                 class_weights=None):
        super(BertClassifier, self).__init__()
        self.num_classes = num_classes

        self.backbone = load_pretrained_bert()
        self.use_pooler = use_pooler
        num_outputs = 1 if num_classes == 2 else num_classes
        self.classifier = build_classifier(num_outputs, use_pooler, dropout_rate, hidden_units=hidden_units, to_logits=to_logits)

        self.class_weights = class_weights

    def freeze_bert(self, value=False):
        self.backbone.trainable = not value

    def call(self, inputs, training):
        features = self.backbone(inputs)

        if self.use_pooler:
            pooler_output = features['pooler_output']
            logits = self.classifier(pooler_output, training=training)
        else:
            all_hidden_states = features['hidden_states'][-4:]
            hidden_states = tf.stack(all_hidden_states, axis=1)

            hidden_states = hidden_states[:, :, 0, :]
            shape = tf.shape(hidden_states)
            cls_outputs = tf.reshape(hidden_states, [shape[0], shape[1] * shape[2]])

            logits = self.classifier(cls_outputs, training=training)
        return logits

    @classmethod
    def from_pretrained(cls, name):
        config_path, weights_path = retrieve(name)
        with open(config_path, "r") as f:
            config = json.load(f)
        obj = cls(**config)
        obj.load_weights(weights_path)
        return obj

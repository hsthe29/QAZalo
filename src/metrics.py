from tensorflow import keras
import tensorflow as tf


def create_metrics(from_logits=False):
    threshold = 0.0 if from_logits else 0.5
    return {
        "accuracy": LogitsAccuracy(threshold=threshold),
        "f1": LogitsF1(from_logits=from_logits),
        "precision": keras.metrics.Precision(thresholds=threshold, name="precision"),
        "recall": keras.metrics.Recall(thresholds=threshold, name="recall")
    }


class LogitsAccuracy(keras.metrics.Metric):
    def __init__(self, threshold=0.5, name: str = "accuracy"):
        super(LogitsAccuracy, self).__init__(name=name)
        self.threshold = threshold
        self.metric = keras.metrics.Accuracy()

    def update_state(self, y_true, logits, sample_weight=None):
        y_pred = tf.where(logits >= self.threshold, 1.0, 0.0)
        y_pred = tf.cast(y_pred, dtype=y_true.dtype)
        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()


class LogitsF1(keras.metrics.Metric):
    def __init__(self, from_logits=False, name: str = "f1"):
        super(LogitsF1, self).__init__(name=name)
        self.from_logits = from_logits
        self.metric = keras.metrics.F1Score(average="micro", threshold=0.5)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)

        self.metric.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.metric.result()

    def reset_state(self):
        self.metric.reset_state()

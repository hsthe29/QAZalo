import tensorflow as tf
from tensorflow import keras
from bert_utils import load_pretrained_bert, make_classifier


class BERT_QA(keras.Model):
    def __init__(self, num_classes, use_pooler=False, dropout_rate=0.1, hidden_units=768):
        super(BERT_QA, self).__init__()
        self.num_classes = num_classes

        self.bert = load_pretrained_bert()
        self.use_pooler = use_pooler
        self.classifier = make_classifier(num_classes, use_pooler, dropout_rate, hidden_units=hidden_units)

        # self.weight_class = config.weight_class

    # just need feed input ids and attention mask not need head mask or end position ...
    def __compute(self, input_ids, attention_mask, training):
        features = self.bert(input_ids, attention_mask=attention_mask)

        if self.use_pooler:
            pooler_output = features['pooler_output']
            output = self.classifier(pooler_output, training=training)
        else:
            all_hidden_states = features['hidden_states'][-4:]
            hidden_states = tf.stack(all_hidden_states, axis=1)

            hidden_states = hidden_states[:, :, 0, :]
            shape = tf.shape(hidden_states)
            cls_outputs = tf.reshape(hidden_states, [shape[0], shape[1] * shape[2]])

            output = self.classifier(cls_outputs, training=training)
        return output

    def call(self, inputs, training):
        input_ids, attention_mask = inputs
        logits = self.__compute(input_ids, attention_mask, training)
        return logits

    def train_loss(self, inputs, label, training):
        y_true = label
        input_ids, attention_mask = inputs
        logits = self.__compute(input_ids, attention_mask, training)

        # class_weights = tf.convert_to_tensor(self.weight_class)
        loss = tf.losses.categorical_crossentropy(y_true, logits, from_logits=True)  # , weight=class_weights)

        predict_value = tf.math.reduce_max(logits, axis=1)[1]
        list_predict = predict_value.numpy().tolist()
        list_target = y_true.numpy().tolist()

        return loss, list_predict, list_target

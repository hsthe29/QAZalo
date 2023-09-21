import tensorflow as tf
from qazalo.model import load_tokenizer
import pandas as pd

class DataExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 question,
                 doc,
                 is_has_answer=None):
        self.question = question
        self.doc = doc
        self.is_has_answer = is_has_answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "{"
        s += f"question: {self.question} | "
        s += f"doc: {self.doc} | "
        s += f", is_has_answer: {self.is_has_answer}" + "}"
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 segment_ids,
                 is_has_answer=None):
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_has_answer = is_has_answer


def read_data_from_file(input_path, test=False):
    with open(input_path, "r", encoding='utf-8') as rf:
        all_lines = rf.readlines()

    if test:
        # column_names = ["TestID", "Order", "Question", "Doc"]
        questions = []
        docs = []
        test_id = []

        for line in all_lines:
            parts = line.split("\t")
            questions.append(parts[2].strip())
            docs.append(parts[3].strip())
            test_id.append(parts[0].strip() + "\t" + parts[1].strip())

        data = {
            "test": True,
            "test_id": test_id,
            "question": questions,
            "doc": docs
        }
    else:
        # column_names = ["Question", "Doc", "HasAnswer"]
        questions = []
        docs = []
        has_answer = []

        for line in all_lines:
            parts = line.split("\t")
            questions.append(parts[0].strip())
            docs.append(parts[1].strip())
            has_answer.append(1 if parts[2].strip() == "true" else 0)

        data = {
            "test": False,
            "question": questions,
            "doc": docs,
            "label": has_answer
        }

    return data


def apply_processing(data, tokenizer, max_seq_length, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    questions = data["question"]
    docs = data["doc"]
    indices = range(len(questions))
    max_doc_length = max_seq_length - max_query_length - 3
    if data['test']:
        encoded_data = {
            "test_id": data["test_id"],
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": []
        }
    else:
        encoded_data = {
            "input_ids": [],
            "token_type_ids": [],
            "attention_mask": [],
            "label": data["label"]
        }
    for i in indices:
        query_tokens = tokenizer.tokenize(questions[i])
        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[:max_query_length]

        doc_tokens = tokenizer.tokenize(docs[i])
        if len(doc_tokens) > max_doc_length:
            doc_tokens = doc_tokens[:max_doc_length]

        query = tokenizer.convert_tokens_to_string(query_tokens)
        doc = tokenizer.convert_tokens_to_string(doc_tokens)

        encoded_seq = tokenizer(query, doc)

        for key, val in encoded_seq.items():
            encoded_data[key].append(val)

    return encoded_data


def make_dataset(path_input_data,
                 max_seq_length,
                 max_query_length,
                 batch_size,
                 mode="train"):
    tokenizer = load_tokenizer()

    if mode == "train" or mode == "eval":
        raw_data = read_data_from_file(path_input_data, test=False)

        encoded_data = apply_processing(raw_data, tokenizer, max_seq_length, max_query_length)

        input_ids = tf.ragged.constant(encoded_data['input_ids'])
        token_type_ids = tf.ragged.constant(encoded_data['token_type_ids'])
        attention_mask = tf.ragged.constant(encoded_data['attention_mask'])
        label = encoded_data['label']
        buffer_size = len(label)
        dataset = (tf.data.Dataset
                   .from_tensor_slices((input_ids, token_type_ids, attention_mask, label))
                   .shuffle(buffer_size=buffer_size)
                   .batch(batch_size=batch_size)
                   .map(lambda x, y, z, t: (
                    {
                        "input_ids": x.to_tensor(),
                        "token_type_ids": y.to_tensor(),
                        "attention_mask": z.to_tensor()
                    }, tf.cast(t, dtype=tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
                   .prefetch(buffer_size=tf.data.AUTOTUNE))
        return dataset
    elif mode == "test":
        raw_data = read_data_from_file(path_input_data, test=True)

        encoded_data = apply_processing(raw_data, tokenizer, max_seq_length, max_query_length)

        input_ids = tf.ragged.constant(encoded_data['input_ids'])
        token_type_ids = tf.ragged.constant(encoded_data['token_type_ids'])
        attention_mask = tf.ragged.constant(encoded_data['attention_mask'])
        test_id = encoded_data['test_id']

        dataset = (tf.data.Dataset
                   .from_tensor_slices((input_ids, token_type_ids, attention_mask))
                   .batch(batch_size=batch_size)
                   .map(lambda x, y, z: (
                    {
                        "input_ids": x.to_tensor(),
                        "token_type_ids": y.to_tensor(),
                        "attention_mask": z.to_tensor()
                    }), num_parallel_calls=tf.data.AUTOTUNE)
                   .prefetch(buffer_size=tf.data.AUTOTUNE))
        return dataset, test_id

    else:
        raise ValueError(f"Mode: {mode} not in supported list: ['train', 'eval', 'test']")


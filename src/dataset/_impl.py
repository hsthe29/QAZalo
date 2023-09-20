import tensorflow as tf

from model import load_tokenizer


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


def read_data_from_file(input_path, label=True):
    with open(input_path, "r", encoding='utf-8') as rf:
        questions = []
        docs = []
        labels = None
        if label:
            labels = []

        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")

            if label:
                if arr_e_line[2] == "true":
                    labels.append(1)
                else:
                    labels.append(0)

            question = arr_e_line[0].strip()
            doc = arr_e_line[1].strip()
            questions.append(question)
            docs.append(doc)

    examples = {
        "size": len(questions),
        "questions": questions,
        "docs": docs,
        "labels": labels
    }

    return examples


def apply_processing(data, tokenizer, max_seq_length, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    questions = data["questions"]
    docs = data["docs"]
    indices = range(len(questions))
    max_doc_length = max_seq_length - max_query_length - 3
    encoded_data = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'labels': data['labels']
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
                 max_seq_length=256,
                 max_query_length=64,
                 batch_size=16,
                 training=True):
    tokenizer = load_tokenizer()
    raw_data = read_data_from_file(path_input_data)

    encoded_data = apply_processing(raw_data, tokenizer, max_seq_length, max_query_length)
    buffer_size = raw_data["size"]
    input_ids = tf.ragged.constant(encoded_data['input_ids'])
    token_type_ids = tf.ragged.constant(encoded_data['token_type_ids'])
    attention_mask = tf.ragged.constant(encoded_data['attention_mask'])
    labels = encoded_data['labels']

    if training:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((input_ids, token_type_ids, attention_mask, labels))
                   .shuffle(buffer_size=buffer_size)
                   .batch(batch_size=batch_size)
                   .map(lambda x, y, z, t: (
                    {
                        "input_ids": x.to_tensor(),
                        "token_type_ids": y.to_tensor(),
                        "attention_mask": z.to_tensor()
                    }, tf.cast(t, dtype=tf.float32)), num_parallel_calls=tf.data.AUTOTUNE)
                   .prefetch(buffer_size=tf.data.AUTOTUNE))
    else:
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

    return dataset

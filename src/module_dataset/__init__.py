import tensorflow as tf


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 question_tokens,
                 doc_tokens,
                 is_has_answer=None):
        self.question_tokens = question_tokens
        self.doc_tokens = doc_tokens
        self.is_has_answer = is_has_answer

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = "{"
        s += f"question: {' '.join(self.question_tokens)} | "
        s += f"doc: {' '.join(self.doc_tokens)} | "
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


def read_examples_from_file(input_path, training=True):
    with open(input_path, "r", encoding='utf-8') as rf:
        examples = []
        for e_line in rf.readlines():
            e_line = e_line.replace("\n", "")
            arr_e_line = e_line.split("\t")

            if training:
                if arr_e_line[2] == "true":
                    is_has_answer = 1
                else:
                    is_has_answer = 0
            else:
                is_has_answer = None

            question_tokens = arr_e_line[0].casefold().strip().split(" ")
            doc_tokens = arr_e_line[1].casefold().strip().split(" ")

            example = SquadExample(question_tokens=question_tokens,
                                   doc_tokens=doc_tokens,
                                   is_has_answer=is_has_answer)
            examples.append(example)

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (example_index, example) in enumerate(examples):
        max_doc_length = max_seq_length - max_query_length
        query_ids = tokenizer.encode(example.question_tokens)
        if len(query_ids) > max_query_length:
            query_ids = query_ids[:max_query_length - 1] + [query_ids[-1]]

        doc_ids = tokenizer.encode(example.doc_tokens)
        if len(doc_ids) > max_doc_length:
            doc_ids = doc_ids[:max_doc_length - 1] + [doc_ids[-1]]

        input_ids = query_ids + doc_ids

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)
        l = len(input_ids)

        # Zero-pad up to the sequence length.
        if l < max_seq_length:
            input_ids.extend([0] * (max_seq_length - l))
            attention_mask.extend([0] * (max_seq_length - l))

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length

        features.append((input_ids, attention_mask, example.is_has_answer))

    return features


def make_dataset(path_input_data,
                 tokenizer,
                 max_seq_length=256,
                 max_query_length=64,
                 batch_size=16,
                 is_training=True):
    examples = read_examples_from_file(path_input_data)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length)
    buffer_size = len(features)

    # Convert to Tensors and build dataset
    all_input_ids = []
    all_input_mask = []
    for _, tp in enumerate(features):
        all_input_ids.append(tp[0])
        all_input_mask.append(tp[1])

    if is_training:
        all_label = []
        for _, tp in enumerate(features):
            all_label.append(tp[-1])

        dataset = (tf.data.Dataset
                   .from_tensor_slices((all_input_ids, all_input_mask, all_label))
                   .shuffle(buffer_size=buffer_size)
                   .batch(batch_size=batch_size)
                   .prefetch(buffer_size=tf.data.AUTOTUNE))
    else:
        dataset = (tf.data.Dataset
                   .from_tensor_slices((all_input_ids, all_input_mask))
                   .shuffle(buffer_size=buffer_size)
                   .batch(batch_size=batch_size)
                   .prefetch(buffer_size=tf.data.AUTOTUNE))

    return dataset


def eda_dataset(path_input_data,
                tokenizer,
                max_seq_length=256,
                max_query_length=64):
    examples = read_examples_from_file(path_input_data)

    features = convert_examples_to_features(examples, tokenizer, max_seq_length, max_query_length)
    buffer_size = len(features)

    # Convert to Tensors and build dataset
    all_input_ids = []
    all_input_mask = []
    for _, tp in enumerate(features):
        all_input_ids.append(tp[0])
        all_input_mask.append(tp[1])

    all_label = []
    for _, tp in enumerate(features):
        all_label.append(tp[2])

    all_len = len(all_label)
    true_class = sum(all_label)
    print("true: ", true_class)
    print("false: ", all_len - true_class)

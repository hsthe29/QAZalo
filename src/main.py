from bert_model.bert import BERT_QA
from typing import NamedTuple

class Config(NamedTuple):
    num_labels: int
    use_pooler: bool

if __name__ == "__main__":

    model = BERT_QA()
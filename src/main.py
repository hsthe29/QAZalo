from module_dataset import eda_dataset
from bert_model.bert_utils import load_tokenizer
from bert_model.bert import BERT_QA
from utils import train

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = BERT_QA(num_classes=2)
    eda_dataset("../data/train/train_test_origin_1k_dev.csv",
                                tokenizer,
                                max_seq_length=256,
                                max_query_length=50)
    eda_dataset("../data/train/val_origin_1k.csv",
                              tokenizer,
                              max_seq_length=256,
                              max_query_length=50)



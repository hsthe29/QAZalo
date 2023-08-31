from module_dataset import make_dataset
from bert_model.bert_utils import load_tokenizer
from bert_model.bert import BERT_QA
from utils import train


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    model = BERT_QA(num_classes=2)
    train_dataset = make_dataset("../data/train/train_demo.csv",
                                 tokenizer,
                                 max_seq_length=256,
                                 max_query_length=50,
                                 batch_size=16)
    val_dataset = make_dataset("../data/train/val_demo.csv",
                               tokenizer,
                               max_seq_length=256,
                               max_query_length=50,
                               batch_size=16)

    history = train(model, 10, train_dataset, val_dataset=val_dataset, save_per_epochs=5, fix_epochs=4)
    print(history)

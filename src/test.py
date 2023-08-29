import tensorflow as tf
from utils import load_pretrained_bert_qa

if __name__ == "__main__":
    model = load_pretrained_bert_qa("../save/weights/bert-qa")
    print(model.weights)

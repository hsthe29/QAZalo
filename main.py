from dataset import make_dataset
from src.model import BertClassifier
from src.utils import train, evaluate, predict
from src.parser import Parser

parser = Parser()
parser.DEFINE_integer(
    "num_classes", 2
)
parser.DEFINE_integer(
    "EPOCHS", 10,
    "Number of training epochs"
)
parser.DEFINE_string(
    "mode", "train",
    "Whether to run training."
)
parser.DEFINE_float(
    "init-lr", 2e-5,
    "Initial learning rate for Adam Weight Decay optimizer")
parser.DEFINE_float(
    "warmup-proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training."
)
parser.DEFINE_float("weight-decay", 0.001)
parser.DEFINE_bool(
    "from-logits", False,
    "Output of model 'call'. If from-logits=True, return logits else return probabilities"
)
parser.DEFINE_integer("batch-size", 16)
parser.DEFINE_bool(
    "use-pooler", False,
    "Whether to use pooler output for classification"
)
parser.DEFINE_bool(
    "from-scratch", True,
    "Whether to train model from scratch"
)
parser.DEFINE_string(
    "pretrained-name", None,
    "Saved model for loading"
)
parser.DEFINE_string("log-dir", "logs/")
parser.DEFINE_string("save-dir", "checkpoint/")
parser.DEFINE_string(
    "train-input", None,
    "Training data directory"
)
parser.DEFINE_string(
    "validation-input", None,
    "Validation data directory"
)
parser.DEFINE_string(
    "test-input", None,
    "Test data directory"
)
parser.DEFINE_string(
    "output-dir", None,
    "Output directory to store prediction result of test data"
)

flags = parser.parse()

if __name__ == "__main__":

    if flags.mode == "train":
        if flags.from_scratch:
            model = BertClassifier(num_classes=flags.num_classes, use_pooler=flags.use_pooler)
        else:
            pretrained_name = f"zqa-{flags.num_classes}-P{int(flags.use_pooler)}-L{int(flags.from_logits)}"
            model = BertClassifier.from_pretrained(pretrained_name)

        if flags.train_input is None:
            raise ValueError("There is no data input")
        train_ds = make_dataset(flags.train_input,
                                batch_size=flags.batch_size)
        val_ds = None
        if flags.validation_input:
            val_ds = make_dataset(flags.validation_input,
                                  batch_size=flags.batch_size)
        train(model, (train_ds, val_ds), flags=flags)

    elif flags.mode == "eval":
        model = BertClassifier.from_pretrained(flags.pretrained_name)
        if flags.test_input is None:
            raise ValueError("There is no test input")
        test_dataset = make_dataset(flags.test_input,
                                    batch_size=flags.batch_size)

        evaluate(model, test_dataset, flags=flags)

    elif flags.mode == "test":
        model = BertClassifier.from_pretrained(flags.pretrained_name)
        if flags.test_input is None:
            raise ValueError("There is no test input")
        test_dataset = make_dataset(flags.test_input,
                                    batch_size=flags.batch_size,
                                    training=False)

        predict(model, test_dataset, flags=flags)
    else:
        raise ValueError(f"Mode: {flags.mode} not in supported modes!")

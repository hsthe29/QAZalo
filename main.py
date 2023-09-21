from qazalo.dataset import make_dataset
from qazalo.utils import train, evaluate, predict
from qazalo.parser import Parser
import tensorflow as tf

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
parser.DEFINE_string(
    "monitor", "val_f1",
    "Monitor for checkpoint"
)
parser.DEFINE_integer(
    "max-seq-length", 512
)
parser.DEFINE_integer(
    "max-query-length", 64
)
parser.DEFINE_bool(
    "from-scratch", True,
    "Whether to train model from scratch"
)
parser.DEFINE_string(
    "pretrained-name", None,
    "Saved model for loading"
)
parser.DEFINE_string(
    "update-freq", "epoch"
)
parser.DEFINE_bool(
    "use-tpu", False
)
parser.DEFINE_string(
    "tpu-name", ""
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
    "output", None,
    "Output file to store prediction result of test data"
)


def detect_tpu():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=flags.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All TPUs: ", tf.config.list_logical_devices('TPU'))

    except ValueError:
        resolver = None
        print('No TPU detected')

    return resolver


if __name__ == "__main__":
    flags = parser.parse()
    strategy = None
    # use tpu
    if flags.use_tpu:
        tpu_resolver = detect_tpu()
        if tpu_resolver:
            strategy = tf.distribute.TPUStrategy(tpu_resolver)

    if flags.mode == "train":
        if flags.train_input is None:
            raise ValueError("There is no train input")
        train_ds = make_dataset(flags.train_input,
                                max_seq_length=flags.max_seq_length,
                                max_query_length=flags.max_query_length,
                                batch_size=flags.batch_size)
        val_ds = None
        if flags.validation_input:
            val_ds = make_dataset(flags.validation_input,
                                  max_seq_length=flags.max_seq_length,
                                  max_query_length=flags.max_query_length,
                                  batch_size=flags.batch_size)
        train((train_ds, val_ds), flags, strategy=strategy)

    elif flags.mode == "eval":
        if flags.validation_input is None:
            raise ValueError("There is no validation input")
        val_ds = make_dataset(flags.validation_input,
                              max_seq_length=flags.max_seq_length,
                              max_query_length=flags.max_query_length,
                              batch_size=flags.batch_size,
                              mode="eval")

        evaluate(val_ds, flags, strategy=strategy)

    elif flags.mode == "test":
        if flags.test_input is None:
            raise ValueError("There is no test input")
        test_ds, test_id = make_dataset(flags.test_input,
                                        batch_size=flags.batch_size,
                                        max_seq_length=flags.max_seq_length,
                                        max_query_length=flags.max_query_length,
                                        mode="test")

        predict(test_ds, test_id, flags=flags, strategy=strategy)
    else:
        raise ValueError(f"Mode: {flags.mode} not in supported modes!")

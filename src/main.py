import tensorflow as tf
import sys

f = tf.constant([[1, 5],
                 [4, 3],
                 [3, 2]])

tf.print(f)

tensor = tf.range(10)
tf.print(tensor, output_stream=sys.stdout)
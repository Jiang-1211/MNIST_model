import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

np.set_printoptions(threshold=np.inf)

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "build", "mnist_cnn_model_modified.h5")
model = tf.keras.models.load_model(MODEL_PATH)


def quantization(x, interval, lower, upper):
    x_clip = np.clip(x, lower, upper)

    idx = np.round((x_clip - lower) / interval)

    q = lower + idx * interval

    return q.astype(x.dtype)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test / 255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

index = 1
upper = 0.5
lower = -0.4375
bit = 4
interval = (upper - (-0.5)) / pow(2.0, bit)

intermediate_layer = tf.keras.models.Model(
    inputs=model.layers[0].input, outputs=model.layers[0].output
)
layer_output = intermediate_layer.predict(x_test, batch_size=256, verbose=2)

layer_output = quantization(layer_output, interval, lower, upper)


while index < len(model.layers):
    intermediate_layer = tf.keras.models.Model(
        inputs=model.layers[index].input, outputs=model.layers[index].output
    )
    layer_output = intermediate_layer.predict(layer_output, batch_size=256, verbose=2)
    layer_output = quantization(layer_output, interval, lower, upper)
    index = index + 1

probs = tf.nn.softmax(layer_output, axis=-1).numpy()
y_pred = probs.argmax(axis=-1)

y_true = y_test
if isinstance(y_true, tf.Tensor):
    y_true = y_true.numpy()
if y_true.ndim == 2 and y_true.shape[-1] == 10:
    y_true = y_true.argmax(axis=-1)
else:
    y_true = y_true.astype(np.int64)

accuracy = (y_pred == y_true).mean()
print(f"Accuracy: {accuracy:.4f}")

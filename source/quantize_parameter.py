import tensorflow as tf
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

BASE_DIR = os.path.dirname(__file__)
ORIGINAL_MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_model.h5")
MODIFIED_MODEL_PATH = os.path.join(BASE_DIR, "mnist_cnn_model_modified.h5")


def quantization(x, interval, lower, upper):
    x_clip = np.clip(x, lower, upper)

    idx = np.round((x_clip - lower) / interval)

    q = lower + idx * interval

    return q.astype(x.dtype)


upper = 0.5
lower = -0.4375
bit = 4
interval = (upper - (-0.5)) / pow(2.0, bit)

model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

weights = model.get_weights()

for i in range(len(weights)):
    if len(weights[i]) > 0:
        for j in range(len(weights[i])):
            weights[i][j] = quantization(weights[i][j], interval, lower, upper)
    else:
        weights[i] = weights[i]

model.set_weights(weights)
model.save(MODIFIED_MODEL_PATH)

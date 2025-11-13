import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np
import os

np.set_printoptions(threshold=np.inf)

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "build")

ORIGINAL_MODEL_PATH = os.path.join(MODEL_PATH, "mnist_cnn_model.h5")
ORIGINAL_PARAMETER = os.path.join(MODEL_PATH, "model_parameters.txt")
original_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)

MODIFIED_MODEL_PATH = os.path.join(MODEL_PATH, "mnist_cnn_model_modified.h5")
MODIFIED_PARAMETER = os.path.join(MODEL_PATH, "model_modified_parameters.txt")
modified_model = tf.keras.models.load_model(MODIFIED_MODEL_PATH)

i = 0
with open(ORIGINAL_PARAMETER, "w", encoding="utf-8") as f:
    for layer in original_model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            f.write(f"Layer index: {i}\n")
            f.write(f"Layer: {layer.name}\n")
            f.write(f"Weight shape: {weights[0].shape}\n")
            f.write(str(weights[0]) + "\n\n")
            f.write(f"Bias shape : {weights[1].shape}\n")
            f.write(str(weights[1]) + "\n\n")
        i = i + 1

i = 0
with open(MODIFIED_PARAMETER, "w", encoding="utf-8") as f:
    for layer in modified_model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            f.write(f"Layer index: {i}\n")
            f.write(f"Layer: {layer.name}\n")
            f.write(f"Weight shape: {weights[0].shape}\n")
            f.write(str(weights[0]) + "\n\n")
            f.write(f"Bias shape : {weights[1].shape}\n")
            f.write(str(weights[1]) + "\n\n")
        i = i + 1

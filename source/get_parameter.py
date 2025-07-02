import tensorflow as tf
from tensorflow.keras.utils import plot_model
import numpy as np

np.set_printoptions(threshold=np.inf)

original_model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
)
original_output_file = "C:/VSCode_Projects/ML/model/build/model_parameters.txt"

modified_model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"
)
modified_output_file = "C:/VSCode_Projects/ML/model/build/model_modified_parameters.txt"

i = 0
with open(original_output_file, "w", encoding="utf-8") as f:
    for layer in original_model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            f.write(f"Layer index: {i}\n")
            f.write(f"Layer: {layer.name}\n")
            f.write(f"Weight shape: {weights[0].shape}\n")
            f.write(str(weights[0]) + "\n\n")
            f.write(f"Bias shape : {weights[1].shape}\n")
            f.write(str(weights[1]) + "\n\n")
            # f.write("-" * 50 + "\n")
        i = i + 1
i = 0
with open(modified_output_file, "w", encoding="utf-8") as f:
    for layer in modified_model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:
            f.write(f"Layer index: {i}\n")
            f.write(f"Layer: {layer.name}\n")
            f.write(f"Weight shape: {weights[0].shape}\n")
            f.write(str(weights[0]) + "\n\n")
            f.write(f"Bias shape : {weights[1].shape}\n")
            f.write(str(weights[1]) + "\n\n")
            # f.write("-" * 50 + "\n")
        i = i + 1

print(
    "------------------------------------------------------------------------------------------------------------------------"
)
print(f"Original parameters are saved to {original_output_file}")
print(f"Modified parameters are saved to {modified_output_file}")
print(
    "------------------------------------------------------------------------------------------------------------------------"
)

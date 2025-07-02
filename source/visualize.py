import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

original_model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
)

modified_model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"
)

original_conv_weights = original_model.layers[0].get_weights()[0]
modified_conv_weights = modified_model.layers[0].get_weights()[0]

fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for i in range(16):
    ax = axes[i // 8, i % 8]
    ax.imshow(original_conv_weights[:, :, 0, i], cmap="gray")
    ax.axis("off")
    ax.set_title(f"Filter {i+1}")
    ax.imshow(modified_conv_weights[:, :, 0, i], cmap="gray")
    ax.set_title(f"Filter {i+1}")

plt.show()

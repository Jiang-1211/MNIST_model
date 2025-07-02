import tensorflow as tf
import numpy as np
import re

np.set_printoptions(threshold=np.inf)


def quantization(data):
    data = np.array(data)

    result = np.where(
        data >= 2,
        2,
        np.where(data <= -2, -2, np.round(data * pow(2.0, 8.0)) * pow(2.0, -8.0)),
    )

    return result


original_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
new_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"

model = tf.keras.models.load_model(original_model_path)

weights = model.get_weights()

for i in range(len(weights)):
    if len(weights[i]) > 0:
        for j in range(len(weights[i])):
            weights[i][j] = quantization(weights[i][j])
    else:
        weights[i] = weights[i]

model.set_weights(weights)
model.save(new_model_path)

print(
    "------------------------------------------------------------------------------------------------------------------------"
)
print("Complete modifying parameters")
print(
    "------------------------------------------------------------------------------------------------------------------------"
)

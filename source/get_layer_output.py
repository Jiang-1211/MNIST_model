import numpy as np
import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"

model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
)

inputs = tf.keras.layers.Input(shape=(28,))
# x = Dense(32, activation="relu", name="dense_1")(inputs)  # Naming the layer explicitly

layer_output = tf.keras.models.Model(
    inputs=model.inputs, outputs=model.get_layer("conv2d").output
)

dummy = np.random.rand(5, 28, 28, 1).astype("float32")
out = layer_output.predict(dummy)
print(out)

model.summary()

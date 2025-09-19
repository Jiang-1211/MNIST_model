import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical, plot_model
import os

BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "build")

ORIGINAL_MODEL_PATH = os.path.join(MODEL_PATH, "mnist_cnn_model.h5")
original_model = tf.keras.models.load_model(ORIGINAL_MODEL_PATH)
MODIFIED_MODEL_PATH = os.path.join(MODEL_PATH, "mnist_cnn_model_modified.h5")
modified_model = tf.keras.models.load_model(MODIFIED_MODEL_PATH)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

y_test = to_categorical(y_test, num_classes=10)

loss_original, accuracy_original = original_model.evaluate(x_test, y_test, verbose=2)
loss_modified, accuracy_modified = modified_model.evaluate(x_test, y_test, verbose=2)


print(50 * "-")
print(f"          |  Accuracy  |  Loss ")
print(f"Original  |  {accuracy_original:.4f}    |  {loss_original:.4f}")
print(f"Modified  |  {accuracy_modified:.4f}    |  {loss_modified:.4f}")
print(50 * "-")

plot_model(
    original_model,
    to_file=os.path.join(MODEL_PATH, "model_structure.png"),
    show_shapes=True,
)

original_model.summary()

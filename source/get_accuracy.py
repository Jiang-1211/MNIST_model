import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

y_test = to_categorical(y_test, num_classes=10)

original_model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
)
modified_model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"
)

loss_original, accuracy_original = original_model.evaluate(x_test, y_test, verbose=2)
loss_modified, accuracy_modified = modified_model.evaluate(x_test, y_test, verbose=2)


print(
    "------------------------------------------------------------------------------------------------------------------------"
)
print(f"          |  Accuracy  |  Loss ")
print(f"Original  |  {accuracy_original:.4f}    |  {loss_original:.4f}")
print(f"Modified  |  {accuracy_modified:.4f}    |  {loss_modified:.4f}")
print(
    "------------------------------------------------------------------------------------------------------------------------"
)

original_model.summary()

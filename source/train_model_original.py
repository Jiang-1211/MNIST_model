import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
IMAGE_CHENNL = 1
NUM_CLASS = 10
EPOCH = 16
BATCH_SIZE = 8
MODEL_PATH = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_train = x_train / 255.0
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_test = x_test / 255.0
y_train = tf.one_hot(y_train.astype(np.int32), depth=10)
y_test = tf.one_hot(y_test.astype(np.int32), depth=10)

model = tf.keras.models.Sequential()

model.add(
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=3,
        strides=(1, 1),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        activation="relu",
    )
)
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=(1, 1),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        activation="relu",
    )
)
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

model.add(
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=3,
        strides=(1, 1),
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        activation="relu",
    )
)
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3)))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(
    tf.keras.layers.Dense(
        units=16,
        activity_regularizer=tf.keras.regularizers.l2(1e-4),
        activation="relu",
    )
)
# model.add(tf.keras.layers.BatchNormalization())
# model.add(tf.keras.layers.Activation("relu"))

model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(
    tf.keras.layers.Dense(
        units=NUM_CLASS,
        activity_regularizer=tf.keras.regularizers.l2(1e-4),
        activation="softmax",
    )
)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# history = model.fit(
#     x_train,
#     y_train,
#     batch_size=BATCH_SIZE,
#     epochs=EPOCH,
#     validation_data=(x_test, y_test),
# )

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCH,
    validation_data=(x_test, y_test),
)

model.save(MODEL_PATH)

model.summary()

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

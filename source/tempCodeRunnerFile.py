tf.keras.layers.Input(shape=(10,))
x = Dense(32, activation="relu", name="dense_1")(inputs)  # Naming the layer explicitly

layer_output = tf.keras.models.Model(
    inputs=model.input, outputs=model.get_layer("conv2d").output
)
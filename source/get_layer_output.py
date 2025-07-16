import numpy as np
import tensorflow as tf
from tensorflow import keras

np.set_printoptions(threshold=np.inf)


def quantization(x, interval=2**-6, lower=-2.0, upper=2.0):
    """
    x        : np.ndarray，任意形狀 (…)
    interval : 量化步長；預設 2**-6 = 1/64
    lower    : 下限 (飽和/clamp)
    upper    : 上限
    return   : 與 x 同形狀的量化結果 (np.ndarray)
    """
    # 1) 先用 clip 把數值限制在 [lower, upper]
    x_clip = np.clip(x, lower, upper)

    # 2) 位移到 [0, upper-lower] 再除以 interval → 得到 bucket index
    #    再用 floor 對齊到左端點
    #    idx = floor((x_clip - lower) / interval)
    idx = np.floor((x_clip - lower) / interval)

    # 3) 還原到實際量化值：lower + idx * interval
    q = lower + idx * interval

    return q.astype(x.dtype)  # 保留原 dtype


model = tf.keras.models.load_model(
    "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"
)

# inputs = tf.keras.layers.Input(shape=(28,))
# x = Dense(32, activation="relu", name="dense_1")(inputs)  # Naming the layer explicitly

layer_output = tf.keras.models.Model(
    inputs=model.inputs, outputs=model.get_layer("dense_1").output
)

dummy = np.random.rand(5, 28, 28, 1).astype("float32")
dummy = quantization(dummy)
out = layer_output.predict(dummy)
print(out)
print("==============================")
out = quantization(out)
print(out)

model.summary()

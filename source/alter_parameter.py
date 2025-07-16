import tensorflow as tf
import numpy as np
import re

np.set_printoptions(threshold=np.inf)


def quantization(x, interval, lower, upper):
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


original_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
new_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"

upper = 2.0
lower = -2.0
bit = 8
interval = (upper - lower) / pow(2.0, bit)

model = tf.keras.models.load_model(original_model_path)

weights = model.get_weights()

for i in range(len(weights)):
    if len(weights[i]) > 0:
        for j in range(len(weights[i])):
            weights[i][j] = quantization(weights[i][j], interval, lower, upper)
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

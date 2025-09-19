# 1) 你的自訂卷積層：權重介面與 Conv2D 對齊（kernel, bias），以便承接舊權重
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, activations, Model, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import os

np.set_printoptions(threshold=np.inf)
# ✅ 無論你是 `import keras` 或 `from tensorflow import keras`，都能相容


# --- TF 版量化 ---
def quantization_tf(x, interval, lower, upper):
    x_clip = tf.clip_by_value(x, lower, upper)
    idx = tf.round((x_clip - lower) / interval)
    q = lower + idx * interval
    return tf.cast(q, x.dtype)


def _norm2(x, name):
    """把 kernel_size/strides 變成 2-tuple，行為類似 Keras 的 normalize_tuple。"""
    if isinstance(x, (list, tuple)):
        if len(x) != 2:
            raise ValueError(
                f"`{name}` must be a int or a tuple/list of length 2, got: {x}"
            )
        return (int(x[0]), int(x[1]))
    return (int(x), int(x))


class CustomQuantConv2D(layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="valid",
        use_bias=True,
        activation=None,
        interval=0.1,
        lower=-1.0,
        upper=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = _norm2(kernel_size, "kernel_size")
        self.strides = _norm2(strides, "strides")
        # Keras 3 Conv2D 用 'same'/'valid'；tf.nn.conv2d 要 'SAME'/'VALID'
        self.padding = str(padding).upper()
        if self.padding not in ("SAME", "VALID"):
            raise ValueError(f"padding must be 'same' or 'valid', got {padding}")
        self.use_bias = bool(use_bias)
        self.activation = activations.get(activation)
        self.interval = float(interval)
        self.lower = float(lower)
        self.upper = float(upper)

    def build(self, input_shape):
        kh, kw = self.kernel_size
        in_ch = int(input_shape[-1])
        if in_ch is None:
            # 通常 CNN 輸入通道在 build 時應該是已知的；保險起見拋錯
            raise ValueError("Input channel dimension must be known at build time.")
        self.kernel = self.add_weight(
            name="kernel",
            shape=(kh, kw, in_ch, self.filters),
            initializer="he_normal",
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.bias = None
        super().build(input_shape)

    def call(self, inputs):
        kh, kw = self.kernel_size

        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, kh, kw, 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            rates=[1, 1, 1, 1],
            padding=self.padding,  # 'SAME' 或 'VALID'
        )  # [B, H, W, kh*kw*in_ch]

        shp = tf.shape(patches)
        B, H, W = shp[0], shp[1], shp[2]
        in_ch = inputs.shape[-1]  # 靜態維度（用來 reshape）
        patches = tf.reshape(patches, [B, H, W, kh, kw, in_ch])  # [B,H,W,kh,kw,in_ch]

        # 元素乘法 → 逐乘積量化
        mul = (
            patches[..., None] * self.kernel[None, None, None, ...]
        )  # [B,H,W,kh,kw,in_ch,out_ch]
        mul_q = quantization_tf(mul, self.interval, self.lower, self.upper)

        # 加總
        out = tf.reduce_sum(mul_q, axis=[3, 4, 5])  # [B,H,W,out_ch]

        # 加 bias
        if self.use_bias:
            out = tf.nn.bias_add(out, self.bias, data_format="NHWC")

        # activation
        if self.activation is not None:
            out = self.activation(out)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding.lower(),
                "use_bias": self.use_bias,
                "activation": activations.serialize(self.activation),
                "interval": self.interval,
                "lower": self.lower,
                "upper": self.upper,
            }
        )
        return cfg


def replace_conv_with_quant(
    model,
    default_quant_params=dict(interval=0.1, lower=-2.0, upper=2.0),
    per_layer_quant: dict = None,
    sample_input_shape=None,
):
    """
    model: 已訓練好的 Keras 模型（Sequential 或 Functional）
    default_quant_params: 統一的量化參數
    per_layer_quant: 可選，{layer_name: {"interval":..., "lower":..., "upper":...}}
    sample_input_shape: 用來 build/trace 新模型的 dummy 形狀；若為 None，嘗試用 model.input_shape
    """
    per_layer_quant = per_layer_quant or {}

    def convert(layer):
        # 注意：Keras 3 Conv2D 是 keras.layers.Conv2D；這裡用名字避免跨後端差異
        if layer.__class__.__name__ == "Conv2D":
            cfg = layer.get_config()
            q = per_layer_quant.get(layer.name, default_quant_params)

            # 這個簡化版沒實作 dilation/groups；如有需要再擴充
            dil = cfg.get("dilation_rate", (1, 1))
            if isinstance(dil, int):
                dil = (dil, dil)
            if tuple(dil) != (1, 1):
                raise ValueError(
                    f"Layer {layer.name} uses dilation_rate={dil}, "
                    "current CustomQuantConv2D doesn't support dilation."
                )
            groups = int(cfg.get("groups", 1))
            if groups != 1:
                raise ValueError(
                    f"Layer {layer.name} uses groups={groups}, "
                    "current CustomQuantConv2D doesn't support groups."
                )

            return CustomQuantConv2D(
                filters=cfg["filters"],
                kernel_size=cfg["kernel_size"],
                strides=cfg["strides"],
                padding=cfg["padding"],
                use_bias=cfg["use_bias"],
                activation=cfg.get("activation"),
                interval=q.get("interval", 0.1),
                lower=q.get("lower", -2.0),
                upper=q.get("upper", 2.0),
                name=layer.name,
            )
        # 其他層維持不變
        return layer.__class__.from_config(layer.get_config())

    # 複製結構並置換
    new_model = models.clone_model(model, clone_function=convert)

    # 建立新模型的權重（一次 dummy call）
    if sample_input_shape is None:
        if getattr(model, "inputs", None) is not None and model.inputs:
            # 從舊模型的 input shape 抓
            # input_shape 格式通常為 (None, H, W, C)
            ish = model.input_shape
            if isinstance(ish, list):
                ish = ish[0]
            batch = 1 if ish[0] is None else ish[0]
            sample_input_shape = (batch,) + tuple(ish[1:])
        else:
            raise ValueError(
                "Please provide sample_input_shape (e.g., (1, 28, 28, 1))."
            )

    _ = new_model(tf.zeros(sample_input_shape))

    # 權重轉移（名稱對應）
    name_to_old = {l.name: l for l in model.layers}
    for l in new_model.layers:
        if l.name in name_to_old:
            old = name_to_old[l.name]
            try:
                l.set_weights(old.get_weights())
            except Exception:
                # 非同類型/形狀不符就略過（例如 BatchNorm 的 moving stats）
                pass

    return new_model


BASE_DIR = os.path.dirname(__file__)

MODEL_PATH = os.path.join(BASE_DIR, "..", "build", "mnist_cnn_model_modified.h5")
model = tf.keras.models.load_model(MODEL_PATH)
PARAMETER = os.path.join(BASE_DIR, "..", "build", "model_quan_layer_output.txt")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

y_test = to_categorical(y_test, num_classes=10)

upper = 0.5
lower = -0.4375
bit = 4
interval = (upper - (-0.5)) / pow(2.0, bit)

custom_model = replace_conv_with_quant(
    model,
    default_quant_params=dict(interval=interval, lower=lower, upper=upper),
    # per_layer_quant={"conv2d": {"interval": 0.02, "lower": -1.0, "upper": 1.0}}, # 可選：針對特定層名
    sample_input_shape=(1, 28, 28, 1),  # MNIST
)

custom_model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

loss, acc = custom_model.evaluate(x_test, y_test, verbose=2)


intermediate_layer = tf.keras.models.Model(
    inputs=custom_model.layers[0].input, outputs=custom_model.layers[0].output
)
layer_output = intermediate_layer.predict(x_test[0:1])
index = 1
with open(PARAMETER, "w", encoding="utf-8") as f:
    while index < len(custom_model.layers):
        intermediate_layer = tf.keras.models.Model(
            inputs=custom_model.layers[index].input,
            outputs=custom_model.layers[index].output,
        )
        layer_output = intermediate_layer.predict(layer_output)
        f.write(f"Layer index : {index}\n")
        f.write(str(layer_output))
        f.write("\n\n")
        index = index + 1

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, activations, backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt  # <-- (*** 新增 ***)
import seaborn as sns  # <-- (*** 新增 ***)
import pandas as pd  # <-- (*** 新增 ***)

np.set_printoptions(threshold=np.inf)


# --- TF 版量化 ---
def quantization_tf(x, interval, lower, upper):
    x_clip = tf.clip_by_value(x, lower, upper)
    idx = tf.round((x_clip - lower) / interval)
    q = lower + idx * interval
    return tf.cast(q, x.dtype)


def quantization_np(x, interval, lower, upper):
    x_clip = np.clip(x, lower, upper)
    idx = np.round((x_clip - lower) / interval)
    q = lower + idx * interval
    return q.astype(x.dtype)


def quantization_parameter(model, interval, lower, upper):
    weights = model.get_weights()

    for i in range(len(weights)):
        if len(weights[i]) > 0:
            for j in range(len(weights[i])):
                weights[i][j] = quantization_np(weights[i][j], interval, lower, upper)
        else:
            weights[i] = weights[i]

    model.set_weights(weights)

    return model


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
        # --- (*** 修正 ***) ---
        # 必須從 config 重新建立，才能正確複製層 (例如 Activation, MaxPooling)
        # return layer
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

    # 確保輸入形狀是具體的
    if any(s is None for s in sample_input_shape):
        raise ValueError(
            f"sample_input_shape must be concrete, but got {sample_input_shape}"
        )

    _ = new_model(tf.zeros(sample_input_shape))

    # 權重轉移（名稱對應）
    name_to_old = {l.name: l for l in model.layers}
    for l in new_model.layers:
        if l.name in name_to_old:
            old = name_to_old[l.name]
            # 只轉移有權重的層 (例如 Conv, Dense, BatchNorm)
            if old.get_weights():
                try:
                    l.set_weights(old.get_weights())
                except Exception as e:
                    # 非同類型/形狀不符就略過（例如 BatchNorm 的 moving stats）
                    print(
                        f"Warning: Could not set weights for layer {l.name}. Error: {e}"
                    )
                    pass

    return new_model


# --- (假設的路徑，請自行修改) ---
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "build", "mnist_cnn_model.h5")
PARAMETER = os.path.join(BASE_DIR, "..", "build", "model_quan_layer_output.txt")

# --- 建立一個範例的 pre-trained model ---
# (因為我沒有你的 .h5 檔案，這裡我建立一個)
if not os.path.exists(MODEL_PATH):
    print("Creating a dummy mnist_cnn_model_modified.h5...")
    base_model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, (3, 3), activation="relu", name="conv2d_1"),
            layers.MaxPooling2D((2, 2), name="maxpool_1"),
            layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_2"),
            layers.MaxPooling2D((2, 2), name="maxpool_2"),
            layers.Flatten(name="flatten"),
            layers.Dense(128, activation="relu", name="dense_1"),
            layers.Dense(10, activation="softmax", name="dense_output"),
        ],
        name="mnist_cnn_model",
    )
    base_model.save(MODEL_PATH)
# ------------------------------------

model = tf.keras.models.load_model(MODEL_PATH)
print("--- 原始模型架構(沒做量化) ---")
model.summary()

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

y_test = to_categorical(y_test, num_classes=10)

# --- (*** 新增 ***) 評估 32-bit 浮點 (Float32) 基準模型 ---
print("\n--- 評估 32-bit 浮點 (Float32) 基準模型 ---")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
loss_fp32, acc_fp32 = model.evaluate(x_test, y_test, verbose=2)
print(f"Float32 Baseline: Loss={loss_fp32:.4f}, Accuracy={acc_fp32:.4f}")
print("=" * 50)
# ---------------------------------------------------------

upper = 0.5
lower = -0.4375
bit_width = [4, 5, 6, 7, 8, 9, 10, 11, 12]

results = []  # <-- (*** 新增 ***) 建立一個 list 來儲存 (bit, acc)

for bit in bit_width:
    interval = pow(2.0, (-1) * bit)

    quan_para_model = quantization_parameter(model, interval, lower, upper)

    custom_model = replace_conv_with_quant(
        quan_para_model,
        default_quant_params=dict(interval=interval, lower=lower, upper=upper),
        sample_input_shape=(1, 28, 28, 1),  # MNIST
    )
    custom_model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    loss, acc = custom_model.evaluate(x_test, y_test, verbose=2)
    print(f"# of Bits = {bit}, Custom model acc: {acc:.4f}, loss: {loss:.4f}")
    results.append({"Bit Width": bit, "Accuracy": acc})  # <-- (*** 新增 ***) 儲存結果


# --- (*** 新增：繪製準確率 vs. 位元寬度折線圖 ***) ---
print("\n--- 正在產生準確率 vs. 位元寬度折線圖 ---")
try:
    # 1. 將 results 列表轉換為 DataFrame
    results_df = pd.DataFrame(results)

    # 2. 開始繪圖
    plt.figure(figsize=(10, 6))

    # 3. 繪製基準線 (Float32)
    plt.axhline(
        y=acc_fp32,
        color="r",
        linestyle="--",
        linewidth=1.0,
        label=f"Float32 Baseline ({acc_fp32:.4f})",
    )

    # 4. 繪製量化結果的折線圖
    sns.lineplot(
        data=results_df,
        x="Bit Width",
        y="Accuracy",
        marker="o",
        label="Quantized Model",
    )

    # 5. 在每個點上標註準確率數字
    for _, row in results_df.iterrows():
        plt.text(row["Bit Width"], row["Accuracy"] + 0.001, f"{row['Accuracy']:.4f}")

    # 6. 設定圖表標題和標籤
    plt.title("Model Accuracy vs. Quantization Bit Width")
    plt.xlabel("Bit Width")
    plt.ylabel("Accuracy")
    plt.xticks(bit_width)  # 確保 X 軸只顯示 2, 3, ..., 12
    plt.legend()
    plt.grid(True, linestyle="--")

    # 7. 儲存圖表
    plot_path = os.path.join(
        BASE_DIR, "..", "build", "quantization_accuracy_vs_bits.png"
    )
    plt.savefig(plot_path)
    plt.close()  # 關閉畫布，釋放記憶體
    print(f"--- 折線圖已儲存至 {plot_path} ---")

except Exception as e:
    print(f"繪圖時發生錯誤: {e}")
# -------------------------------------------------


# --- (*** 修正後的輸出迴圈 ***) ---
print(f"\n--- 正在儲存中間層輸出到 {PARAMETER} ---")

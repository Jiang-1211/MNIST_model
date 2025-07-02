import os
import zipfile
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import save_img
import tensorflow as tf
from PIL import Image

# 確保你已經加載了 MNIST 數據集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 預處理數據
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 數據增強設置
datagen = ImageDataGenerator(
    rotation_range=15,  # 隨機旋轉
    width_shift_range=0.1,  # 隨機水平平移
    height_shift_range=0.1,  # 隨機垂直平移
    shear_range=0.1,  # 隨機剪切變換
    zoom_range=0.1,  # 隨機縮放
    horizontal_flip=False,  # 停用隨機水平翻轉
    vertical_flip=False,  # 停用隨機垂直翻轉
    fill_mode="nearest",  # 填充模式
)

# 創建保存目錄
os.makedirs("augmented_images", exist_ok=True)


# 定義函數用來保存增強過的圖片
def save_augmented_images(datagen, x_data, y_data, output_dir, num_images=100):
    i = 0
    for batch_x, batch_y in datagen.flow(
        x_data, y_data, batch_size=1, save_to_dir=output_dir, save_format="png"
    ):
        i += 1
        if i >= num_images:
            break  # 只保存指定數量的增強圖片


# 使用增強數據並保存到磁碟
save_augmented_images(datagen, x_train, y_train, "augmented_images", num_images=100)

# 保存原始 MNIST 圖片
original_images_dir = "original_images"
os.makedirs(original_images_dir, exist_ok=True)

for i in range(100):  # 保存 100 張圖片
    img = Image.fromarray(
        (x_train[i] * 255).astype(np.uint8).squeeze(), mode="L"
    )  # 轉換為圖片格式
    img.save(os.path.join(original_images_dir, f"img_{i}.png"))

# 壓縮增強後和原始圖片到 ZIP 文件
with zipfile.ZipFile(
    "augmented_and_original_images.zip", "w", zipfile.ZIP_DEFLATED
) as zipf:
    # 壓縮增強過的圖片
    for root, dirs, files in os.walk("augmented_images"):
        for file in files:
            zipf.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), "augmented_images"),
            )

    # 壓縮原始圖片
    for root, dirs, files in os.walk(original_images_dir):
        for file in files:
            zipf.write(
                os.path.join(root, file),
                os.path.relpath(os.path.join(root, file), original_images_dir),
            )

print("增強過的圖片和原始圖片已成功保存並壓縮為 zip 檔案！")

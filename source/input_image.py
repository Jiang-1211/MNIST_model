import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from PIL import Image

original_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
modified_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"
img_path = "C:/VSCode_Projects/ML/model/build/test1.jpg"

original_model = load_model(original_model_path)
modified_model = load_model(modified_model_path)
print("Model loaded")


def preprocess_image(img_path):
    img = Image.open(img_path).convert("L")
    img = img.resize((28, 28))
    img_array = np.array(img).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array, img


img_array, img = preprocess_image(img_path)

plt.imshow(img, cmap="gray")
plt.title("Input image")
plt.axis("off")
plt.show()

original_predictions = original_model.predict(img_array)
original_predicted_digit = np.argmax(original_predictions)
modified_predictions = modified_model.predict(img_array)
modified_predicted_digit = np.argmax(modified_predictions)

print(
    "------------------------------------------------------------------------------------------------------------------------"
)
print(f"Original model prediction is : {original_predicted_digit}")
print(f"Modified model prediction is : {modified_predicted_digit}")
print(
    "------------------------------------------------------------------------------------------------------------------------"
)

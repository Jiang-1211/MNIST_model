import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os


def get_histogram(
    model_name,
    model_path,
    histogram_dir,
):
    weight_x_axis_range = (-2, 2)
    bias_x_axis_range = (-2, 2)
    all_x_range = (-2, 2)

    y_axis_range = None
    bins = "auto"

    model = tf.keras.models.load_model(model_path)
    os.makedirs(histogram_dir, exist_ok=True)

    all_weights = []

    for idx, layer in enumerate(model.layers):
        weights = layer.get_weights()

        if len(weights) > 0:
            weight_vals = weights[0].flatten()
            bias_vals = weights[1].flatten()
            all_weights.extend(weight_vals)
            all_weights.extend(bias_vals)

            plt.figure(figsize=(8, 4))
            plt.hist(weight_vals, bins=bins, alpha=0.8, color="blue")
            plt.title(f"{model_name} layer {idx} - {layer.name} - Weights")
            plt.xlabel("Value")
            # plt.ylabel("Frequency")
            if weight_x_axis_range:
                plt.xlim(weight_x_axis_range)
            if y_axis_range:
                plt.ylim(y_axis_range)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(histogram_dir, f"layer_{idx}_{layer.name}_weights.png")
            )
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.hist(bias_vals, bins=bins, alpha=0.8, color="orange")
            plt.title(f"{model_name} layer {idx} - {layer.name} - Biases")
            plt.xlabel("Value")
            # plt.ylabel("Frequency")
            if bias_x_axis_range:
                plt.xlim(bias_x_axis_range)
            if y_axis_range:
                plt.ylim(y_axis_range)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(
                os.path.join(histogram_dir, f"layer_{idx}_{layer.name}_biases.png")
            )
            plt.close()

    all_weights = np.array(all_weights)
    plt.figure(figsize=(10, 5))
    plt.hist(all_weights, bins=100, color="green", alpha=0.8)
    plt.title(f"{model_name} all Parameters")
    plt.xlabel("Value")
    # plt.ylabel("Frequency")
    if all_x_range:
        plt.xlim(all_x_range)
    if y_axis_range:
        plt.ylim(y_axis_range)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(histogram_dir, "all_parameters.png"))
    plt.close()


if __name__ == "__main__":
    original_model_path = "C:/VSCode_Projects/ML/model/build/mnist_cnn_model.h5"
    original_output_dir = "C:/VSCode_Projects/ML/model/build/original_histograms"
    modified_model_path = (
        "C:/VSCode_Projects/ML/model/build/mnist_cnn_model_modified.h5"
    )
    modified_output_dir = "C:/VSCode_Projects/ML/model/build/modified_histograms"

    get_histogram(
        "Original",
        original_model_path,
        original_output_dir,
    )
    get_histogram(
        "Modified",
        modified_model_path,
        modified_output_dir,
    )

    print(
        "------------------------------------------------------------------------------------------------------------------------"
    )
    print("Histograms of original model are saved to " + original_output_dir)
    print("Histograms of modified model are saved to " + modified_output_dir)
    print(
        "------------------------------------------------------------------------------------------------------------------------"
    )

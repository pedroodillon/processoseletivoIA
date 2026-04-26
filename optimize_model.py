import os
import tensorflow as tf


MODEL_PATH = "model.h5"
TFLITE_MODEL_PATH = "model.tflite"


def load_trained_model():
    """Load the trained Keras model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Run train_model.py before optimization."
        )

    print("Loading trained model...")
    return tf.keras.models.load_model(MODEL_PATH)


def convert_to_tflite(model):
    """Convert the Keras model to TensorFlow Lite with dynamic range quantization."""
    print("Converting model to TensorFlow Lite...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply dynamic range quantization to reduce size
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    return converter.convert()


def save_tflite_model(tflite_model):
    """Save the optimized TFLite model to disk."""
    with open(TFLITE_MODEL_PATH, "wb") as f:
        f.write(tflite_model)

    print(f"TFLite model saved as {TFLITE_MODEL_PATH}")


def print_model_sizes():
    """Print model size comparison for analysis."""
    h5_size = os.path.getsize(MODEL_PATH) / 1024
    tflite_size = os.path.getsize(TFLITE_MODEL_PATH) / 1024

    print(f"Original model size: {h5_size:.2f} KB")
    print(f"Optimized model size: {tflite_size:.2f} KB")


def main():
    model = load_trained_model()

    tflite_model = convert_to_tflite(model)

    save_tflite_model(tflite_model)

    print_model_sizes()


if __name__ == "__main__":
    main()

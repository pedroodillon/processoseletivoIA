import tensorflow as tf


def load_data():
    """Load MNIST dataset and apply basic preprocessing."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Add channel dimension (required for CNN input)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Define a small CNN architecture suitable for Edge AI."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        # First block: capture low-level features
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Second block: capture more complex patterns
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),

        # Flatten feature maps
        tf.keras.layers.Flatten(),

        # Dense layer for classification
        tf.keras.layers.Dense(64, activation="relu"),

        # Output layer (10 classes)
        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_model(model, x_train, y_train):
    """Train the model with a lightweight configuration."""
    print("Starting training...")

    model.fit(
        x_train,
        y_train,
        epochs=5,              # Keep it small for CI constraints
        batch_size=128,
        validation_split=0.1,
        verbose=2
    )


def evaluate_model(model, x_test, y_test):
    """Evaluate model performance on test data."""
    print("Evaluating model...")

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print(f"Test loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")


def save_model(model):
    """Save trained model to disk."""
    model.save("model.h5")
    print("Model saved as model.h5")


def main():
    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    train_model(model, x_train, y_train)

    evaluate_model(model, x_test, y_test)

    save_model(model)


if __name__ == "__main__":
    main()

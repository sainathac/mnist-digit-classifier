import random

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_model():
    model = Sequential(
        [
            Flatten(input_shape=(28, 28)),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=SGD(learning_rate=0.01),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def plot_training_curves(history):
    plt.figure()
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    set_seeds()

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    y_train_encoded = to_categorical(y_train, 10)
    y_test_encoded = to_categorical(y_test, 10)

    model = build_model()
    model.summary()

    history = model.fit(
        x_train,
        y_train_encoded,
        epochs=10,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    test_loss, test_accuracy = model.evaluate(x_test, y_test_encoded, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss:     {test_loss:.4f}")

    plot_training_curves(history)

    sample_index = 3
    sample = x_test[sample_index].reshape(1, 28, 28)
    prediction = model.predict(sample)
    predicted_label = int(np.argmax(prediction))

    print(f"Predicted digit for sample {sample_index}: {predicted_label}")
    print(f"Actual digit: {y_test[sample_index]}")


if __name__ == "__main__":
    main()

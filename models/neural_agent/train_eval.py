import argparse
import numpy as np
import tensorflow as tf
from agent_model import NeuralNetworkAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the Neural Network Agent"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer",
    )
    return parser.parse_args()


def train(model, train_data, train_labels, epochs, batch_size):
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.fit(
            train_data,
            train_labels,
            epochs=1,
            batch_size=batch_size,
            verbose=1
        )

def evaluate(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")


def main():
    args = parse_args()

    # Load and preprocess data (replace with actual data loading)
    train_data = np.random.random((1000, 10))
    train_labels = np.random.randint(2, size=(1000, 1))
    test_data = np.random.random((200, 10))
    test_labels = np.random.randint(2, size=(200, 1))

    # Create and compile the model
    model = NeuralNetworkAgent(input_shape=(10,), num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # Train the model
    train(model, train_data, train_labels, args.epochs, args.batch_size)

    # Evaluate the model
    evaluate(model, test_data, test_labels)


if __name__ == "__main__":
    main()

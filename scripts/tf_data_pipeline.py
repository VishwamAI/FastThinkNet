# Import necessary TensorFlow modules
import tensorflow as tf

# Define the data pipeline


def create_data_pipeline():
    # Assuming we are working with MNIST dataset for the example
    (train_images, train_labels), _ = tf.keras.datasets.mnist.load_data()
    train_images = (
        train_images.reshape((-1, 28, 28, 1)).astype("float32") / 255.0
    )

    # Create a TensorFlow dataset
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    )

    # Shuffle, batch, and prefetch the dataset
    train_dataset = (
        train_dataset.shuffle(buffer_size=1024)
        .batch(32)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )

    return train_dataset


# Create the dataset
train_dataset = create_data_pipeline()
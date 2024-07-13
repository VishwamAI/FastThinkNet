# Import necessary modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import tensorflow as tf
from models.pytorch_model import FastThinkNet
from scripts.tf_data_pipeline import create_data_pipeline

def test_integration():
    # Load a small batch of data
    train_dataset = create_data_pipeline()
    for images, labels in train_dataset.take(1):
        # Convert TensorFlow tensors to PyTorch tensors
        images = torch.from_numpy(images.numpy()).float()
        labels = torch.from_numpy(labels.numpy()).long()

        # Reshape images to match the model's input shape
        images = images.view(-1, 784)

        # Instantiate the PyTorch model
        model = FastThinkNet()

        # Move the model to the same device as the data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        images, labels = images.to(device), labels.to(device)

        # Pass the data through the model
        outputs = model(images)

        # Check if the outputs are of the expected shape
        assert outputs.shape == (32, 10), f"Output shape is incorrect. Expected (32, 10), got {outputs.shape}"

        print("Integration test passed successfully.")

if __name__ == "__main__":
    test_integration()
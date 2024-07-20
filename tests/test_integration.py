import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import pyro.nn as pyronn
import gpytorch
from models.advanced_model import AdvancedFastThinkNet
from scripts.tf_data_pipeline import create_data_pipeline


@pytest.fixture
def model():
    return AdvancedFastThinkNet()


@pytest.fixture
def data_pipeline():
    return create_data_pipeline()


def test_model_initialization(model):
    assert isinstance(model, AdvancedFastThinkNet)
    assert model.input_dim == 784
    assert model.output_dim == 10

    # Check for Bayesian fully connected layers
    assert isinstance(model.fc1, pyronn.PyroModule)
    assert isinstance(model.fc2, pyronn.PyroModule)

    # Check for GP components
    assert hasattr(model, "gp_layer")
    assert isinstance(model.gp_layer, gpytorch.models.ExactGP)

    # Check for VAE components
    assert hasattr(model, "fc_mu")
    assert hasattr(model, "fc_logvar")
    assert hasattr(model, "fc_decoder")

    # Check for vae_loss method
    assert hasattr(model, "vae_loss"), "vae_loss method missing from model"
    assert callable(
        getattr(model, "vae_loss")
    ), "vae_loss method is not callable"

    # Check for gp_loss method
    assert hasattr(model, "gp_loss"), "gp_loss method missing from model"
    assert callable(
        getattr(model, "gp_loss")
    ), "gp_loss method is not callable"


def test_forward_pass_different_sizes(model):
    model.eval()  # Set the model to evaluation mode
    batch_sizes = [1, 16, 32, 64]
    for batch_size in batch_sizes:
        input_data = torch.randn(batch_size, 1, 28, 28)  # Match MNIST image shape
        output = model(input_data)
        assert output.shape == (batch_size, model.output_dim), (
            f"Expected output shape ({batch_size}, {model.output_dim}), "
            f"but got {output.shape}"
        )


def test_basic_training_loop(model, data_pipeline):
    model.train()  # Ensure the model is in training mode
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.NLLLoss()

    def custom_loss(outputs, labels, model):
        nll_loss = criterion(outputs, labels)
        vae_loss = model.vae_loss()
        gp_loss = model.gp_loss()
        return nll_loss + 0.1 * vae_loss + 0.01 * gp_loss

    initial_loss = None
    for images, labels in data_pipeline.take(50):
        images = torch.from_numpy(images.numpy()).float()  # Keep 4D shape
        labels = torch.from_numpy(labels.numpy()).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = custom_loss(outputs, labels, model)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()
        else:
            assert loss.item() < initial_loss, "Loss did not decrease during training"


def test_integration(model, data_pipeline):
    for images, labels in data_pipeline.take(1):
        # Convert TensorFlow tensors to PyTorch tensors
        images = torch.from_numpy(images.numpy()).float()
        labels = torch.from_numpy(labels.numpy()).long()

        # Reshape images to match the model's input shape
        images = images.view(-1, 784)

        # Move the model to the same device as the data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        images, labels = images.to(device), labels.to(device)

        # Pass the data through the model
        outputs = model(images)

        # Check if the outputs are of the expected shape
        batch_size = images.shape[0]
        assert outputs.shape == (batch_size, model.output_dim), (
            f"Output shape is incorrect. "
            f"Expected ({batch_size}, {model.output_dim}), got {outputs.shape}"
        )

        # Check if the output is a valid probability distribution
        assert torch.allclose(
            outputs.exp().sum(dim=1), torch.tensor(1.0).to(device), atol=1e-6
        ), "Output is not a valid probability distribution"

        # Check if the model can handle different batch sizes
        single_image = images[:1]
        single_output = model(single_image)
        assert single_output.shape == (1, 10), (
            f"Single image output shape is incorrect. "
            f"Expected (1, 10), got {single_output.shape}"
        )


def test_error_handling(model):
    model.eval()  # Set the model to evaluation mode
    with pytest.raises(ValueError):
        # Test with incorrect input shape
        invalid_input = torch.randn(32, 3, 28, 28)  # Incorrect number of channels
        model(invalid_input)


def test_gpu_support():
    if torch.cuda.is_available():
        model = AdvancedFastThinkNet().cuda()
        assert next(model.parameters()).is_cuda
    else:
        pytest.skip("CUDA is not available, skipping GPU test")


if __name__ == "__main__":
    pytest.main([__file__])

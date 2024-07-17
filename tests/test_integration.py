import os
import sys
import pytest
import torch
from models.pytorch_model import AdvancedFastThinkNet
from scripts.tf_data_pipeline import create_data_pipeline
from pyro.nn import PyroModule

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    # Check for BNN components
    assert isinstance(model.bnn_layer, PyroModule)

    # Check for GP components
    assert hasattr(model, 'gp_approx')
    assert isinstance(model.gp_approx, torch.nn.Sequential)

    # Check for VAE components
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'decoder')


def test_forward_pass_different_sizes(model):
    batch_sizes = [1, 16, 32, 64]
    for batch_size in batch_sizes:
        input_data = torch.randn(batch_size, 784)
        output = model(input_data)
        assert output.shape == (batch_size, 10)


def test_basic_training_loop(model, data_pipeline):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    initial_loss = None
    for images, labels in data_pipeline.take(10):
        images = torch.from_numpy(images.numpy()).float().view(-1, 784)
        labels = torch.from_numpy(labels.numpy()).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if initial_loss is None:
            initial_loss = loss.item()

    assert loss.item() < initial_loss, "Loss should decrease during training"


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
        assert outputs.shape == (
            32,
            10,
        ), f"Output shape is incorrect. Expected (32, 10), got {outputs.shape}"


def test_error_handling(model):
    with pytest.raises(RuntimeError):
        # Test with incorrect input shape
        invalid_input = torch.randn(32, 100)  # Incorrect input size
        model(invalid_input)


def test_gpu_support():
    if torch.cuda.is_available():
        model = AdvancedFastThinkNet().cuda()
        assert next(model.parameters()).is_cuda
    else:
        pytest.skip("CUDA is not available, skipping GPU test")


if __name__ == "__main__":
    pytest.main([__file__])
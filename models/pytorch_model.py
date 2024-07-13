# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the neural network architecture
class FastThinkNet(nn.Module):
    def __init__(self):
        super(FastThinkNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # Example input size for MNIST
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=10)    # Example output size for MNIST

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 784)
        # Apply layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer with log softmax activation
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

# Instantiate the model
model = FastThinkNet()

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
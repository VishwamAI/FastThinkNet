# Import necessary PyTorch modules
import torch
import torch.nn as nn
import torch.nn.functional as F
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the original FastThinkNet architecture
class FastThinkNet(nn.Module):
    def __init__(self):
        super(FastThinkNet, self).__init__()
        # Define layers
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # MNIST input
        self.fc2 = nn.Linear(in_features=128, out_features=10)  # Output layer

    def forward(self, x):
        # Flatten the input tensor
        x = x.view(-1, 784)
        # Apply layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Final layer
        return x


# Define the advanced neural network architecture
class AdvancedFastThinkNet(nn.Module):
    def __init__(
        self, input_dim=784, hidden_dim=128, output_dim=10, latent_dim=20
    ):
        super(AdvancedFastThinkNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim

        # BNN component
        self.bnn_layer = PyroModule[nn.Linear](input_dim, hidden_dim)
        self.bnn_layer.weight = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_dim, input_dim]).to_event(2)
        )
        self.bnn_layer.bias = PyroSample(
            dist.Normal(0.0, 1.0).expand([hidden_dim]).to_event(1)
        )

        # GP component approximation
        self.gp_approx = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # VAE components
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        try:
            # Ensure input is correctly shaped
            if x.dim() == 4:  # (batch_size, channels, height, width)
                x = x.view(-1, self.input_dim)
            elif x.dim() == 2:
                if x.size(1) != self.input_dim:
                    raise ValueError(
                        f"Expected input dim {self.input_dim}, got {x.size(1)}"
                    )
            else:
                raise ValueError(f"Unexpected input shape: {x.shape}")

            # BNN forward pass
            x = F.relu(self.bnn_layer(x))

            # GP approximation forward pass
            x = self.gp_approx(x)

            # VAE forward pass
            encoded = self.encoder(x)
            mu, logvar = encoded.chunk(2, dim=-1)
            z = self.reparameterize(mu, logvar)
            x = self.decoder(z)

            # Final output
            return self.fc_out(x)
        except Exception as e:
            logger.error(f"Error in forward pass: {str(e)}")
            raise

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


# Instantiate the models
fast_think_net = FastThinkNet()
advanced_fast_think_net = AdvancedFastThinkNet()

# Move the models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fast_think_net.to(device)
advanced_fast_think_net.to(device)

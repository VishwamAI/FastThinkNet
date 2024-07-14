import torch.nn as nn
import torch.nn.functional as F


class FastThinkNetDeepLearning(nn.Module):
    def __init__(self, input_channels=3, num_classes=10):
        super(FastThinkNetDeepLearning, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # Assuming input size is 32x32
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)

        # Flatten the output for the fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers with ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def prune(self, pruning_rate=0.1):
        """
        Implement model pruning to find the shortest program that explains the data.
        This is a simple magnitude-based pruning method.
        """
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                tensor = module.weight.data.abs()
                threshold = (
                    tensor.view(-1)
                    .kthvalue(int(tensor.numel() * pruning_rate))
                    .values
                )
                mask = tensor.gt(threshold).float()
                module.weight.data.mul_(mask)


# Example usage:
# model = FastThinkNetDeepLearning(input_channels=3, num_classes=10)
# x = torch.randn(1, 3, 32, 32)
# output = model(x)
# model.prune(pruning_rate=0.1)

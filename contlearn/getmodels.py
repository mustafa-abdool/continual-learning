import torch
from torch import nn
from torch.nn import functional as F

class MLP(nn.Module):
    """
    basic multi-layer perceptron - flatten everything and pass it through three MLP layers and then final output of 10 logits ?
    """
    def __init__(self, hidden_size=400):
        super(MLP, self).__init__()
        self.flat = Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, 10)

    def forward(self, input):
        x = self.flat(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
    
class BasicCNN(nn.Module):
    def __init__(self, fmap_size=256, output_classes=10):
        super(BasicCNN, self).__init__()
        # First 2D convolutional layer, taking in 1 input channel (grayscale image),
        # outputting 256 convolutional features, with a square kernel size of 3
        self.conv1 = nn.Conv2d(1, fmap_size, 3, 1)
        
        # Second layer: Fully connected layer
        self.fc1 = nn.Linear(fmap_size * 26 * 26, 128)
        
        # Third layer: Fully connected layer
        self.fc2 = nn.Linear(128, 64)
        
        # Fourth layer: Fully connected layer
        self.fc3 = nn.Linear(64, output_classes)  # Output layer with 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        
        # Flatten the tensor
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x
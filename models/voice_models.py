import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class FullyConnectedNeuralNetwork(nn.Module):
    def __init__(self, input_dim=256, hidden_dim1=256, hidden_dim2=128):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        
        # Define the layers
        self.layer1 = nn.Linear(input_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, input_dim)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x, params=None):
        if params is None:
            # Use the model's parameters
            x = self.layer1(x)
            x = self.relu(x)
            x = self.layer2(x)
            x = self.relu(x)
            x = self.layer3(x)
            x = self.tanh(x)
        else:
            # Use provided parameters for meta-learning
            x = nn.functional.linear(x, params['layer1.weight'], params['layer1.bias'])
            x = self.relu(x)
            x = nn.functional.linear(x, params['layer2.weight'], params['layer2.bias'])
            x = self.relu(x)
            x = nn.functional.linear(x, params['layer3.weight'], params['layer3.bias'])
            x = self.tanh(x)
            
        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

class VanillaDeepNetwork(nn.Module):
    def __init__(self, num_layers=10, input_size=784, hidden_size=128, output_size=10):
        super(VanillaDeepNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.sigmoid(layer(x))
        x = self.layers[-1](x)
        return x

class ReLUNetwork(nn.Module):
    def __init__(self, num_layers=10, input_size=784, hidden_size=128, output_size=10):
        super(ReLUNetwork, self).__init__()
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x

class BatchNormNetwork(nn.Module):
    def __init__(self, num_layers=10, input_size=784, hidden_size=128, output_size=10):
        super(BatchNormNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_size, hidden_size))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.batch_norms.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.layers[0](x))
        
        for i in range(len(self.batch_norms)):
            x = self.layers[i + 1](x)
            x = self.batch_norms[i](x)
            x = F.relu(x)
        
        x = self.layers[-1](x)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size)
        
    def forward(self, x):
        residual = x
        x = self.linear1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.batch_norm2(x)
        x = x + residual
        x = F.relu(x)
        return x

class ResNetNetwork(nn.Module):
    def __init__(self, num_layers=10, input_size=784, hidden_size=128, output_size=10):
        super(ResNetNetwork, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        
        self.res_blocks = nn.ModuleList()
        num_blocks = max(1, (num_layers - 2) // 2)
        for _ in range(num_blocks):
            self.res_blocks.append(ResidualBlock(hidden_size))
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.input_layer(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        x = self.output_layer(x)
        return x

def get_model(model_type, num_layers, input_size=784, hidden_size=128, output_size=10):
    model_classes = {
        'vanilla': VanillaDeepNetwork,
        'relu': ReLUNetwork,
        'batchnorm': BatchNormNetwork,
        'resnet': ResNetNetwork
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model_classes[model_type](num_layers, input_size, hidden_size, output_size)

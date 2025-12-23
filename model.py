import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        layers.append(nn.ReLU())
        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
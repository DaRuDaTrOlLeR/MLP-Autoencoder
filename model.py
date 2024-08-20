import torch
import torch.nn as nn
import torch.nn.functional as F


class mymodel(torch.nn.Module):
    def __init__(self, n_inputs=784, n_bottleneck=8, n_output=784):
        super(mymodel, self).__init__()
        n2 = n_inputs // 2
        self.fc1 = nn.Linear(n_inputs, n2)
        self.fc2 = nn.Linear(n2, n_bottleneck)
        self.fc3 = nn.Linear(n_bottleneck, n2)
        self.fc4 = nn.Linear(n2, n_output)
        self.type = 'MLP4'
        self.input_shape = (1, 28, 28)

    def forward(self, x):
        return self.decode(self.encode(x))

    def encode(self, x):
        # Encoder
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)

        return x

    def decode(self, x):
        # Decoder
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)

        return x
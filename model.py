import torch.nn as nn

class MLP(nn.Module):
    """
    Architecture de réseau neuronal pour prédire la consommation d'énergie régionale
    """
    def __init__(self, input_size, output_size, hidden_size=339):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h = nn.ELU()(self.fc1(x))
        x = nn.GELU()(h)
        x = nn.Tanh()(x)
        x = self.fc3(x)
        return x
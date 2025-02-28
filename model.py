import torch.nn as nn
import os
import torch
from dragon.search_space.dag_encoding import AdjMatrix, Node

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
    

# Architecture de métamodèle pour DRAGON
class MetaArchi(nn.Module):
    """Classe définissant l'architecture du métamodèle"""
    def __init__(self, args, input_shape):
        super().__init__()
        self.input_shape = input_shape

        # Création du DAG
        assert isinstance(args['Dag'], AdjMatrix), f"The 'Dag' argument should be an 'AdjMatrix'. Got {type(args['Dag'])} instead."
        self.dag = args['Dag']
        self.dag.set(input_shape)

        # Configuration de la couche finale
        assert isinstance(args['Out'], Node), f"The 'Out' argument should be a 'Node'. Got {type(args['Node'])} instead."
        self.output = args["Out"]
        self.output.set(self.dag.output_shape)

    def forward(self, X):
        out = self.dag(X)
        return self.output(out)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        full_path = os.path.join(path, "best_model.pth")
        torch.save(self.state_dict(), full_path)
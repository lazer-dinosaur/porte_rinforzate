import sys

from torch import nn
import torch


def get_model_from_config(config: dict, device):
    thismodule = sys.modules[__name__]
    return getattr(thismodule, config['model_name']).from_config(config=config, device=device)


class MLP(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_units, device):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.hidden_units = hidden_units
        self.device = device
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, self.hidden_units),
            nn.LeakyReLU(),
            # nn.Tanh(),
            nn.Linear(self.hidden_units, self.n_outputs),
            nn.Sigmoid(),
            nn.Softmax(dim=-1)).to(self.device)
    
    def forward(self, state):
        x = torch.FloatTensor(state).to(self.device)
        x = x.flatten(-2, -1)
        return self.network(x)
    
    @classmethod
    def from_config(cls, config, device):
        return cls(n_inputs=3 * config['n_doors'],
                   n_outputs=config['n_doors'],
                   hidden_units=config['hidden_units'],
                   device=device)


class FFNResNorm(nn.Module):
    def __init__(self, model_dim: int, dropout_rate: float):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(model_dim, model_dim),
            nn.Linear(model_dim, model_dim),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate), )
    
    def forward(self, x):
        return x + self.layers(x)
    
class ResMLP(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, model_dim: int, n_layers:int, dropout_rate: float, device):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = device
        self.res_layers = [FFNResNorm(model_dim=model_dim, dropout_rate=dropout_rate) for _ in range(n_layers)]
        # Define network
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, model_dim),
            nn.LayerNorm(model_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            *self.res_layers,
            nn.Linear(model_dim, self.n_outputs),
            nn.Sigmoid(),
            nn.Softmax(dim=-1)).to(device)
    
    def forward(self, state):
        x = torch.FloatTensor(state).to(self.device)
        x = x.flatten(-2, -1)
        return self.network(x)
    
    @classmethod
    def from_config(cls, config, device):
        return cls(n_inputs=3 * config['n_doors'],
                   n_outputs=config['n_doors'],
                   model_dim=config['hidden_units'],
                   dropout_rate=config['dropout_rate'],
                   n_layers=config['n_layers'],
                   device=device)


class Transformer(nn.Module):
    def __init__(self,
                 n_inputs: int,
                 heads: int,
                 n_layers: int,
                 dropout_rate: float,
                 device,
                 model_dim: int = None):
        super().__init__()
        encoder_heads = heads
        self.device = device
        if not model_dim:
            model_dim = 2 ** 3
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim * encoder_heads, nhead=encoder_heads)
        self.encoder = nn.Sequential(
            nn.Linear(n_inputs, model_dim * encoder_heads),
            nn.LayerNorm(model_dim * encoder_heads),
            nn.LeakyReLU(),
            nn.Dropout(dropout_rate),
            nn.TransformerEncoder(encoder_layer, num_layers=n_layers))
        self.output = nn.Sequential(
            nn.Linear(model_dim * encoder_heads, 1),
            nn.Sigmoid(),
            nn.Softmax(dim=1)).to(device)
    
    def forward(self, state):
        x = torch.FloatTensor(state).transpose(-2, -1).to(self.device)
        if len(x.shape) < 3:
            x = x[None]
        x = self.encoder(x)
        x = self.output(x)
        return x.squeeze()
    
    @classmethod
    def from_config(cls, config, device):
        return cls(n_inputs=3,
                   heads=config['heads'],
                   n_layers=config['n_layers'],
                   dropout_rate=config['dropout_rate'],
                   model_dim=config['model_dim'],
                   device=device)

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import copy

# Initialise network parameters
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.bias.data.fill_(0)

        nn.init.xavier_uniform_(m.weight, gain=0.5)

    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# Create the encoder class for the autoencoder, extending from PyTorch's nn.module
class Encoder(nn.Module):
    def __init__(self, latent_size, hidden_size_rule, input_dim, device):
        super(Encoder, self).__init__()

        # Make the network based on the hidden layer sizes
        if len(hidden_size_rule) == 2:
            self.layer_sizes = [input_dim, hidden_size_rule[0], latent_size]
        elif len(hidden_size_rule) == 3:
            self.layer_sizes = [
                input_dim,
                hidden_size_rule[0],
                hidden_size_rule[1],
                latent_size,
            ]
        # Make the model
        modules = []
        for i in range(len(self.layer_sizes) - 2):

            modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            modules.append(nn.ReLU())
        # Combined the layers into a network
        self.feature_encoder = nn.Sequential(*modules)

        print(input_dim, hidden_size_rule, latent_size, self.layer_sizes[-2])

        # Set the distribution layers
        self._mu = nn.Linear(in_features=self.layer_sizes[-2], out_features=latent_size)
        self._logvar = nn.Linear(
            in_features=self.layer_sizes[-2], out_features=latent_size
        )

        # Initialise the network weights
        self.apply(weights_init)

        self.to(device)

    def forward(self, x):
        # Apply the network to the input
        h = self.feature_encoder(x)

        # Get the distribution parameters
        mu = self._mu(h)
        logvar = self._logvar(h)

        return mu, logvar


# Decoder portion of Autoencoder
class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size_rule, output_dim, device):
        super().__init__()

        # Create the network
        self.layer_sizes = [latent_size, hidden_size_rule[-1], output_dim]
        self.feature_decoder = nn.Sequential(
            nn.Linear(latent_size, self.layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(self.layer_sizes[1], output_dim),
        )

        # Initialise the weights
        self.apply(weights_init)

        self.to(device)

    def forward(self, x):
        # Apply the network to the input
        return self.feature_decoder(x)

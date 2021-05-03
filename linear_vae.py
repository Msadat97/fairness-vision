import torch
from torch import nn
import numpy as np


class LinearVAE(nn.Module):
    def __init__(self, latent_dim=2, input_dim=(28, 28)) -> None:
        super().__init__()
        self.latent_dimension = latent_dim
        self.input_shape = input_dim
        self.flatten_dim = int(np.prod(input_dim))
        
        self.encoder = nn.Sequential(
            nn.Linear(self.flatten_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 2*self.latent_dimension)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dimension, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, self.flatten_dim)
        )
                
    def sampling(self, mu, log_var):
        n_samples = mu.shape[0]
        epsilon = torch.randn((n_samples, self.latent_dimension))
        if mu.is_cuda:
            epsilon = epsilon.cuda()
        return mu + torch.exp(0.5*log_var) * epsilon

    def encoder_forward(self, x):
        x = nn.Flatten()(x)
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, chunks=2, dim=-1)
        z = self.sampling(mu, log_var)
        return z, mu, log_var

    def decoder_forward(self, z):
        x = self.decoder(z)
        x = torch.reshape(x, (x.shape[0], *self.input_shape))
        x = torch.sigmoid(x)
        return x

    def forward(self, x):
        z, mu, log_var = self.encoder_forward(x)
        x_hat = self.decoder_forward(z)
        return x_hat, mu, log_var

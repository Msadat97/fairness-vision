from pickle import load
from numpy.core.numeric import Inf
import torch
import torch.nn as nn
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
# import torch.nn.functional as F


class VAE(nn.Module):
    
    def __init__(self, latent_dim = 2, input_shape=(1, 1, 28, 28)):
        super().__init__()
        
        self.latent_dimension = latent_dim
        self.input_shape = input_shape
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.shape, self.flatten_shape = self.get_shape()
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flatten_shape, 64),
            nn.ReLU(),   
            nn.Linear(in_features=64, out_features=2*self.latent_dimension)
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dimension, out_features=self.flatten_shape),
            nn.ReLU(),
        )
        
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
        )  
        
        self.output_decoder = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
        
    def get_shape(self):
        x = torch.randn(self.input_shape)
        x = self.cnn_encoder(x)
        return x.shape[1:], torch.prod(torch.tensor(x.shape)).item()
    
    def sampling(self, mu, log_var):
        n_samples = mu.shape[0]
        epsilon = torch.randn((n_samples, self.latent_dimension))
        if mu.is_cuda:
            epsilon = epsilon.cuda()
        return mu + torch.exp(0.5*log_var) * epsilon
    
    def encoder_forward(self, x):
        x = self.cnn_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_encoder(x)
        mu, log_var = torch.chunk(x, chunks=2, dim=-1)
        z = self.sampling(mu, log_var)
        return z, mu, log_var
    
    def decoder_forward(self, z):
        x = self.fc_decoder(z)
        x = torch.reshape(x, (x.shape[0], *self.shape))
        x = self.cnn_decoder(x)
        x = self.output_decoder(x)
        x = torch.sigmoid(x)
        return x
    
    def forward(self, x):
        z, mu, log_var = self.encoder_forward(x)
        x_hat = self.decoder_forward(z)
        return x_hat, mu, log_var


class Encoder(nn.Module):
    def __init__(self, input_shape=(1, 1, 28, 28), num_classes=10) -> None:
        super().__init__()

        kernel_size = 3
        filters = 64

        self.input_shape = input_shape
        self.num_classes = num_classes

        encoder_modules = nn.ModuleList()

        encoder_modules.append(nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=kernel_size)),
        encoder_modules.append(nn.ReLU())

        for _ in range(2):
            encoder_modules.append(nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size)),
            encoder_modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        kernel_size = 3
        filters = 64

        decoder_moduls = nn.ModuleList()

        for _ in range(2):
            decoder_moduls.append(nn.ConvTranspose2d(in_channels=filters,
                           out_channels=filters, kernel_size=kernel_size)),
            decoder_moduls.append(nn.ReLU())

        decoder_moduls.append(
            nn.ConvTranspose2d(in_channels=filters, out_channels=1, kernel_size=kernel_size)),
        decoder_moduls.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_moduls)
    
    def forward(self, x):
        return self.decoder(x)


class LatentEncoder(nn.Module):
    def __init__(self, input_shape=(1, 1, 28, 28), num_classes=10) -> None:
        super().__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.encoder = Encoder(input_shape, num_classes)
        self.decoder = Decoder()

        drop_out_rate = 0.2
        self.shape, self.flatten_shape = self.get_shape()
        self.classifier = nn.Sequential(
            nn.Dropout(p=drop_out_rate),
            nn.Linear(self.flatten_shape, self.num_classes)
        )

    def get_shape(self):
        x = torch.randn(self.input_shape)
        x = self.encoder(x)
        return x.shape[1:], torch.prod(torch.tensor(x.shape)).item()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        x = self.encoder(x)
        features = torch.flatten(x, start_dim=1)
        out = self.classifier(features)
        return features, out


class AutoEncoder(nn.Module):
    def __init__(self, vae: nn.Module, latent_encoder: nn.Module, freeze: bool = True):
        super().__init__()
        self.vae = vae
        self.encoder = latent_encoder
        
        if freeze:
            self._freeze_vae()
    
    def forward(self, x):
        x = self.vae.decoder_forward(x)
        x, _ = self.encoder(x)
        return x
    
    def _freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad_(False)   


class LatentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.linear = nn.Linear(input_dim, num_classes).to(device)
        self.softmax = nn.Softmax(dim=1).to(device)

    def forward(self, x):
        return self.linear(x).squeeze()

    def predict(self, x):
        _, labels = torch.max(self.linear(x), dim=1)
        return labels

    def logits(self, x):
        return self.softmax(self.linear(x))


# from mnist import MnistLoader
# m = MnistLoader()
# x = m.get_data('x_train')
# le = LatentEncoder()
# enc = le.encode(x[0:1])
# dec = le.decode(enc)
# print(dec.shape)

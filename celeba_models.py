from torch import nn
from torch.nn import init
import torch 
import numpy as np
from models import BaseVAE


def fc_block(in_features, out_features, use_bn=True, use_relu=True):
    moduels = [nn.Linear(in_features, out_features)]
    if use_bn:
        moduels.append(nn.BatchNorm1d(out_features))
    if use_relu:
        moduels.append(nn.ReLU(True))
    return nn.ModuleList(moduels)


def conv_block(in_channels, out_channels, kernel_size,
               stride=1, padding=0,
               use_bn=True, use_relu=True):
    modules = [nn.Conv2d(in_channels, out_channels,
                         kernel_size, stride, padding)]
    if use_bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if use_relu:
        modules.append(nn.ReLU(True))
    return nn.ModuleList(modules)


def deconv_block(in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,
                 use_bn=True, use_relu=True):
    moduels = [nn.ConvTranspose2d(
        in_channels, out_channels, kernel_size, stride, padding, output_padding)]
    if use_bn:
        moduels.append(nn.BatchNorm2d(out_channels))
    if use_relu:
        moduels.append(nn.ReLU(True))
    return nn.ModuleList(moduels)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class CelebaVAE(BaseVAE):
    def __init__(self, latent_dim, input_dim):
        super().__init__(latent_dim, input_dim)

        self.latent_dim = latent_dim
        self.input_dim = input_dim

        hidden_dims = [3, 32, 32, 64, 64]

        modules = []
        for i, _ in enumerate(hidden_dims[:-1]):
            modules.extend(
                conv_block(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1],
                           kernel_size=4, stride=2, padding=1, use_bn=False)
            )

        modules.extend(
            conv_block(in_channels=hidden_dims[-1], out_channels=256,
                       kernel_size=4, stride=1, padding=0, use_bn=False)
        )

        self.cnn_encoder = nn.Sequential(*modules)

        self.shape, self.flatten_shape = self.get_shape()

        self.fc_encoder = nn.Linear(
            in_features=self.flatten_shape, out_features=self.latent_dim*2)

        self.fc_decoder = nn.Linear(
            in_features=self.latent_dim, out_features=self.flatten_shape)

        hidden_dims.reverse()
        modules = []
        modules.extend(
            deconv_block(in_channels=256, out_channels=hidden_dims[0],
                         kernel_size=4, stride=1, padding=0, use_bn=False)
        )

        for i, _ in enumerate(hidden_dims[:-2]):
            modules.extend(
                deconv_block(in_channels=hidden_dims[i], out_channels=hidden_dims[i+1],
                             kernel_size=4, stride=2, padding=1,  use_bn=False)
            )

        modules.append(
            nn.ConvTranspose2d(in_channels=hidden_dims[-2], out_channels=hidden_dims[-1],
                               kernel_size=4, stride=2, padding=1)
        )

        self.cnn_decoder = nn.Sequential(*modules)

        self.apply(kaiming_init)

    def get_shape(self):
        x = torch.zeros(1, *self.input_dim)
        x = self.cnn_encoder(x)
        return x.shape[1:], torch.prod(torch.tensor(x.shape)).item()

    def encode(self, x):
        x = self.cnn_encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_encoder(x)
        mu, log_var = torch.chunk(x, chunks=2, dim=-1)
        z = self.sampling(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        x = self.fc_decoder(z)
        x = torch.reshape(x, (x.shape[0], *self.shape))
        x = self.cnn_decoder(x)
        x = torch.sigmoid_(x)
        return x


class EncoderCelebA(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        
        filters = [3, 64, 64, 128, 128, 256, 256]
        
        modules = []
        for i, _ in enumerate(filters[:-1]):
            modules.extend(
                conv_block(in_channels=filters[i], out_channels=filters[i+1],
                            kernel_size=4, stride=2, padding=1)
            )
        self.cnn_encoder = nn.Sequential(*modules)
        
        self.shape, self.flatten_shape = self.get_shape()
        self.fc_encoder = nn.Sequential(
            *fc_block(in_features=self.flatten_shape, out_features=1024),
        )
        
        self.fc_decoder = nn.Sequential(
            *fc_block(in_features=1024, out_features=self.flatten_shape),
        )
        
        modules = []
        filters.reverse()
        for i, _ in enumerate(filters[:-1]):
            modules.extend(
                deconv_block(in_channels=filters[i], out_channels=filters[i+1],
                           kernel_size=4, stride=2, padding=1)
            )
        
        self.cnn_decoder = nn.Sequential(*modules)
        
    @torch.no_grad()
    def get_shape(self):
        self.eval()
        x = torch.zeros(1, *self.input_dim)
        x = self.cnn_encoder(x)
        return tuple(x.shape[1:]), np.prod(x.shape)
    
    def forward(self, x):
        return self.encode(x)
    
    def encode(self, x):
        x = self.cnn_encoder(x).flatten(1)
        return self.fc_encoder(x)
        
    def decode(self, x):
        x = self.fc_decoder(x)
        x = x.reshape(-1, *self.shape)
        return self.cnn_decoder(x)
        


class ClassifierCelebA(nn.Module):
    def __init__(self, input_dim, num_class=2):
        super().__init__()
        self.input_dim = input_dim
        self.classifier = nn.Linear(input_dim, num_class)

    def forward(self, x):
        return self.classifier(x).squeeze()

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(1)


class AutoEncoderCelebA(nn.Module):
    def __init__(self, vae: BaseVAE, latent_encoder: EncoderCelebA, freeze: bool = True):
        super().__init__()
        self.vae = vae
        self.encoder = latent_encoder

        if freeze:
            self._freeze_vae()

    def forward(self, x):
        return self.encode(x)

    def encode(self, x):
        x = self.vae.decode(x)
        x = self.encoder.encode(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def decode(self, x):
        x = self.encoder.decode(x)
        z, _, _ = self.vae.encode(x)
        return z

    def _freeze_vae(self):
        for param in self.vae.parameters():
            param.requires_grad_(False)


class AttributeClassifierCelebA(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        filters = [3, 64, 64, 128, 128, 256, 256]

        modules = []
        for i, _ in enumerate(filters[:-1]):
            modules.extend(
                conv_block(in_channels=filters[i], out_channels=filters[i+1],
                           kernel_size=4, stride=2, padding=1)
            )
        self.cnn = nn.Sequential(*modules)

        self.shape, self.flatten_shape = self.get_shape()
        self.classifier = nn.Sequential(
            *fc_block(in_features=self.flatten_shape, out_features=1024),
            nn.Linear(in_features=1024, out_features=2)
        )

    @torch.no_grad()
    def get_shape(self):
        self.eval()
        x = torch.zeros(1, *self.input_dim)
        x = self.cnn(x)
        return tuple(x.shape[1:]), np.prod(x.shape)

    def forward(self, x):
        x = self.cnn(x).flatten(1)
        x = self.classifier(x)
        return x

    def predict(self, x):
        logits = self.forward(x)
        return logits.argmax(1)
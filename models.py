from abc import abstractmethod
import torch
import torch.nn as nn
import numpy as np 
from torch.distributions import Bernoulli
from torch.nn import modules
from torch.nn.modules import module
from torch.nn.modules.activation import ReLU
from torch.nn import init


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


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
    modules = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)]
    if use_bn:
        modules.append(nn.BatchNorm2d(out_channels))
    if use_relu:
        modules.append(nn.ReLU(True))
    return nn.ModuleList(modules)


def deconv_block(in_channels, out_channels, kernel_size, 
                 stride=1, padding=0, output_padding=0, 
                 use_bn=True, use_relu=True):
    moduels = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)]
    if use_bn:
        moduels.append(nn.BatchNorm2d(out_channels))
    if use_relu:
        moduels.append(nn.ReLU(True))
    return nn.ModuleList(moduels)



class BaseVAE(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
    
    def sampling(self, mu, log_var):
        n_samples = mu.shape[0]
        epsilon = torch.randn((n_samples, self.latent_dim))
        if mu.is_cuda:
            epsilon = epsilon.cuda()
        return mu + torch.exp(0.5*log_var) * epsilon
    
    @abstractmethod
    def encode(self, inputs):
        raise NotImplemented
    
    @abstractmethod
    def decode(self, z):
        raise NotImplemented
    
    def forward(self, x):
        z, mu, log_var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def dist_predict(self, z):
        dist_param = self.decode(z)
        dist = Bernoulli(dist_param)
        return dist.sample()
    

class LinearVAE(BaseVAE):
    def __init__(self, latent_dim=2, input_dim=(28, 28)) -> None:
        super().__init__(latent_dim, input_dim)
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        flatten_dim = int(np.prod(input_dim))

        self.encoder = nn.Sequential(
            nn.Linear(flatten_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, 2*self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 300),
            nn.ReLU(),
            nn.Linear(300, 400),
            nn.ReLU(),
            nn.Linear(400, self.flatten_dim)
        )

    def encode(self, x):
        x = nn.Flatten()(x)
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, chunks=2, dim=-1)
        z = self.sampling(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        x = self.decoder(z)
        x = torch.reshape(x, (x.shape[0], *self.input_dim))
        x = torch.sigmoid(x)
        return x


class VAE(BaseVAE):

    def __init__(self, latent_dim=2, input_dim=(1, 28, 28)):
        super().__init__(latent_dim, input_dim)
        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.cnn_encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.input_dim[0], out_channels=32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        
        self.shape, self.flatten_shape = self.get_shape()
        
        self.fc_encoder = nn.Sequential(
            nn.Linear(self.flatten_shape, 1024),
            nn.ReLU(),   
            nn.Linear(in_features=1024, out_features=2*self.latent_dim)
        )
        
        self.fc_decoder = nn.Sequential(
            nn.Linear(in_features=self.latent_dim, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=1024),
            nn.Linear(in_features=1024, out_features=self.flatten_shape),
            nn.ReLU(),
            nn.BatchNorm1d(self.flatten_shape)
        )
        
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=self.input_dim[0], kernel_size=4, stride=2),
        )
        
        # self.output_decoder = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)

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
        x = torch.sigmoid(x)
        return x



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
        
        # modules = []
        # modules.extend(fc_block(in_features=self.flatten_shape, out_features=256, use_bn=False))
        # modules.extend(fc_block(in_features=256, out_features=256, use_bn=False))
        # modules.extend(fc_block(in_features=256, out_features=self.latent_dim*2, use_bn=False))
            
        self.fc_encoder = nn.Linear(in_features=self.flatten_shape, out_features=self.latent_dim*2)
        
        
        
        # modules = []
        # modules.extend(fc_block(in_features=self.latent_dim, out_features=256, use_bn=False))
        # modules.extend(fc_block(in_features=256, out_features=256, use_bn=False))
        # modules.extend(fc_block(in_features=256, out_features=self.flatten_shape, use_bn=False))
        
        self.fc_decoder = nn.Linear(in_features=self.latent_dim, out_features=self.flatten_shape)
        
        
        hidden_dims.reverse()
        modules = []
        modules.extend(
            deconv_block(in_channels=256, out_channels=hidden_dims[0], 
                         kernel_size=4, stride=1, padding=0, use_bn=False)
        )
        
        for i, _ in enumerate(hidden_dims[:-2]):
            
            # # adding output_padding to one layer before the last layer.
            # output_padding = 1 if i == len(hidden_dims) - 3 else 0
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
            
        self.fc_encoder = nn.Linear(in_features=self.flatten_shape, out_features=self.latent_dim*2)
        
        self.fc_decoder = nn.Linear(in_features=self.latent_dim, out_features=self.flatten_shape)
        
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


class Encoder(nn.Module):
    def __init__(self, input_shape=(1, 1, 28, 28), num_classes=10) -> None:
        super().__init__()

        kernel_size = 3
        filters = 64

        self.input_shape = input_shape
        self.num_classes = num_classes

        encoder_modules = nn.ModuleList()

        encoder_modules.append(nn.Conv2d(in_channels=1, out_channels=filters, kernel_size=kernel_size))
        encoder_modules.append(nn.ReLU())

        for _ in range(2):
            encoder_modules.append(nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size))
            encoder_modules.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        kernel_size = 3
        filters = 64

        decoder_modules = nn.ModuleList()

        for _ in range(2):
            decoder_modules.append(nn.ConvTranspose2d(in_channels=filters, out_channels=filters, kernel_size=kernel_size))
            decoder_modules.append(nn.ReLU())

        decoder_modules.append(nn.ConvTranspose2d(in_channels=filters, out_channels=1, kernel_size=kernel_size))
        decoder_modules.append(nn.ReLU())

        self.decoder = nn.Sequential(*decoder_modules)
    
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
        x = torch.zeros(self.input_shape)
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
    def __init__(self, vae: VAE, latent_encoder: LatentEncoder, freeze: bool = True):
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
        x = x.reshape((-1, *self.encoder.shape))
        x = self.encoder.decode(x)
        z, _, _ = self.vae.encode(x)
        return z

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
        _, labels = torch.max(self.linear(x), dim=-1)
        return labels

    def logits(self, x):
        return self.softmax(self.linear(x))


class DataModel(nn.Module):
    def __init__(self, encoder: AutoEncoder, classifier: LatentClassifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def encode(self, x):
        return self.encoder.encode(x)
    
    def classify(self, x):
        return self.classifier(x)
    
    def forward(self, x):
        return self.classify(self.encode(x))
    
    def predict(self, z):
        x = self.encode(z)
        return self.classifier.predict(x)
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad_(False)


class ClassifierCelebA(nn.Module):
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
            nn.Linear(in_features=1024, out_features=1)
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
        return x.ravel()
    
    def predict(self, x):
        logits = self.forward(x)
        return torch.where(logits > 0, 1.0, 0.0)

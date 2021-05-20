import torch
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from torch.utils.data import TensorDataset
from torch import optim
from mnist import MnistLoader
from model import VAE, LatentEncoder
from train import VAETrainer, LatentTrainer
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
#
# data = MnistLoader(shuffle=True, normalize=False, batch_size=128)
# clf = LatentEncoder()
# clf.to(device)
# optimizer = optim.RMSprop(clf.parameters(), lr=1e-3)
#
# trainer = LatentTrainer(clf,
#                         optimizer,
#                         loss_fn=nn.CrossEntropyLoss(),
#                         train_loader=data.train_loader,
#                         val_loader=data.test_loader)
#
# trainer.train(epochs=10)
# torch.save(clf.state_dict(), 'latent_state_dict_v1')

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_normal_(m.weight)
#         nn.init.constant_(m.bias, 0.05)

if __name__ == "__main__":
    data = MnistLoader(batch_size=100, shuffle=True, normalize=False)
    vae = VAE(latent_dim=10)
    # x = data.get_data('x_train')[0:5].cuda()
    # vae = LinearVAE(16)
    vae.to(device)
    # vae.apply(weights_init)

    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    trainer = VAETrainer(vae, optimizer, train_loader=data.train_loader, beta=4)
    trainer.train(epochs=75)
    torch.save(vae.state_dict(), 'saved_models/vae-state-dict-v4')

from os.path import split
from torch.cuda.random import seed
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision.datasets import MNIST, CelebA
from model import VAE
import os
from pathlib import Path
import os

data_path = Path.cwd().joinpath('data')
data_path.mkdir(parents=True, exist_ok=True)

class CostumMNIST(Dataset):
    def __init__(self, train, vae: VAE = None, with_channel=False) -> None:
        super().__init__()
        self.dataset = MNIST(root=str(data_path), train=train, download=True)
        self.vae = vae
        self.with_channel = with_channel
        
    def __len__(self):
        return len(self.dataset.data)
    
    def __getitem__(self, index):
        data = self.dataset.data[index]
        data = data/255.0
        targets = self.dataset.targets[index]
        
        if self.vae is not None:
            data = data[None, None, ...] if isinstance(index, int) else data[:, None, ...]
            data = self._get_latents(data)[0]
            return (data, targets)
        else:
            if self.with_channel:
                data = data[None, ...] if isinstance(index, int) else data[:, None, ...]
            return (data, targets)
        
    
    @torch.no_grad()
    def _get_latents(self, data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)
        vae = self.vae.to(device)
        z_batch, _, _ = self.vae.encoder_forward(data)
        return z_batch.cpu()


class CustomCelebA(Dataset):
    def __init__(self, split='train') -> None:
        super().__init__()
        self.data = CelebA(root=str(data_path), split=split, download=True)


t = CustomCelebA()

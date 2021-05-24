from pathlib import Path

from torch.functional import split
from models import BaseVAE
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CelebA
import torch
import tensorflow_datasets as tfds

DATA_PATH = Path.cwd().joinpath('data')
DATA_PATH.mkdir(parents=True, exist_ok=True)


class CustomMNIST(Dataset):
    def __init__(self, train, with_channel=False) -> None:
        super().__init__()
        self.dataset = MNIST(root=str(DATA_PATH), train=train, download=True)
        self.with_channel = with_channel
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset.data[index]
        targets = self.dataset.targets[index]
        data = data/255.0
        if self.with_channel:
            data = data[None, ...] if isinstance(index, int) else data[:, None, ...]
        return data, targets


class CustomCelebA(Dataset):
    def __init__(self, split='train') -> None:
        super().__init__()
        self.data = CelebA(root=str(DATA_PATH), split=split, download=False)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        pass


class VAEWrapper(Dataset):
    def __init__(self, vae: BaseVAE, dataset: Dataset):
        super(VAEWrapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dataset = dataset
        self.vae = vae.to(self.device)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data, targets = self.dataset[index]
        if isinstance(index, int):
            data = data[None, ...]
        if data.ndim == 3:
            data = data[None, ...]
        assert data.ndim == 4
        latents = self._get_latents(data)
        return latents, targets

    @torch.no_grad()
    def _get_latents(self, data):
        data = data.to(self.device)
        z_batch, _, _ = self.vae.encode(data)
        return z_batch.cpu()

t = CustomCelebA()
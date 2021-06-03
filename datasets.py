from pathlib import Path
from models import BaseVAE
from torch.utils.data import Dataset
from torchvision.datasets import MNIST, CelebA
import torch
from torchvision import transforms
from utils import get_celeba_att
from tqdm import tqdm

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
        self.dataset = CelebA(root=str(DATA_PATH), split=split, download=False)
        self.transform = self.default_data_transforms()
        self.att_idx = get_celeba_att()['smiling']
        
        self.image_size = self.__getitem__(0)[0].shape

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        pic, label = self.dataset[index]
        data = self.transform(pic)
        
        return data, label[self.att_idx]
    
    def default_data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        transform = transforms.Compose(
            [
                # transforms.RandomHorizontalFlip(),
                # transforms.CenterCrop(148),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                # SetRange
            ]
        )
        return transform


class VAEWrapper(Dataset):
    def __init__(self, vae: BaseVAE, dataset: Dataset):
        super(VAEWrapper, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        self.vae = vae.to(self.device)
        self.dataset = dataset
        self.cached_data = {'data': [], 'target': []}
        # self._cache_data(dataset)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # return self.cached_data['data'][index], self.cached_data['target'][index]
        data, targets = self.dataset[index]
        data = data[None, ...]
        if data.ndim == 3:
            data = data[None, ...]
        assert data.ndim == 4
        latents = self._get_latents(data)
        img = self.vae.decode(latents)
        return img[0], targets
            
    def _cache_data(self, dataset):
        print("="*50)
        print("Caching the Data")
        for item in tqdm(iter(dataset), total=len(dataset)):
            data, targets = item
            data = data[None, ...]
            if data.ndim == 3:
                data = data[None, ...]
            assert data.ndim == 4
            latents = self._get_latents(data)
            img = self.vae.decode(latents)
            self.cached_data['data'].append(img[0].cpu())
            self.cached_data['target'].append(targets.cpu())
        print("="*50)
    
    @torch.no_grad()
    def _get_latents(self, data):
        data = data.to(self.device)
        z_batch, _, _ = self.vae.encode(data)
        return z_batch

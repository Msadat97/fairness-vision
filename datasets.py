from pathlib import Path
import pickle
from unicodedata import name

from torch.utils.data.dataloader import DataLoader
from models import BaseVAE
from torch.utils.data import Dataset, dataset
from torchvision.datasets import MNIST, CelebA
import torch
from torchvision import transforms
from utils import get_celeba_att
from tqdm import tqdm
import os 
import lmdb
import io
from PIL import Image
import numpy as np
from copy import deepcopy

DATA_PATH = Path.cwd().joinpath('data')
DATA_PATH.mkdir(parents=True, exist_ok=True)

def celeba_default_data_transforms():
    transform = transforms.Compose(
        [
            # transforms.RandomHorizontalFlip(),
            # transforms.CenterCrop(148),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
    )
    return transform

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


# class CustomCelebA(Dataset):
#     def __init__(self, split='train') -> None:
#         super().__init__()
#         self.dataset = CelebA(root=str(DATA_PATH), split=split, download=False)
#         self.transform = self.default_data_transforms()
#         self.att_idx = get_celeba_att()['smiling']
        
#         self.image_size = self.__getitem__(0)[0].shape

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         pic, label = self.dataset[index]
#         data = self.transform(pic)
        
#         return data, label[self.att_idx]
    
#     def default_data_transforms(self):
#         SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
#         transform = transforms.Compose(
#             [
#                 # transforms.RandomHorizontalFlip(),
#                 # transforms.CenterCrop(148),
#                 transforms.Resize((64, 64)),
#                 transforms.ToTensor(),
#                 # SetRange
#             ]
#         )
#         return transform


class CustomCelebA(Dataset):
    def __init__(self, root=None, split='train') -> None:
        super().__init__()
        if root is None:
            root = DATA_PATH.joinpath('celeba_lmdb')
        self.dataset = LMDBDataset(root=root, name='celeba', split=split, is_encoded=True)
        self.transform = celeba_default_data_transforms()
        self.att_idx = get_celeba_att()['blond_hair']

        self.image_size = self.__getitem__(0)[0].shape

    def __len__(self):
        return self.dataset.dataset_size

    def __getitem__(self, index):
        pic, label = self.dataset[index]
        data = self.transform(pic)

        return data, label[self.att_idx]


class VAEWrapper(Dataset):
    def __init__(self, vae: BaseVAE, dataset: Dataset):
        super(VAEWrapper, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        self.vae = deepcopy(vae)
        self.vae.to(self.device)
        self.dataset = dataset
        self.cached_data = {'data': [], 'target': []}
        # self._cache_data()

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
        # img = self.vae.decode(latents)
        return latents[0], targets
    
    @torch.no_grad()   
    def _cache_data(self):
        print("="*50)
        print("Caching the Data")
        loader = DataLoader(self.dataset, num_workers=32)
        for item in tqdm(loader, total=len(loader)):
            data, targets = item
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


class LMDBDataset(Dataset):
    def __init__(self, root, name="", split='train', is_encoded=False) -> None:
        super().__init__()
        self.split = split
        self.name = name
        lmdb_path = os.path.join(root, f'{self.split}.lmdb')
        self.data_lmdb = lmdb.open(lmdb_path, readonly=True, max_readers=1,
                                   lock=False, readahead=False, meminit=False)
        self.is_encoded = is_encoded
        
        data_size_path = Path(lmdb_path).joinpath(f'{self.split}_dsize.pkl')
        with open(data_size_path, 'rb') as f:
            self.dataset_size = pickle.load(f)

    def __getitem__(self, index):
        # target = [0]
        with self.data_lmdb.begin(write=False, buffers=True) as txn:
            data = txn.get(str(index).encode())
            data, target = pickle.loads(data)
            target = torch.tensor(target, dtype=torch.long)
            if self.is_encoded:
                img = Image.open(io.BytesIO(data))
                img = img.convert('RGB')
            else:
                img = np.asarray(data, dtype=np.uint8)
                # assume data is RGB
                size = int(np.sqrt(len(img) / 3))
                img = np.reshape(img, (size, size, 3))
                img = Image.fromarray(img, mode='RGB')

        return img, target

    def __len__(self):
        return self.dataset_size
    
# x = CustomCelebA(root="./data/celeba64_lmdb")
# print(x[0])

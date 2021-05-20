import torch
from sklearn.metrics import accuracy_score
from models import VAE
from torch.utils.data import TensorDataset, DataLoader


def get_latents(vae: VAE, data_loader: DataLoader, shuffle, device=torch.device('cpu')):
    z_list = []
    y_list = []
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        z_batch, _, _ = vae.encode(inputs)
        z_list.append(z_batch)
        y_list.append(targets)
    z_tensor = torch.cat(z_list)
    y_tensor = torch.cat(y_list)
    data_set = TensorDataset(z_tensor, y_tensor)
    batch_size = data_loader.batch_size
    return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)


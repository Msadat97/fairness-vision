from pathlib import Path
import torch
from sklearn.metrics import accuracy_score
from models import VAE
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import shutil
import json

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



def load_model(model: torch.nn.Module, model_path):
    model.load_state_dict(
        torch.load(model_path)
    )


def rmdir(directory):
    for item in directory.iterdir():
        if item.is_dir():
            rmdir(item)
        else:
            item.unlink()
    directory.rmdir()


def prepare_tensorboard(dir: str):
    dir_path = Path(dir)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(dir_path))
    return tb

def pred1d(logits):
    with torch.no_grad():
        return torch.where(logits > 0, 1.0, 0.0)

def predNd(logits):
    with torch.no_grad():
        return logits.argmax(1)


def load_json(file_dir):
    with open(file_dir, 'r') as f:
        return json.load(f)
    

def get_celeba_att():
    att_list = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
    att_list = att_list.lower().split()
    att_list = {att: idx for idx, att in enumerate(att_list)}
    return att_list
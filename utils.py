import json
import shutil
from pathlib import Path

import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import VAE


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


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def prepare_tensorboard(dir: str):
    dir_path = Path(dir)
    if dir_path.exists():
        shutil.rmtree(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(str(dir_path))
    return tb


def load_json(file_dir):
    with open(file_dir, 'r') as f:
        return json.load(f)
    

def get_celeba_att():
    att_list = "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
    att_list = att_list.lower().split()
    att_list = {att: idx for idx, att in enumerate(att_list)}
    return att_list


def image_to_vid(images, path):
    to_pil_image = transforms.ToPILImage()
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(path, imgs)


def get_logdir(config):
    dir = Path(config['log_path'])
    exp_id = config['exp_id']
    dir = dir.joinpath(f"run{exp_id}")
    return str(dir)

def prepare_config(path):
    config = load_json(path)
    im_size = config['celeba']['image_size']
    config['celeba']['input_dim'] = (3, im_size, im_size)
    
    return config

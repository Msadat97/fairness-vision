import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from os import stat
from pathlib import Path
import os
import imageio
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from models import BaseVAE


def get_latents(vae: BaseVAE, data_loader: DataLoader, shuffle, device=torch.device('cpu')):
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


def write_dict_tb(writer: SummaryWriter, stats: dict, epoch: int):
    for key, val in stats.items():
        writer.add_scalar()


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


@dataclass
class VAEStat:
    epoch: int = 0
    loss: float = 0.0
    recon_loss: float = 0.0
    kl_loss: float = 0.0
    
    def __repr__(self) -> str:
        return f"epoch={self.epoch:<10} total-loss={self.loss:<20.5f} recon-loss={self.recon_loss:<20.5f} kl-loss={self.kl_loss:<20.5f}"
    
    def get_vals(self):
        return tuple(self.__dict__.values())
    
    def get_names(self):
        return tuple(self.__dict__.keys())
        
    def reset(self):
        for key in self.__dict__.keys():
            self.__setattr__(key, 0.0)
    
    def update(self, epoch, loss, recon_loss, kl_loss):
        self.epoch = epoch
        self.loss = loss
        self.recon_loss = recon_loss
        self.kl_loss = kl_loss


class TensorBoardWriter:
    def __init__(self, path) -> None:
        self.patht = path
        self.writer = prepare_tensorboard(path)

    def add_dict_scaler(self, stat_dict: dict):
        epoch = stat_dict.pop('epoch')
        for key, val in stat_dict.items():
            self.writer.add_scalar(key, val, epoch)

    def add_dataclass_scaler(self, stat_dc):
        stat_dict = vars(stat_dc)
        self.add_dict_scaler(stat_dict)

    def close(self):
        self.writer.close()

    def flush(self):
        self.writer.flush()
        self.writer = None


class BasicLogger(object):
    def __init__(self, save):
        log_format = '%(message)s'
        logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
        fh = logging.FileHandler(os.path.join(save, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

    def info(self, string, *args):
        logging.info(string, *args)
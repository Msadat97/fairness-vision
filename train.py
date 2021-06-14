from collections import defaultdict
from pathlib import Path
from torch.optim.optimizer import Optimizer
from torch.serialization import load
from torch.utils.data import dataset

from torch.utils.data.dataloader import DataLoader
from models import BaseVAE
import numpy as np 
from tqdm.autonotebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from utils import prepare_tensorboard, save
from metrics import accuracy, balanced_accuracy
from running_mean import RunningMean
from torchmetrics import ConfusionMatrix

tensor = torch.Tensor


class VAETrainer(object):
    
    def __init__(self, 
                 model: BaseVAE, 
                 optimizer: Optimizer,
                 train_loader: DataLoader, 
                 val_loader: DataLoader = None,
                 multi_gpu: bool = False,
                 save_path: str = None,
                 use_mse: bool = False,
                 device=None,
                 beta=1) -> None:
         
        super(VAETrainer, self).__init__()
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vae = model
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, patience=5, mode='min', threshold=1e-2, factor=0.5)
        self.save_path = save_path
        
        # beta-vae parameters for loss function
        self.beta = beta
        self.use_mse = use_mse
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.multi_gpu = False
        if multi_gpu & (torch.cuda.device_count() > 1):
            self.multi_gpu = True

        self.train_stat = defaultdict(lambda: 0.0)
        self.val_stat = defaultdict(lambda: 0.0)
        

    def train(self, epochs):
        """
        Train the network
        """
        if self.multi_gpu:
            self.vae = nn.DataParallel(self.vae)
        
        tb = prepare_tensorboard('logs/run0')

        for epoch in range(1, epochs+1):
            
            train_loss_list = []
            train_kl_loss_list = []
            train_recon_loss_list = []
            loop = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    leave=True,
                    position=0,
                )
            loop.set_description("[train]")
            self.vae.train()
            current_data_size = 0
            for _, batch in loop:
                loss, recon_loss, kl_loss, batch_size = self.train_step(batch)
                current_data_size += batch_size
                train_loss_list.append(loss.data.item()*batch_size)
                train_recon_loss_list.append(recon_loss.item()*batch_size)
                train_kl_loss_list.append(kl_loss.item()*batch_size)
                training_loss = np.sum(train_loss_list)/current_data_size
                recon_loss = np.sum(train_recon_loss_list)/current_data_size
                kl_loss = np.sum(train_kl_loss_list)/current_data_size
                loop.set_description(
                    f"[train] "
                    f"epoch = {epoch}, "
                    f"loss = {training_loss:0.4f}, "
                    f"recon_loss = {recon_loss:0.4f}, "
                    f"kl_loss = {kl_loss:0.4f}, "
                )
            
            tb.add_scalar("train-loss", training_loss, epoch)
            tb.add_scalar("train-KL-loss", kl_loss, epoch)
            tb.add_scalar("train-recon-loss", recon_loss, epoch)
            
            training_loss = np.sum(train_loss_list)/current_data_size
            self.train_stat[f'Epoch {epoch}'] = training_loss.item()
            
            save(self.vae, self.save_path)
            
            loop = tqdm(
                enumerate(self.val_loader),
                total=len(self.val_loader),
                leave=True,
                position=0,
            )
            
            val_loss_list = []
            current_data_size = 0
            val_kl_loss_list = []
            val_recon_loss_list = []
            for _, batch in loop:
                loss, recon_loss, kl_loss, batch_size = self.train_step(batch)
                current_data_size += batch_size
                val_loss_list.append(loss.data.item()*batch_size)
                val_recon_loss_list.append(recon_loss.item()*batch_size)
                val_kl_loss_list.append(kl_loss.item()*batch_size)
                val_loss = np.sum(val_loss_list)/current_data_size
                recon_loss = np.sum(val_recon_loss_list)/current_data_size
                kl_loss = np.sum(val_kl_loss_list)/current_data_size
                loop.set_description(
                    f"[valid] "
                    f"epoch = {epoch}, "
                    f"loss = {val_loss:0.4f}, "
                    f"recon_loss = {recon_loss:0.4f}, "
                    f"kl_loss = {kl_loss:0.4f}, "
                )

            tb.add_scalar("val-loss", val_loss, epoch)
            tb.add_scalar("val-KL-loss", kl_loss, epoch)
            tb.add_scalar("val-recon-loss", recon_loss, epoch)
        
        tb.close()
        self.scheduler.step(val_loss)
    
    def run(self, split):
        
        train = True if split == 'train' else False
        loader = self.train_loader if train else self.val_loader
        
        self.vae.eval()
        if train:
            self.vae.train()
        
        loop = tqdm(
            enumerate(loader),
            total=len(loader),
            leave=True,
            position=0,
        )
        current_data_size = 0
        for _, batch in loop:
            if train:
                loss, batch_size = self.train_step(batch)
                


    def step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        batch_size = len(targets)

        x_hat, mu, log_var = self.vae(inputs)

        loss, recon_loss, kl_loss = self.vae_loss(
            torch.flatten(x_hat),
            torch.flatten(inputs),
            mu, log_var
        )
        
        return loss, recon_loss, kl_loss, batch_size
    
    def train_step(self, batch):
        
        loss, recon_loss, kl_loss, batch_size = self.step(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss, recon_loss, kl_loss, batch_size
    
    def validation_step(self, batch):
        return self.step(batch)
        
        
    def vae_loss(self, inputs: tensor, targets: tensor, mu: tensor, log_var: tensor):
    
        if self.use_mse:
            reconstruction_loss = F.mse_loss(inputs, targets, reduction='sum')
        else:
            reconstruction_loss = F.binary_cross_entropy(inputs, targets, reduction='sum')  
        
        batch_size = mu.shape[0]
        reconstruction_loss /= batch_size
        
        kl_loss = -0.5*(1 + log_var - mu.pow(2) - log_var.exp())
        kl_loss = kl_loss.sum(1).mean(0)
        loss = reconstruction_loss + self.beta * kl_loss
        
        return loss, reconstruction_loss, kl_loss
        

class ClassifierTrainer(object):

    def __init__(self,
                 model: nn.Module,
                 optimizer: torch.optim,
                 train_loader: DataLoader,
                 loss_fn,
                 val_loader: DataLoader = None,
                 multi_gpu: bool = False,
                 device: torch.device = None) -> None:

        super().__init__()
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_stat = defaultdict(lambda:0.0)
        self.val_stat = defaultdict(lambda:0.0)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.multi_gpu = False
        if multi_gpu & (torch.cuda.device_count() > 1):
            self.multi_gpu = True
            
        self.model.to(self.device)

    def train(self, epochs):
        """
        Train the network
        """
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        for epoch in range(1, epochs+1):
            self.train_one_epoch('train', epoch)
            if self.val_loader is not None:
                self.train_one_epoch('val', epoch)
    
    def train_one_epoch(self, split, epoch):
        
        loss_stat, acc_stat = RunningMean(), RunningMean()
        
        if split == 'train':
            loader = self.train_loader
        else:
            loader = self.val_loader
            
        loop = tqdm(
            enumerate(loader),
            total=len(loader),
            leave=True,
            position=0,
        )
        
        for _, batch in loop:
            if split == 'train':
                loss, acc = self.train_step(batch)
            else:
                loss, acc = self.val_step(batch)
            
            loss_stat.update(loss)
            acc_stat.update(acc)
            loop.set_description(
                f"[{split}] "
                f"epoch={epoch:d}, "
                f"loss={loss_stat.mean:.4f}, "
                f"acc={acc_stat.mean:.4f}"
            )
    
    def step(self, batch):
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        targets = targets.long()
        logits = self.model(inputs)
        loss = self.loss_fn(logits, targets)
        labels = logits.argmax(1)
        acc = balanced_accuracy(labels, targets)
        return loss, acc
    
    def train_step(self, batch):
        self.model.train()
        loss, acc = self.step(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item(), acc.item()
    
    @torch.no_grad()
    def val_step(self, batch):
        self.model.eval()
        loss, acc = self.step(batch)
        return loss.item(), acc.item()
    

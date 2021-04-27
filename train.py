from collections import defaultdict
import numpy as np 
from tqdm.autonotebook import tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import get_label, accuracy


def vae_loss(inputs, targets, mu, log_var):
    image_size = 28
    beta = 5
    reconstruction_loss = nn.BCELoss()(inputs, targets)
    reconstruction_loss *= image_size ** 2
    kl_loss = 1 + log_var - torch.square(mu) - torch.exp(log_var)
    kl_loss = -0.5 * torch.sum(kl_loss, dim=-1)
    return torch.mean(reconstruction_loss + beta*kl_loss)


class VAETrainer(object):
    
    def __init__(self, 
                 model, 
                 optimizer, 
                 loss_fn, 
                 train_loader, 
                 val_loader=None, 
                 multi_gpu=False,
                 device=None) -> None:
        
        super(VAETrainer, self).__init__()
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vae = model
        self.optimizer = optimizer
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=3, threshold=1e-2)
        self.loss_fn = loss_fn

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.multi_gpu = False
        if multi_gpu & (torch.cuda.device_count() > 1):
            self.multi_gpu = True

        self.train_stat = defaultdict(lambda: 0.0)
        self.val_stat = defaultdict(lambda: 0.0)

    def train(self, epochs=20):
        """
        Train the network
        """
        if self.multi_gpu:
            self.vae = nn.DataParallel(self.vae)
        
        for epoch in range(1, epochs+1):
            
            train_loss_list = []
            loop = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    leave=True,
                    position=0,
                )
            loop.set_description(f'Epoch: {epoch}/{epochs}')
            self.vae.train()
            for _, batch in loop:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                x_hat, mu, log_var = self.vae(inputs)
                    
                loss = self.loss_fn(torch.flatten(x_hat), torch.flatten(inputs), mu, log_var)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.data.item())

            training_loss = np.mean(train_loss_list)
            self.train_stat[f'Epoch {epoch}'] = training_loss.item()
            loop.write(f" Training loss:{training_loss:04f}")
            
            
            # self.scheduler.step(training_loss)
            
            # if self.val_loader is not None:
            #     valid_loss_list = []
            #     pred_list = []
            #     target_list = []
            #     for batch in self.val_loader:
            #         inputs, targets = batch
            #         inputs = inputs.to(self.device)
            #         output = self.model(inputs)
            #         targets = targets.to(self.device)
            #         loss = self.loss_fn(output,targets) 
            #         valid_loss_list.append(loss.data.item())
            #         predictions = predict(inputs, self.model)
            #         pred_list.append(predictions.flatten())
            #         target_list.append(targets.flatten())
                
            #     valid_loss = np.mean(valid_loss_list)
            #     self.val_stat[f'Epoch {epoch}'] = valid_loss
                
            #     loop.set_description(f'Epoch: {epoch}/{epochs}')
            #     loop.set_postfix(train_loss=training_loss.item(), valid_loss=valid_loss.item())
            #     # print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
            #     # valid_loss, accuracy(pred_list, target_list)))
            # else:
            #     loop.write(f" Training loss:{training_loss:04f}")
            #     # loop.set_postfix(train_loss=0.5)
            #     # loop.display(f'Loss: {training_loss}')

         
class LatentTrainer(object):
    
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader=None, multi_gpu=False, device=None) -> None:
        super(LatentTrainer, self).__init__()
        
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

    def train(self, epochs=20):
        """
        Train the network
        """
        if self.multi_gpu:
            self.model = nn.DataParallel(self.model)
        
        for epoch in range(1, epochs+1):

            train_loss_list = []
            train_target_list = []
            train_pred_list = []
            
            loop = tqdm(
                    enumerate(self.train_loader),
                    total=len(self.train_loader),
                    leave=True,
                    position=0,
                )
            
            self.model.train()
            current_data_size = 0

            for _, batch in loop:
                self.optimizer.zero_grad()
                inputs, targets = batch
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                targets = targets.type(torch.long)

                # for calculating the average loss
                input_size = len(inputs)
                current_data_size += input_size
                
                _, output = self.model(inputs)
                
                train_target_list.append(targets)
                train_pred_list.append(get_label(output))
                    
                loss = self.loss_fn(output, targets)
                loss.backward()
                self.optimizer.step()
                train_loss_list.append(loss.data.item()*input_size)
                current_acc = accuracy(train_target_list, train_pred_list)
                current_loss = np.sum(train_loss_list)/current_data_size

                loop.set_description(
                    f"[train] "
                    f"epoch={epoch:d}, "
                    f"loss = {current_loss:.4f}, "
                    f"acc = {current_acc:.4f}"
                )

            train_size = len(self.train_loader.dataset)
            training_acc = accuracy(train_pred_list, train_target_list)
            training_loss = np.sum(train_loss_list)/ train_size

            self.train_stat[f"Epoch {epoch}"] = training_loss.item()
            
            if self.val_loader is not None:
                
                valid_loss_list = []
                val_pred_list = []
                val_target_list = []

                for batch in self.val_loader:
                    
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    targets = targets.type(torch.long)
                    input_size = len(inputs)
                    
                    _, output = self.model(inputs)
                    
                    val_pred_list.append(get_label(output))
                    val_target_list.append(targets.flatten())
                    
                    loss = self.loss_fn(output, targets.type(torch.long)) 
                    valid_loss_list.append(loss.data.item()*input_size)

                val_size = len(self.val_loader.dataset)
                valid_acc = accuracy(val_pred_list, val_target_list)
                valid_loss = np.sum(valid_loss_list)/ val_size
                self.val_stat[f"Epoch {epoch}"] = valid_loss.item()
                
                loop.write(f" Test Loss: {valid_loss:04f}, Test Accuracy: {valid_acc:04f}")
                # loop.set_postfix(train_loss=training_loss.item(), valid_loss=valid_loss.item())
                # print('Epoch: {}, Training Loss: {:.2f}, Validation Loss: {:.2f}, accuracy = {:.2f}'.format(epoch, training_loss,
                # valid_loss, accuracy(pred_list, target_list)))
            else:
                loop.write(f" Training loss:{training_loss:04f}, Accuracy:{training_acc}")
                # loop.set_postfix(train_loss=0.5)
                # loop.display(f'Loss: {training_loss}')
    

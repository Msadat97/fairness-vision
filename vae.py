import matplotlib.pyplot as plt
import torch
from numpy.core.fromnumeric import alen
from torch import optim
from torch.serialization import save
from torch.utils import data
from torch.utils.data.dataloader import DataLoader
from torchinfo import summary
from torchvision.utils import make_grid, save_image

from datasets import CustomCelebA, VAEWrapper
from models import VAE
from celeba_models import CelebaVAE, AttributeClassifierCelebA
from train import ClassifierTrainer, VAETrainer
from utils import load, image_to_vid

from tqdm import trange

import numpy as np 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
#
# data = MnistLoader(shuffle=True, normalize=False, batch_size=128)
# clf = LatentEncoder()
# clf.to(device)
# optimizer = optim.RMSprop(clf.parameters(), lr=1e-3)
#
# trainer = LatentTrainer(clf,
#                         optimizer,
#                         loss_fn=nn.CrossEntropyLoss(),
#                         train_loader=data.train_loader,
#                         val_loader=data.test_loader)
#
# trainer.train(epochs=10)
# torch.save(clf.state_dict(), 'latent_state_dict_v1')

# def weights_init(m):
#     if isinstance(m, nn.Conv2d):
#         nn.init.xavier_normal_(m.weight)
#         nn.init.constant_(m.bias, 0.05)



def size_output_test():
    dataset = CustomCelebA(split='train')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))[0]
    vae = CelebaVAE(latent_dim=32, input_dim=dataset.image_size)
    print(vae.decode(vae.encode(data)[0]).shape, vae.shape)
    # print(vae)


def train_vae():
    batch_size = 64
    dataset = CustomCelebA(split='train')
    val_dataset = CustomCelebA(split='valid')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=32, pin_memory=True)
    vae = CelebaVAE(latent_dim=32, input_dim=dataset.image_size)
    vae.to(device)
    
    # img_size = dataset.image_size
    # print(vae)
    # summary(vae, input_size=(batch_size, *img_size))
    
    optimizer = optim.Adam(vae.parameters(), lr=2.5e-4)
    trainer = VAETrainer(vae, optimizer, train_loader=train_loader, val_loader=val_loader, 
                         beta=10, save_path='saved_models/celeba-vae-bce-v0')
    trainer.train(epochs=50)


def generate_random_image():
    dataset = CustomCelebA(split='train')
    vae = CelebaVAE(latent_dim=32, input_dim=dataset.image_size)
    vae.to(device)
    load(vae, 'saved_models/celeba-vae-bce-v0')
    vae.eval()
    index = np.random.randint(0, len(dataset))
    img = dataset[index][0].to(device)
    img = img[None, ...]
    
    z, _, _ = vae.encode(img)
    for i in range(0, 32):
        Z = z.clone()
        img = vae.decode(z)[0]
        save_image(img, 'test.png')
        comps = torch.linspace(-3, 3, 50)
        img_list = []
        for comp in comps:
            Z[:, i] = comp
            img = vae.decode(Z)[0]
            img_list.append(img)
        image_to_vid(img_list, path=f'./visualization-outputs/comp{i}.gif')

def generate_dataset():
    dset = CustomCelebA(split='train')
    img_size = dset.image_size
    vae = CelebaVAE(latent_dim=32, input_dim=img_size)
    vae.to(device)
    load(vae, 'saved_models/celeba-vae-bce-v0')
    vae.eval()
    
    for i in trange(10000):
        z = torch.randn(1, 32, device=device)
        img = vae.decode(z)
        save_image(img, f"./data/fid_test/fake/{i}.jpg")
        real, _ = dset[i]
        save_image(real, f"./data/fid_test/real/{i}.jpg")
    
def train_classifier():
    train_dset = CustomCelebA(split='train')
    val_dset = CustomCelebA(split='valid')
    
    img_size = train_dset.image_size
    batch_size = 100
    
    vae = CelebaVAE(latent_dim=32, input_dim=img_size)
    vae.to(device)
    load(vae, 'saved_models/celeba-vae-bce-v0')
    vae.eval()
    
    for param in vae.parameters():
        param.requires_grad_(True)
    
    # train_dset = VAEWrapper(dataset=train_dset, vae=vae)
    # val_dset = VAEWrapper(dataset=val_dset, vae=vae)
    
    train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True,
                              drop_last=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dset, batch_size=batch_size, shuffle=False,
                            drop_last=True, num_workers=32, pin_memory=True)
    
    model = AttributeClassifierCelebA(input_dim=img_size)
    
    # print(model)
    summary(model, input_size=(batch_size, *img_size))
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    trainer = ClassifierTrainer(model=model, 
                                optimizer=optimizer, 
                                loss_fn=loss_fn, 
                                train_loader=train_loader,
                                val_loader=val_loader)
    trainer.train(epochs=10)
    

def visualize_vae_outputs():
    train_dset = CustomCelebA(split='train')

    img_size = train_dset.image_size

    vae = CelebaVAE(latent_dim=32, input_dim=img_size)
    vae.to(device)
    load(vae, 'saved_models/celeba-vae-bce-v0')
    vae.eval()

    for param in vae.parameters():
        param.requires_grad_(True)
        
    train_dset = VAEWrapper(dataset=train_dset, vae=vae)
    train_loader = DataLoader(train_dset, batch_size=32, shuffle=False, drop_last=True)
    
    images, _= next(iter(train_loader))
    a = make_grid(images)
    save_image(a, 'test.png')

if __name__ == "__main__":
    # size_output_test()
    # generate_random_image()
    # train_vae()
    train_classifier()
    # visualize_vae_outputs()
    # generate_dataset()

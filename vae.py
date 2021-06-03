import matplotlib.pyplot as plt
from numpy.core.fromnumeric import alen
import torch
from torch import optim
from torch.serialization import save
from torch.utils import data
from torch.utils.data.dataloader import DataLoader

from datasets import CustomCelebA, VAEWrapper
from models import CelebaVAE, ClassifierCelebA, VAE
from train import ClassifierTrainer, VAETrainer
from torchvision.utils import save_image, make_grid
from torchvision import transforms
import numpy as np
import imageio

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


to_pil_image = transforms.ToPILImage()

def image_to_vid(images, name):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(f'outputs/{name}.gif', imgs)


def size_output_test():
    dataset = CustomCelebA(split='train')
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    data = next(iter(train_loader))[0]
    vae = CelebaVAE(latent_dim=32, input_dim=dataset.image_size)
    print(vae.decode(vae.encode(data)[0]).shape, vae.shape)
    # print(vae)


def train_vae():
    dataset = CustomCelebA(split='train')
    val_dataset = CustomCelebA(split='valid')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    vae = CelebaVAE(latent_dim=32, input_dim=dataset.image_size)
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=1e-4)
    trainer = VAETrainer(vae, optimizer, train_loader=train_loader, val_loader=val_loader, beta=10)
    trainer.train(epochs=100)


def generate_random_image():
    dataset = CustomCelebA(split='train')
    vae = CelebaVAE(latent_dim=32, input_dim=dataset.image_size)
    vae.to(device)
    vae.load_state_dict( 
        torch.load('saved_models/celeba-vae-state-dict-v4')
    )
    vae.eval()
    img = dataset[10][0].to(device)
    img = img[None, ...]
    z, _, _ = vae.encode(img)
    Z = z.clone()
    img = vae.decode(z)[0]
    save_image(img, 'test.png')
    comps = torch.linspace(-3, 3, 50)
    img_list = []
    for comp in comps:
        Z[:, 1] = comp
        img = vae.decode(Z)[0]
        img_list.append(img)
    image_to_vid(img_list, 'test')


def train_classifier():
    train_dset = CustomCelebA(split='train')
    val_dset = CustomCelebA(split='valid')
    
    img_size = train_dset.image_size
    
    # vae = CelebaVAE(latent_dim=32, input_dim=img_size)
    # vae.to(device)
    # vae.load_state_dict(
    #     torch.load('saved_models/celeba-vae-state-dict-v4')
    # )
    # vae.eval()
    
    # for param in vae.parameters():
    #     param.requires_grad_(True)
    
    # train_dset = VAEWrapper(dataset=train_dset, vae=vae)
    # val_dset = VAEWrapper(dataset=val_dset, vae=vae)
    
    train_loader = DataLoader(train_dset, batch_size=100, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dset, batch_size=100, shuffle=False, drop_last=True)
    
    model = ClassifierCelebA(input_dim=img_size)
    
    print(model)
    # summary(model, input_size=train_dset.image_size)
    
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # trainer = ClassifierTrainer(model=model, 
    #                             optimizer=optimizer, 
    #                             loss_fn=loss_fn, 
    #                             train_loader=train_loader,
    #                             val_loader=val_loader)
    # trainer.train(epochs=10)
    

def visualize_vae_outputs():
    train_dset = CustomCelebA(split='train')

    img_size = train_dset.image_size

    vae = CelebaVAE(latent_dim=32, input_dim=img_size)
    vae.to(device)
    vae.load_state_dict(
        torch.load('saved_models/celeba-vae-state-dict-v4')
    )
    vae.eval()

    for param in vae.parameters():
        param.requires_grad_(True)
        
    # train_dset = VAEWrapper(dataset=train_dset, vae=vae)
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

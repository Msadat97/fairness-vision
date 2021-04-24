import torch
from torch import nn
import torchvision
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from torch.utils.data import TensorDataset
from torch import optim
from mnist import MnistLoader
from model import VAE, LatentEncoder, concat
from train import VAETrainer, vae_loss
import matplotlib.pyplot as plt
import numpy as np
import imageio
import torchvision.transforms as transforms
from utils import get_label
import pandas as pd
from attack import Attack


to_pil_image = transforms.ToPILImage()


def image_to_vid(images, name):
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave(f'outputs/{name}.gif', imgs)


data = MnistLoader(batch_size=128, shuffle=True, normalize=False)
vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load('saved_models/vae_state_dict_v1'))
vae.cuda()


clf = LatentEncoder()
clf.load_state_dict(torch.load('saved_models/latent_state_dict_v1'))
clf.cuda()


vae.eval()
x_train = data.get_data('x_train').cuda()
z, _, _ = vae.encoder_forward(x_train[18:19])
im = vae.decoder_forward(z).detach().cpu()
im_to_show = im.numpy()[0, 0]
plt.imshow(im_to_show)
plt.show()


features, _ = clf.forward(im.cuda())


model = concat(vae, clf)

attack = Attack(model=model,
                loss_fn=nn.MSELoss(),
                eps=0.08,
                device='cuda')

adv = attack.pgd_attack(alpha=0.5, 
                        input_vec=z, 
                        targets=features.detach(), 
                        iterations=10, 
                        num_restarts=1, 
                        random_start=True)
# print(adv - z)
# adv = attack.fgsm_attack(z.detach(),  features.detach())

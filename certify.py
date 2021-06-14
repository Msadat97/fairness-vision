from datasets import CustomCelebA, VAEWrapper
from celeba_models import AutoEncoderCelebA, CelebaVAE, ClassifierCelebA, EncoderCelebA
from time import time
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnist import MnistLoader
from lcifr.code.attacks import PGD
from models import VAE, LatentClassifier, AutoEncoder, LatentEncoder, DataModel
from dl2.training.supervised.oracles import DL2_Oracle
from lcifr.code.experiments.args_factory import get_args
from lcifr.code.constraints.general_categorical_constraint import SegmentConstraint
from utils import get_latents, load, prepare_config
from attack import SegmentPDG
from metrics import standard_accuracy, robust_accuracy, smoothing_accuracy
from running_mean import RunningMean

config = prepare_config('./metadata.json')
vae_path = config["celeba_save_path"]['vae']
ae_path = config["celeba_save_path"]['lcifr_autoencoder']
classifier_path = config["celeba_save_path"]['robust_classifier']

delta = config['lcifr_experiment']['delta']
epsilon = config['lcifr_experiment']['epsilon']
latent_index = config['lcifr_experiment']['latent_index']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# vae = VAE(latent_dim=16)
# vae.load_state_dict(
#     torch.load(
#         vae_path,
#         map_location=torch.device('cpu')
#     )
# )
# vae.to(device)

# latent_encoder = LatentEncoder()
# latent_encoder.to(device)

# autoencoder = AutoEncoder(vae, latent_encoder)
# autoencoder.load_state_dict(
#     torch.load(
#         ae_path,
#         map_location=lambda storage, loc: storage
#     )
# )
# autoencoder.to(device)

# classifier = LatentClassifier(
#     latent_encoder.flatten_shape, latent_encoder.num_classes)
# classifier.load_state_dict(
#     torch.load(
#         classifier_path, 
#         map_location=torch.device('cpu')
#     )
# )
# classifier.to(device)

latent_dim = config['vae_experiment']['latent_dim']
input_dim = config['celeba']['input_dim']

vae = CelebaVAE(latent_dim=latent_dim, input_dim=input_dim)
load(vae, vae_path)
vae.to(device)

latent_encoder = EncoderCelebA(input_dim=input_dim)
latent_encoder.to(device)

autoencoder = AutoEncoderCelebA(vae, latent_encoder)
load(autoencoder, ae_path)
autoencoder.to(device)

classifier = ClassifierCelebA(1024)
load(classifier, classifier_path)
classifier.to(device)


data_model = DataModel(autoencoder, classifier)
data_model.freeze()

# data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)
# train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.test_loader
# train_loader = get_latents(vae=vae, data_loader=train_loader, shuffle=True, device=device)
# val_loader = get_latents(vae=vae, data_loader=val_loader, shuffle=False, device=device)
# test_loader = get_latents(vae=vae, data_loader=test_loader, shuffle=False, device=device)

train_data, val_data = CustomCelebA(split='train'), CustomCelebA(split='valid')
train_data = VAEWrapper(vae=vae, dataset=train_data)
val_data = VAEWrapper(vae=vae, dataset=val_data)

batch_size = config['lcifr_experiment']['batch_size']
num_workers = config['lcifr_experiment']['num_workers']

train_loader = DataLoader(train_data, batch_size=batch_size,
                          num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False, drop_last=True, pin_memory=True)

# robust = robust_accuracy(data_model, val_loader, epsilon, latent_index)
# acc = standard_accuracy(data_model,  val_loader)
# print(f'accuracy = {acc}')
# print(f'robust-accuracy = {robust}')

print(smoothing_accuracy(data_model, epsilon, val_loader, latent_index))

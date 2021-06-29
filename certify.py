import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm

from args import get_arguments
from attack import SegmentPDG
from celeba_models import AutoEncoderCelebA, CelebaVAE, ClassifierCelebA, EncoderCelebA
from datasets import CustomCelebA, VAEWrapper
from lcifr.code.constraints.general_categorical_constraint import SegmentConstraint
from lcifr.code.experiments.args_factory import get_args
from metrics import robust_accuracy, smoothing_accuracy, standard_accuracy
from mnist import MnistLoader
from models import VAE, AutoEncoder, DataModel, LatentClassifier, LatentEncoder
from utils import get_latents, load, prepare_config

my_args = get_arguments()
    
config = prepare_config('./metadata.json')
vae_path = config["celeba_save_path"]['vae']

if my_args.robust:
    ae_path = config["celeba_save_path"]['lcifr_autoencoder']
    classifier_path = config["celeba_save_path"]['robust_classifier']
    
else:
    ae_path = config["celeba_save_path"]['base_autoencoder']
    classifier_path = config["celeba_save_path"]['base_classifier']



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


acc = standard_accuracy(data_model,  val_loader)
robust = robust_accuracy(data_model, val_loader, epsilon, latent_index)
smooth = smoothing_accuracy(data_model, epsilon, val_loader, latent_index)

print(f'accuracy = {acc["acc"]}\nbalanced_acc = {acc["balanced_acc"]}')
print(f'robust_accuracy = {robust}')
print(f'smoothed_accuracy = {smooth["acc"]}\nsmoothed_certified = {smooth["certified"]}')


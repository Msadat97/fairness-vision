from pathlib import Path
from torch.functional import split
from torch.utils.data.dataloader import DataLoader
from datasets import CustomCelebA, VAEWrapper
import numpy as np
import seaborn as sns
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dl2.training.supervised.oracles import DL2_Oracle
from lcifr.code.constraints import GeneralCategoricalConstraint
from lcifr.code.constraints.general_categorical_constraint import \
    SegmentConstraint
from lcifr.code.experiments.args_factory import get_args
from lcifr.code.utils.statistics import Statistics
from metrics import accuracy, balanced_accuracy
from mnist import MnistLoader
from models import Encoder, VAE, AutoEncoder, LatentClassifier, LatentEncoder
from celeba_models import CelebaVAE, ClassifierCelebA, EncoderCelebA, AutoEncoderCelebA
from utils import get_latents, get_logdir, load, prepare_tensorboard, prepare_config


config = prepare_config('./metadata.json')
vae_path = config["celeba_save_path"]['vae']


# parameters
lr = config['lcifr_experiment']['learning_rate']
dl2_lr = config['lcifr_experiment']['dl2_lr']
patience = config['lcifr_experiment']['patience']
weight_decay = config['lcifr_experiment']['weight_decay']
dl2_iters = config['lcifr_experiment']['dl2_iters']
dl2_weight = config['lcifr_experiment']['dl2_weight']
num_epochs = config['lcifr_experiment']['num_epochs_encoder']
base_run = config['lcifr_experiment']['base_run']
args = get_args()

if base_run:
    ae_path = config["celeba_save_path"]['base_autoencoder']
else:
    ae_path = config["celeba_save_path"]['lcifr_autoencoder']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

latent_dim = config['vae_experiment']['latent_dim']
input_dim = config['celeba']['input_dim']
vae = CelebaVAE(latent_dim=latent_dim, input_dim=input_dim)
load(vae, vae_path)
vae.to(device)

latent_encoder = EncoderCelebA(input_dim=input_dim)
latent_encoder.to(device)

autoencoder = AutoEncoderCelebA(vae, latent_encoder)
autoencoder.to(device)

classifier = ClassifierCelebA(1024)
classifier.to(device)

# data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)


batch_size = config['lcifr_experiment']['batch_size']
num_workers = config['lcifr_experiment']['num_workers']

train_data, val_data = CustomCelebA(split='train'), CustomCelebA(split='valid')
train_data = VAEWrapper(vae=vae, dataset=train_data)
val_data = VAEWrapper(vae=vae, dataset=val_data)

train_loader = DataLoader(train_data, batch_size=batch_size,
                          num_workers=num_workers, shuffle=True, drop_last=True, pin_memory=True)
val_loader = DataLoader(val_data, batch_size=batch_size,
                        num_workers=num_workers, shuffle=False, drop_last=True, pin_memory=True)
# train_loader, val_loader = data.train_loader, data.val_loader
# train_loader = get_latents(vae=vae, data_loader=train_loader, shuffle=True, device=device)
# val_loader = get_latents(vae=vae, data_loader=val_loader, shuffle=False, device=device)

delta = config['lcifr_experiment']['delta']
epsilon = config['lcifr_experiment']['epsilon']
latent_index = config['lcifr_experiment']['latent_index']

constraint = SegmentConstraint(model=autoencoder, delta=delta, epsilon=epsilon, latent_idx=latent_index)
oracle = DL2_Oracle(
    learning_rate=dl2_lr, net=autoencoder,
    use_cuda=torch.cuda.is_available(),
    constraint=constraint
)

cre_loss = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(autoencoder.parameters()) + list(classifier.parameters()),
    lr=lr, weight_decay=weight_decay
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=patience, factor=0.5
)


def run(autoencoder, classifier, optimizer, loader, split, epoch):

    if split == 'train':
        autoencoder.train()
        classifier.train()
    
    predictions, targets, l_inf_diffs = list(), list(), list()
    constrain_satisfactinos = []
    tot_mix_loss, tot_ce_loss, tot_dl2_loss = Statistics.get_stats(3)

    progress_bar = tqdm(loader)
    for data_batch, targets_batch in progress_bar:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.to(device)
        targets_batch = targets_batch.to(device)
        targets_batch = targets_batch.long()

        x_batches, y_batches = list(), list()
        assert batch_size % oracle.constraint.n_tvars == 0
        k = batch_size // oracle.constraint.n_tvars

        for i in range(oracle.constraint.n_tvars):
            x_batches.append(data_batch[i: i + k])
            y_batches.append(targets_batch[i: i + k])
        
        
        latent_data = autoencoder.encode(data_batch)

        logits = classifier(latent_data)

        cross_entropy = cre_loss(logits, targets_batch)
        predictions_batch = classifier.predict(latent_data)

        predictions.append(predictions_batch.detach().cpu())
        targets.append(targets_batch.detach().cpu())
        
        if base_run:
            z_batches = x_batches
        else:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(
                x_batches, y_batches, domains, num_restarts=1,
                num_iters=dl2_iters, args=args
            )

        latent_rep = autoencoder.encode(z_batches[0]).detach()
        l_inf_diffs.append(
            torch.abs(latent_data - latent_rep).max(1)[0].detach().cpu()
        )
        
        if base_run:
            dl2_loss, sat = torch.tensor([0.0]).cuda(), np.array([0])
        else:
            _, dl2_loss, sat = oracle.evaluate(
                x_batches, y_batches, z_batches, args=args
            )
        
        constrain_satisfactinos.append(sat)

        mix_loss = torch.mean(
            cross_entropy + dl2_weight * dl2_loss
        )

        if split == 'train':
            optimizer.zero_grad()
            mix_loss.backward()
            optimizer.step()

        tot_ce_loss.add(cross_entropy.mean().item())
        tot_dl2_loss.add(dl2_loss.mean().item())
        tot_mix_loss.add(mix_loss.mean().item())
        acc = balanced_accuracy(predictions, targets)
        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, ce_loss={tot_ce_loss.mean():.4f}, '
            f'dl2_loss={tot_dl2_loss.mean():.4f}, '
            f'mix_loss={tot_mix_loss.mean():.4f}, '
            f'acc = {acc:.4f}, ' 
            # f'sat = {np.mean(sat):.4f}'
        )
    
    l_inf_diffs = torch.cat(l_inf_diffs)
    satisfaction = np.mean(np.concatenate(constrain_satisfactinos))
    print(f'[{split}]: satisfaction_rate = {satisfaction}')
    writer.add_scalar('Loss/%s' % split, tot_mix_loss.mean(), epoch)
    writer.add_scalar('DL2 Loss/%s' % split, tot_dl2_loss.mean(), epoch)
    writer.add_scalar('Cross Entropy/%s' % split, tot_ce_loss.mean(), epoch)
    writer.add_histogram('L-inf Differences/%s' %split, l_inf_diffs, epoch)
        
    return tot_mix_loss


log_dir = get_logdir(config)
writer = prepare_tensorboard(log_dir)

for epoch in range(num_epochs):
    
    run(autoencoder, classifier, optimizer, train_loader, 'train', epoch)

    autoencoder.eval()
    classifier.eval()
    
    valid_mix_loss = run(autoencoder, classifier, optimizer, val_loader, 'valid', epoch)
    scheduler.step(valid_mix_loss.mean())

    torch.save(
        autoencoder.state_dict(), ae_path
    )
    # torch.save(
    #     classifier.state_dict(),
    #     path.join(models_dir, f'classifier_{epoch}.pt')
    # )
    
writer.close()

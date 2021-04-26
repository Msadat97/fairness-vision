from datetime import datetime
from dl2.querying.models.cifar import models
from os import makedirs, path
from argparse import ArgumentParser
from utils import accuracy, predict
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.utils.data
from lcifr.code.experiments.args_factory import get_args
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mnist import MnistLoader

from lcifr.code.constraints import GeneralCategoricalConstraint
from dl2.training.supervised.oracles import DL2_Oracle
from model import VAE, LatentEncoder, AutoEncoder, LatentClassifier
from lcifr.code.utils.statistics import Statistics


def get_latents(vae: VAE, data_loader, device='cpu'):
    z_list = []
    y_list = []
    for batch in data_loader:
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        z_batch, _, _ = vae.encoder_forward(inputs)
        z_list.append(z_batch)
        y_list.append(targets)
    z_tensor = torch.cat(z_list)
    y_tensor = torch.cat(y_list)
    data_set = TensorDataset(z_tensor, y_tensor)
    batch_size = data_loader.batch_size
    return DataLoader(data_set, batch_size=batch_size)


# parameters
lr = 0.01
dl2_lr = 1.0
patience = 5
weight_decay = 0.01
dl2_iters = 25
dl2_weight = 5.0
dec_weight = 0.0
num_epochs = 10
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load(
    'saved_models/vae_state_dict_v1', map_location=torch.device('cpu')))
vae.to(device)

latent_encoder = LatentEncoder()
latent_encoder.to(device)

autoencoder = AutoEncoder(vae, latent_encoder)
autoencoder.to(device)

classifier = LatentClassifier(latent_encoder.flatten_shape, latent_encoder.num_classes)
classifier.to(device)

data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)
train_loader, val_loader = data.train_loader, data.val_loader
train_loader, val_loader = get_latents(vae, train_loader, device), get_latents(vae, val_loader, device)

constraint = GeneralCategoricalConstraint(model=autoencoder, delta=0.01, epsilon=0.3)
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
    predictions, targets, l_inf_diffs = list(), list(), list()
    tot_mix_loss, tot_ce_loss, tot_dl2_loss = Statistics.get_stats(3)

    progress_bar = tqdm(loader)
    target_list = []
    pred_list = []
    for data_batch, targets_batch in progress_bar:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.to(device)
        targets_batch = targets_batch.to(device)
        targets_batch = targets_batch.type(torch.long)
        target_list.append(targets_batch)

        x_batches, y_batches = list(), list()
        assert batch_size % oracle.constraint.n_tvars == 0
        k = batch_size // oracle.constraint.n_tvars

        for i in range(oracle.constraint.n_tvars):
            x_batches.append(data_batch[i: i + k])
            y_batches.append(targets_batch[i: i + k])

        if split == 'train':
            autoencoder.train()
            classifier.train()

        latent_data = autoencoder.encode(data_batch)

        data_batch_dec = autoencoder.decode(latent_data)
        l2_loss = torch.norm(data_batch_dec - data_batch, dim=1)

        logits = classifier(latent_data)
        pred_list.append(classifier.predict(logits))

        cross_entropy = cre_loss(logits, targets_batch)
        predictions_batch = classifier.predict(latent_data)

        predictions.append(predictions_batch.detach().cpu())
        targets.append(targets_batch.detach().cpu())

        autoencoder.eval()
        classifier.eval()

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(
                x_batches, y_batches, domains, num_restarts=1,
                num_iters=dl2_iters, args=args
            )
        else:
            z_batches = None

        latent_adv = autoencoder.encode(z_batches[0]).detach()
        l_inf_diffs.append(
            torch.abs(latent_data - latent_adv).max(1)[0].detach().cpu()
        )

        if split == 'train':
            autoencoder.train()
            classifier.train()

        _, dl2_loss, _ = oracle.evaluate(
            x_batches, y_batches, z_batches, args=args
        )
        mix_loss = torch.mean(
            cross_entropy + dl2_weight * dl2_loss +
            dec_weight * l2_loss
        )

        if split == 'train':
            optimizer.zero_grad()
            mix_loss.backward()
            optimizer.step()

        tot_ce_loss.add(cross_entropy.mean().item())
        tot_dl2_loss.add(dl2_loss.mean().item())
        tot_mix_loss.add(mix_loss.mean().item())
        print(accuracy(pred_list, target_list))
        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, ce_loss={tot_ce_loss.mean():.4f}, '
            f'dl2_loss={tot_dl2_loss.mean():.4f}, '
            f'mix_loss={tot_mix_loss.mean():.4f}'
        )
    return tot_mix_loss


for epoch in range(num_epochs):
    
    run(autoencoder, classifier, optimizer, train_loader, 'train', epoch)

    autoencoder.eval()
    classifier.eval()
    valid_mix_loss = run(
        autoencoder, classifier, optimizer, val_loader, 'valid', epoch
    )
    scheduler.step(valid_mix_loss.mean())

    torch.save(
        autoencoder.state_dict(), 'saved_models/vae_lcifr_trained'
    )
    # torch.save(
    #     classifier.state_dict(),
    #     path.join(models_dir, f'classifier_{epoch}.pt')
    # )

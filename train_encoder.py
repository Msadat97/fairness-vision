from datetime import datetime
from os import makedirs, path

import torch.nn as nn
import torch.utils.data
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from mnist import MnistLoader

from lcifr.code.constraints import ConstraintBuilder
from dl2.training.supervised.oracles import DL2_Oracle
from lcifr.code.experiments.args_factory import get_args
from lcifr.code.metrics import equalized_odds, statistical_parity
from model import VAE, LatentEncoder, AutoEncoder, LatentClassifier
from utils import Statistics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)

train_loader, val_loader = data.train_loader, data.val_loader

vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load('saved_models/vae_state_dict_v1'))

latent_encoder = LatentEncoder()
autoencoder = AutoEncoder(vae, latent_encoder)
classifier = LatentClassifier(latent_encoder.flatten_shape, latent_encoder.num_labels)

oracle = DL2_Oracle(
    learning_rate=1.0, net=autoencoder,
    use_cuda=torch.cuda.is_available(),
    constraint=ConstraintBuilder.build(
        autoencoder, data.train_data, args.constraint
    )
)

cross_entropy = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    list(autoencoder.parameters()) + list(classifier.parameters()),
    lr=0.01, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=args.patience, factor=0.5
)


def run(autoencoder, classifier, optimizer, loader, split, epoch):
    predictions, targets, l_inf_diffs = list(), list(), list()
    tot_mix_loss, tot_ce_loss, tot_dl2_loss = Statistics.get_stats(3)
    tot_l2_loss, tot_stat_par, tot_eq_odds = Statistics.get_stats(3)

    progress_bar = tqdm(loader)

    for data_batch, targets_batch, protected_batch in progress_bar:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.to(device)
        targets_batch = targets_batch.to(device)
        protected_batch = protected_batch.to(device)

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
        cross_entropy = binary_cross_entropy(logits, targets_batch)
        predictions_batch = classifier.predict(latent_data)

        stat_par = statistical_parity(predictions_batch, protected_batch)
        eq_odds = equalized_odds(
            targets_batch, predictions_batch, protected_batch
        )

        predictions.append(predictions_batch.detach().cpu())
        targets.append(targets_batch.detach().cpu())

        autoencoder.eval()
        classifier.eval()

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(
                x_batches, y_batches, domains, num_restarts=1,
                num_iters=args.dl2_iters, args=args
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
            x_batches, y_batches, z_batches, args
        )
        mix_loss = torch.mean(
            cross_entropy + args.dl2_weight * dl2_loss +
            args.dec_weight * l2_loss
        )

        if split == 'train':
            optimizer.zero_grad()
            mix_loss.backward()
            optimizer.step()

        tot_ce_loss.add(cross_entropy.mean().item())
        tot_dl2_loss.add(dl2_loss.mean().item())
        tot_mix_loss.add(mix_loss.mean().item())
        tot_l2_loss.add(l2_loss.mean().item())
        tot_stat_par.add(stat_par.mean().item())
        tot_eq_odds.add(eq_odds.mean().item())

        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, ce_loss={tot_ce_loss.mean():.4f}, '
            f'dl2_loss={tot_dl2_loss.mean():.4f}, '
            f'mix_loss={tot_mix_loss.mean():.4f}'
        )

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    l_inf_diffs = torch.cat(l_inf_diffs)

    accuracy = accuracy_score(targets, predictions)
    balanced_accuracy = balanced_accuracy_score(targets, predictions)
    tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    f1 = f1_score(targets, predictions)

    writer.add_scalar('Accuracy/%s' % split, accuracy, epoch)
    writer.add_scalar('Balanced Accuracy/%s' % split, balanced_accuracy, epoch)
    writer.add_scalar('Cross Entropy/%s' % split, tot_ce_loss.mean(), epoch)
    writer.add_scalar('Decoder Loss/%s' % split, tot_l2_loss.mean(), epoch)
    writer.add_scalar('DL2 Loss/%s' % split, tot_dl2_loss.mean(), epoch)
    writer.add_scalar('Loss/%s' % split, tot_mix_loss.mean(), epoch)
    writer.add_scalar('True Positives/%s' % split, tp, epoch)
    writer.add_scalar('False Negatives/%s' % split, fn, epoch)
    writer.add_scalar('True Negatives/%s' % split, tn, epoch)
    writer.add_scalar('False Positives/%s' % split, fp, epoch)
    writer.add_scalar('F1 Score/%s' % split, f1, epoch)
    writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    writer.add_scalar('Stat. Parity/%s' % split, tot_stat_par.mean(), epoch)
    writer.add_scalar('Equalized Odds/%s' % split, tot_eq_odds.mean(), epoch)
    writer.add_histogram('L-inf Differences/%s' % split, l_inf_diffs, epoch)

    return tot_mix_loss


print('saving model to', models_dir)
writer = SummaryWriter(log_dir)

for epoch in range(args.num_epochs):
    run(autoencoder, classifier, optimizer, train_loader, 'train', epoch)

    autoencoder.eval()
    classifier.eval()
    valid_mix_loss = run(
        autoencoder, classifier, optimizer, val_loader, 'valid', epoch
    )
    scheduler.step(valid_mix_loss.mean())

    torch.save(
        autoencoder.state_dict(),
        path.join(models_dir, f'autoencoder_{epoch}.pt')
    )
    torch.save(
        classifier.state_dict(),
        path.join(models_dir, f'classifier_{epoch}.pt')
    )

writer.close()

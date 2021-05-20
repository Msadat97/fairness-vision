from matplotlib import pyplot as plt
from lcifr.code.constraints.general_categorical_constraint import SegmentConstraint
import torch.nn as nn
import torch.utils.data
from lcifr.code.experiments.args_factory import get_args
from tqdm import tqdm
from mnist import MnistLoader
from lcifr.code.constraints import GeneralCategoricalConstraint
from dl2.training.supervised.oracles import DL2_Oracle
from models import VAE, LatentEncoder, AutoEncoder, LatentClassifier
from lcifr.code.utils.statistics import Statistics
from utils import get_latents
from metrics import accuracy
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import numpy as np

vae_path = "saved_models/vae-state-dict-v3"
ae_path = "saved_models/vae-lcifr-trained-v7"

# parameters
lr = 1e-3
dl2_lr = 0.005
patience = 5
weight_decay = 0.01
dl2_iters = 25
dl2_weight = 50
dec_weight = 0.0
num_epochs = 50
args = get_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(latent_dim=16)
vae.load_state_dict(torch.load(
    vae_path, map_location=torch.device('cpu')))
vae.to(device)

latent_encoder = LatentEncoder()
latent_encoder.to(device)

autoencoder = AutoEncoder(vae, latent_encoder)
autoencoder.to(device)

classifier = LatentClassifier(latent_encoder.flatten_shape, latent_encoder.num_classes)
classifier.to(device)

data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)
train_loader, val_loader = data.train_loader, data.val_loader
train_loader = get_latents(vae=vae, data_loader=train_loader, shuffle=True, device=device)
val_loader = get_latents(vae=vae, data_loader=val_loader, shuffle=False, device=device)

constraint = SegmentConstraint(model=autoencoder, delta=0.005, epsilon=0.5, latent_idx=5)
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
    
    autoencoder.train()
    classifier.train()

    # if split == 'train':
    #     autoencoder.train()
    #     classifier.train()
        
    predictions, targets, l_inf_diffs = list(), list(), list()
    constrain_satisfactinos = []
    tot_mix_loss, tot_ce_loss, tot_dl2_loss = Statistics.get_stats(3)

    progress_bar = tqdm(loader)
    for data_batch, targets_batch in progress_bar:
        batch_size = data_batch.shape[0]
        data_batch = data_batch.to(device)
        targets_batch = targets_batch.to(device)
        targets_batch = targets_batch.type(torch.long)

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

        cross_entropy = cre_loss(logits, targets_batch)
        predictions_batch = classifier.predict(latent_data)

        predictions.append(predictions_batch.detach().cpu())
        targets.append(targets_batch.detach().cpu())

        if oracle.constraint.n_gvars > 0:
            domains = oracle.constraint.get_domains(x_batches, y_batches)
            z_batches = oracle.general_attack(
                x_batches, y_batches, domains, num_restarts=1,
                num_iters=dl2_iters, args=args
            )
        else:
            z_batches = None

        latent_rep = autoencoder.encode(z_batches[0]).detach()
        l_inf_diffs.append(
            torch.abs(latent_data - latent_rep).max(1)[0].detach().cpu()
        )

        _, dl2_loss, sat = oracle.evaluate(
            x_batches, y_batches, z_batches, args=args
        )
        
        constrain_satisfactinos.append(sat)
        # dl2_loss = 0.0
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
        acc = accuracy(predictions, targets)
        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, ce_loss={tot_ce_loss.mean():.4f}, '
            f'dl2_loss={tot_dl2_loss.mean():.4f}, '
            f'mix_loss={tot_mix_loss.mean():.4f}, '
            f'acc = {acc:.4f}, ' 
            f'sat = {np.mean(sat):.4f}'
        )
    
    l_inf_diffs = torch.cat(l_inf_diffs)
    satisfaction = np.mean(np.concatenate(constrain_satisfactinos))
    print(f'[{split}]: satisfaction_rate = {satisfaction}')
    writer.add_scalar('Loss/%s' % split, tot_mix_loss.mean(), epoch)
    writer.add_scalar('DL2 Loss/%s' % split, tot_dl2_loss.mean(), epoch)
    writer.add_scalar('Cross Entropy/%s' % split, tot_ce_loss.mean(), epoch)
    writer.add_histogram('L-inf Differences/%s' %split, l_inf_diffs, epoch)
        
    return tot_mix_loss

log_dir = "logs/"
writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    
    run(autoencoder, classifier, optimizer, val_loader, 'train', epoch)

    autoencoder.eval()
    classifier.eval()
    valid_mix_loss = run(
        autoencoder, classifier, optimizer, val_loader, 'valid', epoch
    )
    scheduler.step(valid_mix_loss.mean())

    torch.save(
        autoencoder.state_dict(), ae_path
    )
    # torch.save(
    #     classifier.state_dict(),
    #     path.join(models_dir, f'classifier_{epoch}.pt')
    # )
writer.close()

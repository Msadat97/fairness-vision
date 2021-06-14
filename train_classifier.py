import numpy as np
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from mnist import MnistLoader
from lcifr.code.attacks import PGD
from models import VAE, LatentClassifier, AutoEncoder, LatentEncoder
from dl2.training.supervised.oracles import DL2_Oracle
from lcifr.code.experiments.args_factory import get_args
from lcifr.code.constraints.general_categorical_constraint import GeneralCategoricalConstraint, SegmentConstraint
from utils import load, prepare_config
from attack import SegmentPDG
from celeba_models import CelebaVAE, EncoderCelebA, ClassifierCelebA, AutoEncoderCelebA
from datasets import CustomCelebA, VAEWrapper
from torch.utils.data import DataLoader
from metrics import accuracy, balanced_accuracy
import seaborn as sns
import matplotlib.pyplot as plt

config = prepare_config('./metadata.json')
vae_path = config["celeba_save_path"]['vae']
ae_path = config["celeba_save_path"]['lcifr_autoencoder']
classifier_path = config["celeba_save_path"]['robust_classifier']

# parameters
# parameters
lr = config['lcifr_experiment']['learning_rate']
dl2_lr = config['lcifr_experiment']['dl2_lr']
patience = config['lcifr_experiment']['patience']
weight_decay = config['lcifr_experiment']['weight_decay']
dl2_iters = config['lcifr_experiment']['dl2_iters']
dl2_weight = config['lcifr_experiment']['dl2_weight']
num_epochs = config['lcifr_experiment']['num_epochs_classifier']
args = get_args()
args.adversarial = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# vae = CelebaVAE(latent_dim=16)
# vae.load_state_dict(
#     torch.load(
#         vae_path, map_location=torch.device('cpu')
#         )
#     )
# vae.to(device)

# latent_encoder = EncoderCelebA()
# latent_encoder.to(device)

# autoencoder = AutoEncoderCelebA(vae, latent_encoder)
# autoencoder.to(device)

# classifier = ClassifierCelebA(
#     latent_encoder.flatten_shape, latent_encoder.num_classes)
# classifier.to(device)

# for param in autoencoder.parameters():
#     param.requires_grad_(False)

# autoencoder.load_state_dict(
#     torch.load(
#         ae_path,
#         map_location=lambda storage, loc: storage
#     )
# )

# data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)

# train_loader, val_loader = data.train_loader, data.val_loader
# train_loader = get_latents(vae=vae, data_loader=train_loader, shuffle=True, device=device)
# val_loader = get_latents(vae=vae, data_loader=val_loader, shuffle=False, device=device)

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

delta = config['lcifr_experiment']['delta']
epsilon = config['lcifr_experiment']['epsilon']
latent_index = config['lcifr_experiment']['latent_index']

constraint = GeneralCategoricalConstraint(model=autoencoder, delta=delta, epsilon=epsilon)
oracle = DL2_Oracle(
    learning_rate=args.dl2_lr, net=autoencoder,
    use_cuda=torch.cuda.is_available(),
    constraint=constraint
)

cre_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    classifier.parameters(), lr=lr,
    weight_decay=weight_decay
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=patience, factor=0.5, threshold=0.005
)


def run(autoencoder, classifier, optimizer, loader, split):
    predictions, targets = [], []
    tot_ce_loss = []

    progress_bar = tqdm(loader)

    if args.adversarial:
        attack = PGD(
            classifier, delta, F.cross_entropy,
            clip_min=float('-inf'), clip_max=float('inf')
        )

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

        if split == 'train':
            classifier.train()

        latent_data = autoencoder.encode(data_batch)

        if args.adversarial:
            latent_data = attack.attack(
                delta / 10, latent_data, 20, targets_batch,
                targeted=False, num_restarts=1, random_start=True
            )

        logits = classifier(latent_data)
        ce_loss = cre_loss(logits, targets_batch)
        predictions_batch = classifier.predict(latent_data)

        predictions.append(predictions_batch.detach().cpu())
        targets.append(targets_batch.detach().cpu())

        if split == 'train':
            optimizer.zero_grad()
            ce_loss.mean().backward()
            optimizer.step()

        tot_ce_loss.append(ce_loss.mean().item())
        acc = balanced_accuracy(predictions, targets)

        progress_bar.set_description(
            f'[{split}] epoch={epoch:d}, ce_loss={np.mean(tot_ce_loss):.4f}, '
            f'acc={acc:0.4f}'
        )

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)
    
    # sns.histplot(predictions)
    # plt.savefig('test.png')

    # accuracy = accuracy_score(targets, predictions)
    # balanced_accuracy = balanced_accuracy_score(targets, predictions)
    
    # tn, fp, fn, tp = confusion_matrix(targets, predictions).ravel()
    # f1 = f1_score(targets, predictions)

    # writer.add_scalar('Accuracy/%s' % split, accuracy, epoch)
    # writer.add_scalar('Balanced Accuracy/%s' % split, balanced_accuracy, epoch)
    # writer.add_scalar('Cross Entropy/%s' % split, tot_ce_loss.mean(), epoch)
    # writer.add_scalar('True Positives/%s' % split, tp, epoch)
    # writer.add_scalar('False Negatives/%s' % split, fn, epoch)
    # writer.add_scalar('True Negatives/%s' % split, tn, epoch)
    # writer.add_scalar('False Positives/%s' % split, fp, epoch)
    # writer.add_scalar('F1 Score/%s' % split, f1, epoch)
    # writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
    # writer.add_scalar('Stat. Parity/%s' % split, tot_stat_par.mean(), epoch)
    # writer.add_scalar('Equalized Odds/%s' % split, tot_eq_odds.mean(), epoch)

    return tot_ce_loss


# print('saving model to', models_dir)
# writer = SummaryWriter(log_dir)

for epoch in range(num_epochs):
    run(autoencoder, classifier, optimizer, train_loader, 'train')

    autoencoder.eval()
    classifier.eval()

    valid_loss = run(autoencoder, classifier, optimizer, val_loader, 'valid')
    scheduler.step(np.mean(valid_loss))

    torch.save(
        classifier.state_dict(), classifier_path
    )

# writer.close()

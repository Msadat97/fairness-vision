import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch.utils.data import DataLoader
from tqdm import tqdm
from mnist import MnistLoader
from lcifr.code.attacks import PGD
from model import VAE, LatentClassifier, AutoEncoder, LatentEncoder, DataModel
from dl2.training.supervised.oracles import DL2_Oracle
from lcifr.code.experiments.args_factory import get_args
from lcifr.code.constraints.general_categorical_constraint import SegmentConstraint
from utils import accuracy, get_latents
from attack import SegmentPDG


def standard_acc(model, data, device):
    model = model.to(device)
    pred_list, target_list = [], []
    for inputs, targets in data:
        inputs = inputs.to(device)
        targets = targets.to(device)
        pred_list.append(model.predict(inputs))
        target_list.append(targets)
    return accuracy(pred_list, target_list)
    
    
def robust_acc(device, epsilon: float, data: DataLoader, model: DataModel):
    prediction_list, target_list = [], []
    latent_pdg = SegmentPDG(
        model.encoder, epsilon, F.l1_loss,
        clip_min=float('-inf'), clip_max=float('inf'), idx=5
    )
    for _, batch in enumerate(tqdm(data)):
        inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device)
        targets = targets.type(torch.long)
        
        latents = model.encode(inputs)
        inputs_ = latent_pdg.attack(epsilon/10, inputs, 20, latents, False, 1, True)
        
        latents_ = model.encode(inputs_)
        deltas, _ = torch.max(torch.abs(latents_ - latents), dim=1)
        deltas = deltas[:, None]
        # attack = PGD(
        #     classifier, deltas, F.cross_entropy,
        #     clip_min=float('-inf'), clip_max=float('inf')
        # )
        # latent_advs = attack.attack(
        #     delta / 10, latents, 20, targets,
        #     targeted=False, num_restarts=1, random_start=True
        # )
        latent_advs = []
        for delta, latent, target in zip(deltas, latents, targets):
            attack = PGD(
                model.classifier, delta, F.cross_entropy,
                clip_min=float('-inf'), clip_max=float('inf')
            )
            latent = latent.unsqueeze(0)
            target = target.unsqueeze(0)
            latent_adv = attack.attack(
                delta / 10, latent, 20, target,
                targeted=False, num_restarts=1, random_start=True
            )
            latent_advs.append(latent_adv)
        latent_advs = torch.cat(latent_advs)
        prediction_list.append(model.classifier.predict(latent_advs))
        target_list.append(targets)
        
    return accuracy(prediction_list, target_list)


encoder_path = "saved_models/vae-lcifr-trained-v6"
classifier_path = "saved_models/test-classifier-v0"

# parameters
lr = 0.005
dl2_lr = 2.5
patience = 5
weight_decay = 0.01
dl2_iters = 25
dl2_weight = 10.0
dec_weight = 0.0
num_epochs = 10
args = get_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(latent_dim=16)
vae.load_state_dict(
    torch.load(
        "saved_models/vae-state-dict-v3",
        map_location=torch.device('cpu')
    )
)
vae.to(device)

latent_encoder = LatentEncoder()
latent_encoder.to(device)

autoencoder = AutoEncoder(vae, latent_encoder)
autoencoder.load_state_dict(
    torch.load(
        encoder_path,
        map_location=lambda storage, loc: storage
    )
)
autoencoder.to(device)

classifier = LatentClassifier(
    latent_encoder.flatten_shape, latent_encoder.num_classes)
classifier.load_state_dict(
    torch.load(
        classifier_path, 
        map_location=torch.device('cpu')
    )
)
classifier.to(device)

data_model = DataModel(autoencoder, classifier)
data_model.freeze()

data = MnistLoader(batch_size=128, shuffle=True, normalize=False, split_ratio=0.8)
train_loader, val_loader, test_loader = data.train_loader, data.val_loader, data.test_loader
train_loader = get_latents(vae, train_loader, device)
val_loader = get_latents(vae, val_loader, device)
test_loader = get_latents(vae, test_loader, device)


delta = 0.005
epsilon = 0.5
constraint = SegmentConstraint(model=autoencoder, delta=delta, epsilon=epsilon, latent_idx=5)
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
    optimizer, 'min', patience=patience, factor=0.5
)

robust = robust_acc(device, epsilon, test_loader, data_model)
acc = standard_acc(data_model,  test_loader, device)
print(f'accuracy = {acc}')
print(f'robust-accuracy = {robust}')

# for batch in train_loader:
#     inputs, _ = batch
#     inputs = inputs.to(device)
#     targets = autoencoder.encode(inputs)
#     inputs_ = latent_pdg.attack(delta/10, inputs, 20, targets, targeted=False, num_restarts=1, random_start=True)
#     targets_ = autoencoder.encode(inputs_)
#     print((targets_ - targets).max(1))
#     break

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
from utils import get_latents
from attack import SegmentPDG
from metrics import standard_accuracy, robust_accuracy, smoothing_accuracy

vae_path = "saved_models/vae-state-dict-v3"
ae_path = "saved_models/vae-lcifr-trained-v7"
classifier_path = "saved_models/robust-classifier-v2"

# parameters
lr = 0.005
dl2_lr = 2.5
patience = 5
weight_decay = 0.01
dl2_iters = 25
dl2_weight = 10.0
dec_weight = 0.0
num_epochs = 10
delta = 0.005
epsilon = 0.5
args = get_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

vae = VAE(latent_dim=16)
vae.load_state_dict(
    torch.load(
        vae_path,
        map_location=torch.device('cpu')
    )
)
vae.to(device)

latent_encoder = LatentEncoder()
latent_encoder.to(device)

autoencoder = AutoEncoder(vae, latent_encoder)
autoencoder.load_state_dict(
    torch.load(
        ae_path,
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
train_loader = get_latents(vae=vae, data_loader=train_loader, shuffle=True, device=device)
val_loader = get_latents(vae=vae, data_loader=val_loader, shuffle=False, device=device)
test_loader = get_latents(vae=vae, data_loader=test_loader, shuffle=False, device=device)

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

# robust = robust_acc(device, epsilon, test_loader, data_model)
# acc = standard_acc(data_model,  test_loader, device)
# print(f'accuracy = {acc}')
# print(f'robust-accuracy = {robust}')


from datasets import ModifiedMNIST
t = time()
data = ModifiedMNIST(train=True, vae=vae)
loader = DataLoader(data, batch_size=100)
count = 0
for batch in tqdm(loader):
    count += 1
print(time() - t)
# print(smoothing_accuracy(data_model, epsilon, test_loader))

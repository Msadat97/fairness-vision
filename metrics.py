from typing import Sequence, Tuple
import torch 
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from attack import PGD, SegmentPDG
from models import DataModel
from tqdm import tqdm
from smoothing import MeanSmoothing
from sklearn.metrics import balanced_accuracy_score


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensor = torch.Tensor


def batch_generator(data: DataLoader, device=DEVICE, to_type=None):
    for inputs, targets in tqdm(data):
        inputs = inputs.to(device)
        targets = targets.to(device)
        if to_type is not None:
            targets = targets.type(to_type) 
        yield inputs, targets


def get_adv_latents(model, inputs, attack, epsilon):
    latents = model.encode(inputs)
    inputs_ = attack.attack(epsilon/10, inputs, 20, latents, False, 1, True)
    latents_ = model.encode(inputs_)
    return latents, latents_


def accuracy(predictions, targets):
    """
    Computes the accuracy score
    """
    y_pred = torch.cat(predictions, dim=0) if isinstance(predictions, Sequence) else predictions
    y_true = torch.cat(targets, dim=0) if isinstance(targets, Sequence) else targets
    
    return torch.sum(y_pred == y_true)/len(y_pred)


# def balanced_accuracy(confusion_matrix):
#     "computes balanced accuracy score from confusion matrix"
#     per_class = confusion_matrix.diag()/(confusion_matrix.sum(0))
#     nan_mask = torch.isnan(per_class)
#     print(torch.isnan(per_class))
#     if torch.any(nan_mask):
#         per_class = per_class[~nan_mask]
#     return torch.mean(per_class)


def balanced_accuracy(predictions, targets):
    """
    Computes the balanced accuracy score
    """
    y_pred = torch.cat(predictions, dim=0) if isinstance(predictions, Sequence) else predictions
    y_true = torch.cat(targets, dim=0) if isinstance(targets, Sequence) else targets
    
    y_pred = y_pred.detach().cpu()
    y_true = y_true.detach().cpu()
    
    return balanced_accuracy_score(y_true=y_true, y_pred=y_pred)


def standard_accuracy(model: nn.Module, data: DataLoader):
    pred_list, target_list = [], []
    for inputs, targets in batch_generator(data):
        pred_list.append(model.predict(inputs))
        target_list.append(targets)
    return balanced_accuracy(pred_list, target_list)


def robust_accuracy(model: DataModel, data: DataLoader, epsilon: float, latent_index: int):
    model = model.to(DEVICE)
    prediction_list, target_list = [], []
    latent_pdg = SegmentPDG(
        model.encoder, epsilon, F.l1_loss,
        clip_min=float('-inf'), clip_max=float('inf'), idx=latent_index
    )
    for inputs, targets in  batch_generator(data, to_type=torch.long):
        latents, latents_ = get_adv_latents(model, inputs, latent_pdg, epsilon)
        deltas, _ = torch.max(torch.abs(latents_ - latents), dim=1)
        deltas = deltas[:, None]
        attack = PGD(
            model.classifier, deltas, F.cross_entropy,
            clip_min=float('-inf'), clip_max=float('inf')
        )
        latent_advs = attack.attack(
            deltas / 10, latents, 20, targets,
            targeted=False, num_restarts=1, random_start=True
        )
        prediction_list.append(model.classifier.predict(latent_advs))
        target_list.append(targets)

    return accuracy(prediction_list, target_list)


def smoothing_accuracy(model, epsilon, data: DataLoader, latent_index: int):
    model = model.to(DEVICE)
    latent_pdg = SegmentPDG(
        model.encoder, epsilon, F.mse_loss,
        clip_min=float('-inf'), clip_max=float('inf'), idx=latent_index
    )
    num_correct = 0
    for inputs, targets in batch_generator(data):
        latents, latents_ = get_adv_latents(model, inputs, latent_pdg, epsilon)
        deltas = torch.norm(latents_ - latents, dim=1)
        for delta, latent, target in zip(deltas, latents, targets):
            smooth = MeanSmoothing(model.classifier, 2, 0.1)
            chat, radi = smooth.certify(latent, 1000, 50000, 0.001, batch_size=2000)
            if chat == target and radi > delta.item():
                num_correct += 1

    return num_correct/len(data.dataset)

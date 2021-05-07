import torch
from sklearn.metrics import accuracy_score
from model import VAE
from torch.utils.data import TensorDataset, DataLoader


def accuracy(pred_list, target_list):
    """
        Computes the accuracy score
        """
    y_pred = torch.cat(pred_list)
    y_true = torch.cat(target_list)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    return accuracy_score(y_pred, y_true)


def predict(model, x):
    """
    Takes the input and the model and then returns the labels
    """
    _, out = model.forward(x)
    _, prediction = torch.max(out, dim=1)
    return prediction


def get_label(logit):
    """
    Returns the labels corresponding to a logit
    """
    _, labels = torch.max(logit, dim=1)
    return labels


def get_latents(vae: VAE, data_loader, device=torch.device('cpu')):
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

import torch
from sklearn.metrics import accuracy_score


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

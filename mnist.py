import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader, random_split


def normalizer(x, device):
    mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
    sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)

    return (x - mean) / sigma


def get_mnist(normalize, device):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path="mnist.npz"
    )

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    x_train = torch.as_tensor(x_train, dtype=torch.float32).to(device)
    x_train.unsqueeze_(1)
    x_test = torch.as_tensor(x_test, dtype=torch.float32).to(device)
    x_test.unsqueeze_(1)

    if normalize:
        x_train = normalizer(x_train, device)
        x_test = normalizer(x_test, device)

    y_train = torch.as_tensor(y_train, dtype=torch.float32).to(device)
    y_test = torch.as_tensor(y_test, dtype=torch.float32).to(device)

    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

    return train_data, test_data


class MnistLoader:
    def __init__(
        self, batch_size=None, shuffle=True, 
        normalize=True, device="cpu", split_ratio=None
    ) -> None:
        self.train_data, self.test_data = get_mnist(normalize, device)
        
        if split_ratio is None:
            self.train_loader = DataLoader(
                self.train_data, batch_size=batch_size, shuffle=True
            )
            self.val_loader = None
        else:
            train, val = self._train_val_split(split_ratio)
            self.train_loader = DataLoader(
                train, batch_size=batch_size, shuffle=True
            )
            self.val_loader = DataLoader(
                val, batch_size=batch_size, shuffle=False
            )
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, shuffle=False
        )

    def _train_val_split(self, ratio):
        n_samples = len(self.train_data)
        n_train = int(ratio * n_samples)
        n_val = n_samples - n_train
        train_data, val_data = random_split(self.train_data, [n_train, n_val])
        return train_data, val_data

    def get_data(self, str):
        if str == "x_train":
            return self.train_data.tensors[0]
        elif str == "x_test":
            return self.test_data.tensors[0]
        elif str == "y_train":
            return self.train_data.tensors[1]
        elif str == "y_test":
            return self.test_data.tensors[1]
        else:
            raise (ValueError)


if __name__ == "__main__":
    data = MnistLoader(batch_size=128, shuffle=True, normalize=False, val_ratio=0.5)

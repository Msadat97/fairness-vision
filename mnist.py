import torch
import tensorflow as tf
from torch.utils.data import TensorDataset, DataLoader

def normalizer(x, device):
    mean = torch.FloatTensor([0.1307]).view((1, 1, 1, 1)).to(device)
    sigma = torch.FloatTensor([0.3081]).view((1, 1, 1, 1)).to(device)
    
    return (x - mean) / sigma

def get_mnist(normalize, device):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(path='mnist.npz')
    
    x_train = x_train/255.0
    x_test = x_test/255.0
    
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
    
    def __init__(self, batch_size = 64, shuffle = True, normalize = True, device='cpu') -> None:
        self.train_data, self.test_data = get_mnist(normalize, device)
        self.train_loader = DataLoader(self.train_data, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=batch_size, shuffle=False)
    
    
    def get_data(self, str):
        if (str == 'x_train'):
            return self.train_data.tensors[0]
        elif (str == 'x_test'):
            return self.test_data.tensors[0]
        elif (str == 'y_train'):
            return self.train_data.tensors[1]
        elif (str == 'y_train'):
            return self.test_data.tensors[1]
        else:
            raise(ValueError)
import torch 

tensor = torch.Tensor


class RunningMean(object):
    def __init__(self, epsilon: float = 1e-5, shape: Tuple[int, ...] = ()):
        """
        Calulates the running mean data stream
        :param epsilon: helps with arithmetic issues
        :param shape: the shape of the data stream's output
        [Taken mainly from stable-baselines3 package]
        """
        self.mean = torch.zeros(shape, device='cpu')
        self.count = epsilon

    @torch.no_grad()
    def update(self, arr: tensor) -> None:
        arr = arr.detach().cpu()
        batch_mean = torch.mean(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_count)

    def update_from_moments(self, batch_mean: tensor, batch_count: int) -> None:
        self.mean = (self.mean * self.count + batch_mean *
                     batch_count)/(self.count + batch_count)
        self.count = batch_count + self.count

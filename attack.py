from numpy.core import overrides
import torch


class FGSM(object):
    def __init__(self, model, epsilon, loss_fn, clip_min, clip_max):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss_fn = loss_fn

    def attack(self, images, targets, targeted):
        self.model.eval()

        images_ = images.clone().detach()
        images_.requires_grad_()

        logits = self.model(images_)
        if logits.ndim == 1:
            logits = logits.unsqueeze(0)
        self.model.zero_grad()
        loss = self.loss_fn(logits, targets, reduction='sum')
        loss.backward()

        if targeted:
            perturbed_images = images_ - self.epsilon * images_.grad.sign()
        else:
            perturbed_images = images_ + self.epsilon * images_.grad.sign()

        if (self.clip_min is not None) or (self.clip_max is not None):
            perturbed_images.clamp_(min=self.clip_min, max=self.clip_max)

        return perturbed_images.detach()


class PGD(object):
    def __init__(self, model, epsilon, loss_fn, clip_min, clip_max):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss_fn = loss_fn
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def attack(self, alpha, inputs, iterations, targets, targeted, num_restarts, random_start):
        self.model.eval()

        inputs_ = inputs.clone().detach()
        inputs_min, inputs_max = inputs_ - self.epsilon, inputs_ + self.epsilon

        fgsm = FGSM(self.model, alpha, self.loss_fn,
                    self.clip_min, self.clip_max)

        if not random_start:
            num_restarts = 1

        for i in range(num_restarts):
            if random_start:
                inputs_ = inputs.clone().detach() + torch.mul(
                    self.epsilon,
                    torch.rand_like(inputs, device=self.device).uniform_(-1, 1)
                )

            for _ in range(iterations):
                inputs_ = fgsm.attack(inputs_, targets, targeted)

                # project onto epsilon-ball around original inputs
                inputs_ = torch.max(inputs_min, inputs_)
                inputs_ = torch.min(inputs_max, inputs_)

        return inputs_.detach()


class SegmentPDG(PGD):
    def __init__(self, model, epsilon, loss_fn, clip_min, clip_max, idx):
        super().__init__(model, epsilon, loss_fn, clip_min, clip_max)
        self.idx = idx 
        self.eps = epsilon
        
    def attack(self, alpha, inputs, iterations, targets, targeted, num_restarts, random_start):
        self.epsilon = torch.zeros_like(inputs[0:1])
        self.epsilon[:, self.idx] = self.eps
        return super().attack(alpha, inputs, iterations, targets, targeted, num_restarts, random_start)

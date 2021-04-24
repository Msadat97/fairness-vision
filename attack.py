import torch

class Attack:

    def __init__(self,
                 model,
                 loss_fn,
                 eps,
                 clip_min=None,
                 clip_max=None,
                 device='cpu',
                 ) -> None:

        self.model = model
        self.loss_fn = loss_fn
        self.eps = eps
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.loss_fn = loss_fn
        self.device = device

    def fgsm_attack(self, input_vec, targets, eta=None):

        self.model.eval()

        if eta is None:
            eta = self.eps

        input_vec_ = input_vec.clone().detach()
        input_vec_.requires_grad_()

        out = self.model(input_vec_)
        self.model.zero_grad()
        loss = self.loss_fn(out, targets)
        loss.backward()

        perturbed_vec = input_vec_ + eta*input_vec_.grad.sign()

        if (self.clip_min is not None) or (self.clip_max is not None):

            perturbed_vec.clamp_(min=self.clip_min, max=self.clip_max)

        return perturbed_vec

    def pgd_attack(self, alpha, input_vec, targets, iterations, num_restarts, random_start=False):

        self.model.eval()

        input_vec_ = input_vec.clone().detach()
        input_vec_min = input_vec_ - self.eps
        input_vec_max = input_vec_ + self.eps

        if not random_start:
            num_restarts = 1

        for i in range(num_restarts):
            if random_start:

                input_vec_ += torch.mul(
                    self.eps,
                    torch.rand_like(
                        input_vec, device=self.device).uniform_(-1, 1)
                )

            for _ in range(iterations):

                input_vec_ = self.fgsm_attack(input_vec_, targets, eta=alpha)
                input_vec_ = torch.max(input_vec_min, input_vec_)
                input_vec_ = torch.min(input_vec_max, input_vec_)

        return input_vec_.detach()

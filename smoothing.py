import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from typing import Tuple

tensor = torch.Tensor


class MeanSmoothing(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, 
                 base_classifier: torch.nn.Module, 
                 num_classes: int, 
                 sigma: float, 
                 device=None
                 ):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param confidence_measure: confidence measure to certify. Values: pred_score, margin
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def certify(self, x: tensor, n0: int, n: int, alpha: float, batch_size: int) -> Tuple[int, float]:
        """ Monte Carlo algorithm for certifying, with probability at least 1 - alpha, that the confidence score is
        above a certain threshold  within some L2 radius.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: lower bounds on expected confidence score at different radii with and without the CDF information
        """

        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)[0]
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)

        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return MeanSmoothing.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius


    def predict(self, x: tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return MeanSmoothing.ABSTAIN
        else:
            return top2[0]
    
    @torch.no_grad()
    def _sample_noise(self, x: tensor, num: int, batch_size,  top_class=-1):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :param top_class: guess for the class with the highest expected confidence score
        :return: an ndarray[float] of length num_classes containing the average confidence scores for each class and an
        ndarray[float] of length num containing the top class scores if top_class is specified.
        """
        counts = np.zeros(self.num_classes, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1))
            noise = torch.randn_like(batch, device=self.device) * self.sigma
            predictions = self.base_classifier(batch + noise).argmax(1)
            labels, batch_count = np.unique(predictions.cpu(), return_counts=True)
            counts[labels] += batch_count
        return counts
    
    
    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    

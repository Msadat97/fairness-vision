from typing import Tuple
import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil, sqrt, log
from statsmodels.stats.proportion import proportion_confint


class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, 
                 base_classifier: torch.nn.Module, 
                 num_classes: int, 
                 sigma: float, 
                 input_radius: float,
                 confidence_measure: str):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        :param confidence_measure: confidence measure to certify. Values: pred_score, margin
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.input_radius = input_radius
        
        self.confidence_measure = confidence_measure

        if self.confidence_measure == 'margin':
            self.exp_cutoff = 0.0
            self.range_min = -1.0
        else:
            self.exp_cutoff = 0.5
            self.range_min = 0.0

        self.range_max = 1.0

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> Tuple[int, float]:
        """ Monte Carlo algorithm for certifying, with probability at least 1 - alpha, that the confidence score is
        above a certain threshold  within some L2 radius.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: lower bounds on expected confidence score at different radii with and without the CDF information
        """

        # set number of thresholds for the CDF
        num_thr = 10000

        # compute epsilon, the statistical confidence bound on the CDF of the scores
        eps = sqrt(log(1 / alpha) / (2 * n))

        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        avg_score_selection = self._sample_noise(x, n0, batch_size)[0]
        # use these samples to take a guess at the top class
        cAHat = avg_score_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        avg_score, top_scores = self._sample_noise(x, n, batch_size, cAHat)

        gap = ceil(n / num_thr)
        thresholds = top_scores[::gap]

        # compute lower bound on expected score
        exp_bar = self._exp_lbd(thresholds, eps, 0.0, self.range_min)
        # print(exp_bar)

        if exp_bar < self.exp_cutoff:
            return Smooth.ABSTAIN, 0.0
        
        exp_cdf = self._exp_lbd(thresholds, eps, self.input_radius, self.range_min)

        return cAHat, exp_cdf

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
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
            return Smooth.ABSTAIN
        else:
            return top2[0]
    
    def _sample_noise(self, x: torch.tensor, num: int, batch_size,  top_class=-1):
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :param top_class: guess for the class with the highest expected confidence score
        :return: an ndarray[float] of length num_classes containing the average confidence scores for each class and an
        ndarray[float] of length num containing the top class scores if top_class is specified.
        """
        num_samples = num
        with torch.no_grad():
            avg_score = np.zeros(self.num_classes, dtype=float)     # average score for each class

            if top_class >= 0:
                top_scores = np.zeros(num, dtype=float)             # scores for top class
            else:
                top_scores = np.zeros(0)

            for batch_num in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                predictions = self.base_classifier(batch + noise).softmax(1)
                pred_np = predictions.cpu().numpy()

                if self.confidence_measure == 'margin':
                    pred_argmax = np.argmax(pred_np, axis=1)
                    pred_sort = np.sort(pred_np, axis=1)
                    pred_max = pred_sort[:, -1]
                    margin = pred_sort[:, -1] - pred_sort[:, -2]

                    pred_np = pred_np - pred_max.reshape((-1, 1))
                    for row in range(pred_np.shape[0]):
                        pred_np[row, pred_argmax[row]] = margin[row]

                avg_score += np.sum(pred_np, axis=0)

                if top_class >= 0:
                    start = batch_num * batch_size
                    end = start + this_batch_size
                    top_scores[start:end] = pred_np[:, top_class]

            top_scores = np.sort(top_scores)
            avg_score = avg_score/num_samples
            return avg_score, top_scores

    def _exp_lbd(self, thresholds: np.ndarray, eps: float, disp: float, range_min: float) -> float:
        """
        Function to compute a lower bound on the expected confidence score using the CDF based method.

        :param thresholds: different thresholds on the confidence scores such that the number of samples between any two
        consecutive values is the same.
        :param eps: statistical confidence bound on the CDF of the scores
        :param disp: L2 length of displacement from input point
        :param range_min: minimum value in the range of the confidence scores
        :return: lower bound on the expected score, after a displacement, using the CDF based method
        """
        exp_bar = range_min
        num_thr = thresholds.size
        for i in range(num_thr):
            prob = max(((num_thr - i) / num_thr) - eps, 0)
            phi_inv = norm.ppf(prob, scale=self.sigma)
            prob = norm.cdf(phi_inv, loc=disp, scale=self.sigma)
            if i == 0:
                exp_bar += (thresholds[i] - range_min) * prob
            else:
                exp_bar += (thresholds[i] - thresholds[i - 1]) * prob
        return exp_bar

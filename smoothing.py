from math import ceil, sqrt, log
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


class CenterSmoothing(object):
    """A smoothed version of a function f """

    # to abstain, Smooth returns this int
    ABSTAIN = -1
    ITER = 0

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 sigma: float,
                 delta: float,
                 alpha_1: float,
                 alpha_2: float,
                 device=None
                 ):
        
        self.base_functino = base_classifier
        self.sigma = sigma
        self.delta = delta
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def certify(self, x: tensor, m: int) -> Tuple[int, float]:
        raise NotImplemented

    def predict(self, x: tensor, n: int, batch_size: int) -> tensor:
        self.ITER += 1
        Z = self._sample_noise(x, n, batch_size=batch_size)
        delta1 = sqrt(1/(2*n) * log(2/self.alpha_1))
        z = self._get_meb(Z)
        Z = self._sample_noise(x, n, batch_size=batch_size)
        p_delta1 = self._get_pdelta(Z)
        delta2 = 0.5 - p_delta1
        if self.delta < max(delta1, delta2):
            if self.ITER > 10:
                return self.ABSTAIN
            else:
                self.predict(x, n, batch_size)
        else:
            self.ITER = 0
            return z
                
    def _get_meb(self, Z):
        distance_mat = Z.pow(2).sum(1).reshape(-1, 1) + Z.pow(2).sum(1).reshape(1, -1) - 2 * Z @ (Z.T)
        distance_mat.fill_diagonal_(0.0)
        distance_mat.sqrt_()
        values, _ = torch.median(distance_mat, dim=1)
        index = values.argmin()
        return Z[index.item()]

    
    def _get_pdelta(self, Z):
        pass
        
    @torch.no_grad()
    def _sample_noise(self, x: tensor, num: int, batch_size: int):

        predictions = []
    
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size

            batch = x.repeat((this_batch_size, 1))
            noise = torch.randn_like(batch, device=self.device) * self.sigma
            predictions.append(self.base_functino(batch + noise).cpu())
    
        return torch.cat(predictions, dim=0)
        

class ConfidenceSmoothing(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self,
                 base_classifier: torch.nn.Module,
                 num_classes: int,
                 sigma: float,
                 input_radius: float,
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
        self.input_radius = input_radius

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
            return ConfidenceSmoothing.ABSTAIN, 0.0

        exp_cdf = self._exp_lbd(
            thresholds, eps, self.input_radius, self.range_min)

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
            return ConfidenceSmoothing.ABSTAIN
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
            # average score for each class
            avg_score = np.zeros(self.num_classes, dtype=float)

            if top_class >= 0:
                # scores for top class
                top_scores = np.zeros(num, dtype=float)
            else:
                top_scores = np.zeros(0)

            for batch_num in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda') * self.sigma

                predictions = self.base_classifier(batch + noise).softmax(1)
                pred_np = predictions.cpu().numpy()

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
        index = np.arange(0, num_thr)
        prob = np.maximum((num_thr - index) / num_thr - eps, 0)
        phi_inv = norm.ppf(prob, scale=0.25)
        prob = norm.cdf(phi_inv, loc=disp, scale=0.25)
        exp_bar = np.diff(np.insert(thresholds, 0, range_min)) * prob
        exp_bar = range_min + np.sum(exp_bar)
        return exp_bar
    



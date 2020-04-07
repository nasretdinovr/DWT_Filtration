import math

import torch
from torch import nn
import torch.nn.functional as F


class Threshold_DWT(nn.Module):
    """Class for speech denoising with using learnable discrete wavelet transform
            Args:
            threshold (flat): threshold value

            requires_grad (bool): whether train wavelet filters or not

            thresholding_algorithm (string): thresholding algorithm used on wavelet decomposition of noise speech.
            Can be hard or soft. <https://ieeexplore.ieee.org/document/7455802>

            threshold_mode (string):
            "level-dependent": usage different thresholding values for each wavelet decomposition level
            "global" - usage single thresholding values for all wavelet decomposition level

            signal_length (int): length of input signal in points

            num_wavelet_levels (int): number of decomposition levels of wavelet transform

            sigma (float): noise variance for using in baseline model

            thresholding_parameter (float): thresholding function parameter. If 0, then ether hard or soft
            thresholding functions became vanilla.
            For more information - https://pdfs.semanticscholar.org/f81f/a9ab84ddad5e8f8730b4ed7a0879924666b2.pdf
        """

    def __init__(self, threshold=0.05,
                 requires_grad=True,
                 thresholding_algorithm='hard',
                 mode='global',
                 signal_length=None,
                 num_wavelet_levels=None,
                 sigma=None,
                 thresholding_parameter=0.05):
        super(Threshold_DWT, self).__init__()

        assert thresholding_algorithm in ['hard', 'soft'], \
            "Incorrect thresholding_algorithm: " + thresholding_algorithm
        self.thresholding_algorithm = thresholding_algorithm
        assert mode in ['level_dependent', 'global'], "Inorect mod: " + mode
        self.mode = mode
        self.thresholding_parameter = nn.Parameter(torch.tensor(thresholding_parameter), requires_grad=False)
        self.requires_grad = requires_grad
        self.signal_length = signal_length
        self.num_wavelet_levels = num_wavelet_levels
        self.sigma = sigma

        if self.mode == 'global':
            self.threshold = nn.Parameter(torch.tensor(threshold, dtype=torch.float32),
                                          requires_grad=requires_grad)
            self.compute_threshold()
        else:
            assert signal_length is not None, "level_dependent mode requires: signal_length"
            assert num_wavelet_levels is not None, "level_dependent mode requires: num_wavelet_levels"

            if self.signal_length % 2 ** self.num_wavelet_levels != 0:
                self.signal_length += 2 ** self.num_wavelet_levels - (self.signal_length % 2 ** self.num_wavelet_levels)
            threshold = self.compute_level_threshold(self.num_wavelet_levels)
            self.threshold = nn.Parameter(torch.ones((2 ** self.num_wavelet_levels, 1))*threshold,
                                          requires_grad=self.requires_grad)


    def update_level_dependent_threshold(self):
        self.threshold = self.thresholds.reshape(-1, 1)
        self.threshold = self.threshold.repeat((1, self.signal_length))

    def compute_threshold(self):
        self.threshold.data.fill_(self.sigma * math.sqrt(2 * math.log(self.signal_length)))

    def compute_level_threshold(self, level):
        return self.sigma * math.sqrt(2 * math.log(self.signal_length))/math.log2(self.num_wavelet_levels)
    
    def soft(self, x):
        labmda = self.thresholding_parameter
        first = torch.sqrt(torch.pow(x-self.threshold, 2)+labmda)
        second = torch.sqrt(torch.pow(x+self.threshold, 2)+labmda)
        return x+(first-second)/2
    
    def hard(self, x):
        mu = self.thresholding_parameter
        first = 1+torch.exp((-x+self.threshold)/mu)
        second = 1+torch.exp((-x-self.threshold)/mu)
        return x*(1/first-1/second+1)
    
    def forward(self, x):
        if self.mode == 'level_dependent':
            x = x.reshape(x.size(0), x.size(1), 2 ** self.num_wavelet_levels, -1)

            if self.thresholding_algorithm == 'hard':
                x = self.hard(x)
                return x.reshape(x.size(0), x.size(1), -1)
            else:
                x = self.soft(x)
                return x.reshape(x.size(0), x.size(1), -1)
        else:
            if self.thresholding_algorithm == 'hard':
                return self.hard(x)
            else:
                mask = torch.abs(x)<self.threshold
                return self.soft(x)

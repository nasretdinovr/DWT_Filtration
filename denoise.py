import math
import torch
from torch import nn
from wavelets.Wavelet_DWT import Wavelet
from threholds.Threshold import ThresholdDWT as Threshold

class Denoiser(nn.Module):
    """Class for speech denoising with using learnable discrete wavelet transform
            Args:
            wavelet_num_levels (int): number of decomposition levels of wavelet transform

            wavelet_kernel_size (Tensor): size of first low-pass and high-pass filters

            sigma (float): noise variance for using in baseline model

            thresholding_algorithm (string): thresholding algorithm used on wavelet decomposition of noise speech.
            Can be hard or soft. <https://ieeexplore.ieee.org/document/7455802>

            threshold_mode (string):
            "level-dependent": usage different thresholding values for each wavelet decomposition level
            "global" - usage single thresholding values for all wavelet decomposition level

        """

    def __init__(self, wavelet_num_levels, wavelet_kernel_size, sigma, thresholding_algorithm='hard',
                 threshold_mode='global', signal_length=None):
        super(Denoiser, self).__init__()

        self.wavelet_num_layers = wavelet_num_levels
        self.wavelet_kernel_size = wavelet_kernel_size
        self.wavelet = Wavelet(wavelet_num_levels, wavelet_kernel_size)
        self.sigma = sigma
        self.thresholding_algorithm = thresholding_algorithm
        self.threshold_mode = threshold_mode

        self.threshold = Threshold(0.005, requires_grad=True, thresholding_algorithm=thresholding_algorithm,
                                   mode=threshold_mode, signal_length=signal_length,
                                   num_layers=wavelet_num_levels, sigma=sigma)

    def forward(self, x):
        h = self.wavelet.decomposition(x)
        h = self.threshold(h)
        output = self.wavelet.reconstruction(h)
        return output


def rmse_loss(x, y):
    criterion = nn.MSELoss()
    loss = torch.sqrt(criterion(x, y))
    return loss


def signal_noise_rate(x, y):
    criterion = nn.MSELoss(reduction='sum')
    loss = 10 * torch.log10(torch.pow(x, 2).sum() / criterion(x, y))
    return loss
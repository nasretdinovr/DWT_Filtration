import math
import torch
from torch import nn
from wavelets.Wavelet_DWT import Wavelet_DWT as Wavelet
from threholds.Threshold import Threshold_DWT as Threshold

class Denoiser(nn.Module):
    """Class for speech denoising with using learnable discrete wavelet transform
            Args:
            num_wavelet_levels (int): number of decomposition levels of wavelet transform

            wavelet_kernel_size (Tensor): size of first low-pass and high-pass filters

            sigma (float): noise variance for using in baseline model

            thresholding_algorithm (string): thresholding algorithm used on wavelet decomposition of noise speech.
            Can be hard or soft. <https://ieeexplore.ieee.org/document/7455802>

            threshold_mode (string):
            "level-dependent": usage different thresholding values for each wavelet decomposition level
            "global" - usage single thresholding values for all wavelet decomposition level

            wavelet_name (string): initialize filters with names from http://wavelets.pybytes.com/

            trainable_wavelets (bool): whether train wavelet filters or not
        """

    def __init__(self, num_wavelet_levels,
                 wavelet_kernel_size,
                 sigma,
                 thresholding_algorithm='hard',
                 threshold_mode='global',
                 thresholding_parameter=0.05,
                 signal_length=None,
                 wavelet_name=None,
                 trainable_wavelets=False,
                 trainable_threshold=True):
        super(Denoiser, self).__init__()

        self.wavelet_num_layers = num_wavelet_levels
        self.wavelet_kernel_size = wavelet_kernel_size
        self.wavelet = Wavelet(num_wavelet_levels,
                               wavelet_kernel_size,
                               trainable_wavelets,
                               name=wavelet_name)
        self.sigma = sigma
        self.thresholding_algorithm = thresholding_algorithm
        self.threshold_mode = threshold_mode
        self.thresholding_parameter = thresholding_parameter
        self.threshold = Threshold(requires_grad=trainable_threshold,
                                   thresholding_algorithm=thresholding_algorithm,
                                   mode=threshold_mode,
                                   signal_length=signal_length,
                                   num_wavelet_levels=num_wavelet_levels,
                                   sigma=sigma,
                                   thresholding_parameter=thresholding_parameter)

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


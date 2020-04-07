import torch
from torch import nn
import torch.nn.functional as F

from wavelets.Wavelet_base import Wavelet_base


class Wavelet_DWT_Full(Wavelet_base):
    """Class for computing learnable discrete wavelet transform for full filterbank
    """
    def __init__(self, num_wavelet_levels, wavelet_size, trainable_wavelets=False, name=None):
        super(Wavelet_DWT_Full, self).__init__(num_wavelet_levels, wavelet_size, trainable_wavelets, name)

    def decomposition_step(self, signal, wavelet_level):
        signal_hi = self.compute_next_level_decomposition(signal, self.hi)
        signal_lo = self.compute_next_level_decomposition(signal, self.lo)
        if wavelet_level == 1:
            return torch.cat((signal_hi, signal_lo), dim=2)

        decomposition_hi = self.decomposition_step(signal_hi, wavelet_level-1)
        decomposition_lo = self.decomposition_step(signal_lo, wavelet_level-1)
        return torch.cat((decomposition_hi, decomposition_lo), dim=2)

    def reconstruction_step(self, signal, length):
        if 2*length >= signal.size(2):
            first = signal[:, :, :length]
            second = signal[:, :, length:]
            hi = self.compute_next_level_reconstruction(first, self.hi)
            lo = self.compute_next_level_reconstruction(second, self.lo)
            return hi + lo

        idxs = torch.arange(0, signal.size(2), length).reshape(-1, 2)
        for first_id, second_id in idxs:
            first = signal[:, :, first_id:second_id]
            second = signal[:, :, second_id:second_id+length]
            hi = self.compute_next_level_reconstruction(first, self.hi)
            lo = self.compute_next_level_reconstruction(second, self.lo)
            signal[:, :, first_id:second_id+length] = hi + lo
        return self.reconstruction_step(signal, length*2)

    def decomposition(self, signal):
        if signal.size(2)%2**self.num_wavelet_levels != 0:
            signal = F.pad(signal, (0, 2**self.num_wavelet_levels-(signal.size(2)%2**self.num_wavelet_levels)))
        decomposition = self.decomposition_step(signal, self.num_wavelet_levels)
        return decomposition


    def reconstruction(self, decomposition):
        length = decomposition.size(2)//(2**self.num_wavelet_levels)
        return self.reconstruction_step(decomposition, length)
                               
    def forward(self, signal):
        decomposition = self.decomposition(signal) 
        return decomposition

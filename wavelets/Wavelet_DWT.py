import torch
import torch.nn.functional as F
from wavelets.Wavelet_base import Wavelet_base


class Wavelet_DWT(Wavelet_base):
    """Class for computing learnable vanilla discrete wavelet transform
    """
    def __init__(self, num_wavelet_levels, wavelet_size, trainable_wavelets=False, name=None):
        super(Wavelet_DWT, self).__init__(num_wavelet_levels, wavelet_size, trainable_wavelets, name)

    def decomposition(self, signal):
        if signal.size(2)%2**self.num_wavelet_levels != 0:
            signal = F.pad(signal, (0,2**self.num_wavelet_levels-(signal.size(2)%2**self.num_wavelet_levels)))
        start = 0
        length = signal.size(2)
        decomposition = torch.Tensor(signal.size(0), 1, signal.size(2)).to(signal.device)
        for i in range(self.num_wavelet_levels):
            length = length//2
            decomposition[:, :, start:start+length] = self.compute_next_level_decomposition(signal,self.hi)
            start += length
            signal = self.compute_next_level_decomposition(signal,self.lo)
        decomposition[:, :, start:] = signal
        return decomposition  

    def reconstruction(self, decomposition):
        length = decomposition.size(2)//(2**self.num_wavelet_levels)
        end = decomposition.size(2)-length
        reconstruction = decomposition[:, :, -length:]
        for i in range(self.num_wavelet_levels):
            data = decomposition[:, :, end-length:end]
            hi = self.compute_next_level_reconstruction(data, self.hi)
            lo = self.compute_next_level_reconstruction(reconstruction, self.lo)
            end -= length
            length = length*2
            reconstruction = hi + lo
        return reconstruction  
                               
    def forward(self, signal):
        decomposition = self.decomposition(signal) 
        return decomposition
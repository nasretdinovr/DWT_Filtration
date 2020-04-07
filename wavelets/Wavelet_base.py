import warnings

import pywt
import torch
from torch import nn
import torch.nn.functional as F


class Wavelet_base(nn.Module):
    """Base class for computing learnable wavelet transform
            Args:
            num_wavelet_levels (int): number of decomposition levels of wavelet transform

            wavelet_size (int): size of first low-pass and high-pass filters

            trainable_wavelets (bool): whether train wavelet filters or not

            name (string): initialize filters with names from http://wavelets.pybytes.com/

    """
    def __init__(self, num_wavelet_levels,
                 wavelet_size,
                 trainable_wavelets=False,
                 name=None):
        super(Wavelet_base, self).__init__()

        self.name = name
        self.trainable_wavelets = trainable_wavelets

        self.num_wavelet_levels = num_wavelet_levels
        self.wavelet_size = wavelet_size

        self.hi = nn.Parameter(torch.Tensor(1, 1, wavelet_size))
        self.lo = nn.Parameter(torch.Tensor(1, 1, wavelet_size))

        self.downsample_filter = nn.Parameter(torch.zeros(1, 1, 2), requires_grad=False)
        self.downsample_filter[0, 0, 0] = 1
        self.upsample_filter = nn.Parameter(torch.zeros(1, 1, 2), requires_grad=False)
        self.upsample_filter[0, 0, 1] = 1

        self.reset_parameters()
    
    def energy(self, filter):
        """Computes energy of filter
        Args:
            filter (Tensor: a filter whose energy to be calculated
        """
        return (filter.pow(2).sum())
    
    def reset_parameters(self):
        """Initializes filters with xavier uniform distribution and imposes conditions on them in order to be
        wavelet-like
        """
        if self.name is not None:
            wavelet = pywt.Wavelet(self.name)
            if self.wavelet_size != wavelet.dec_len:
                warnings.warn('Wavelet size has been changed from {} to {}'.format(self.wavelet_size, wavelet.dec_len))
                self.wavelet_size = wavelet.dec_len

            self.hi = nn.Parameter(torch.Tensor(1, 1, self.wavelet_size), requires_grad=self.trainable_wavelets)
            self.hi.data[0, 0, :] = torch.tensor(wavelet.dec_hi)

            self.lo = nn.Parameter(torch.Tensor(1, 1, self.wavelet_size), requires_grad=self.trainable_wavelets)
            self.lo.data[0, 0, :] = torch.tensor(wavelet.dec_lo)
        else:
            nn.init.xavier_uniform_(self.hi)
            hi = self.hi-self.hi.mean()
            self.hi = nn.Parameter(hi/torch.sqrt(self.energy(hi)))
            lo = self.reverse(self.hi)
            odd = torch.arange(1, self.lo.size(2)-1, 2).long()
            lo[:, :, odd] = lo[:, :, odd]*-1
            self.lo = nn.Parameter(lo)

    def downsample(self, signal):                              
        return F.conv1d(signal, self.downsample_filter, stride=2)
    
    def upsample(self, signal):
        """Inserts zeros between elements of given signal
        Args:
            signal (Tensor): signal to be upsampled
        """
        return F.conv_transpose1d(signal, self.upsample_filter, stride=2)
    
    def reverse(self, signal):
        """ Reverses given signal
        Args:
            signal (Tensor): signal to be reversed
        """
        idx = torch.arange(signal.size(2)-1, -1, -1).long()
        return signal[:,:,idx]
    
    def keep(sekf, signal, length, start):
        """Extracts vector of given length from a signal beginning with start point
        Args:
            signal (Tensor): signal to be cut
            length (int): length of output vector
            start (int): starting point output vector is extracted from
        """
        return(signal[:,:,start:start+length])

    def periodized_extension(self, x, pad):
        if x.size(2) % 2 != 0:
            x = torch.cat((x, x[:, :, -1:]), dim=2)
        return F.pad(x, (pad,pad), mode='circular')

    def compute_next_level_decomposition(self, data, filter):
        """Computes (k)th level wavelet decomposition
        Args:
            data (Tensor): (k-1) wavelet decomposition or initial signal
            filter (Tensor): (k-1)th layer low/high-pass filter
        Returns:
            Tensor: kth level wavelet decomposition
        """

        padded_data = self.periodized_extension(data, self.wavelet_size // 2)
        up_filer = self.keep(F.conv1d(F.pad(padded_data, (self.wavelet_size - 1, self.wavelet_size - 1)),
                                      self.reverse(filter)), data.size(2), self.wavelet_size)
        data = self.downsample(up_filer)
        return data

    def compute_next_level_reconstruction(self, data, filt):
        """Computes (k)th level wavelet reconstruction
        Args:
            data (Tensor): (k-1) wavelet reconstruction or initial signal
            filter (Tensor): (k-1)th layer low/high-pass filter
        Returns:
            Tensor: kth level wavelet reconstruction
        """

        data = self.upsample(data)
        padded_data = self.periodized_extension(data, self.wavelet_size//2)
        data = self.keep(F.conv1d(F.pad(padded_data, (self.wavelet_size-1, self.wavelet_size-1)),filt),
                         data.size(2), self.wavelet_size)
        return data

if __name__ == "__main__":
    wavelet = Wavelet_base(3, 8, 'db8')
    print (wavelet.hi)
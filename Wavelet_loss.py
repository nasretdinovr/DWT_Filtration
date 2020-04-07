import torch
import torch.nn as nn


class WaveletLoss(nn.Module):
    def __init__(self):
        super(WaveletLoss, self).__init__()

    @staticmethod
    def orthonormal(w_1, w_2=None):
        kernel_size = w_1.size(2)
        if w_2 is None:
            w_2 = w_1
            start = 1
            wavelet_restriction = (w_1.pow(2).sum()-1).pow(2)
        else:
            start = 0
            wavelet_restriction = torch.zeros(1).type_as(w_1)

        for m in range(start, w_1.size(2) // 2):
            tmp = torch.zeros((1, 1, 1)).type_as(w_1)
            prods = [w_1[:, :, i] * w_2[:, :, i + 2 * m] for i in range(kernel_size - 2 * m)]
            for n in prods:
                tmp += n
            wavelet_restriction += tmp[0, 0, 0].pow(2)
        return wavelet_restriction

    def forward(self, wavelet):
        w_hi = wavelet.hi
        w_lo = wavelet.lo

        wavelet_restriction1 = self.orthonormal(w_hi)
        wavelet_restriction2 = self.orthonormal(w_lo)
        wavelet_restriction3 = w_hi.sum().pow(2)
        wavelet_restriction4 = (w_lo.sum() - 2 ** (1 / 2)).pow(2)
        wavelet_restriction5 = self.orthonormal(w_lo, w_hi)

        return (wavelet_restriction1 + wavelet_restriction2 + wavelet_restriction3 +
                wavelet_restriction4 + wavelet_restriction5)


class DenoiserLoss(nn.Module):
    def __init__(self, wavelet_reg, net_reg=None):
        super(DenoiserLoss, self).__init__()

        self.wavelet_reg = wavelet_reg
        self.net_reg = net_reg

        self.criterion = nn.MSELoss()
        self.wavelet_criterion = WaveletLoss()

    def forward(self, targets, outputs, net):

        speech_loss = self.criterion(outputs, targets)

        wavelet_loss = self.wavelet_criterion(net.wavelet)

        regularization = self.wavelet_reg * wavelet_loss
        if self.net_reg is not None:
            l2_reg = torch.tensor(0.).type_as(outputs)
            for name, param in net.named_parameters():
                if param.requires_grad:
                    if 'wavelet' in name:
                        continue
                    else:
                        l2_reg += param.norm(2)
            regularization += self.net_reg * l2_reg

        return speech_loss + regularization, wavelet_loss, speech_loss
import time
import torch
from torch import nn
from dataloader import LoadDataset_VCTK as LoadDataset
from denoise import Denoiser
from trainer import Trainer

db4_L = [[[-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114,
           -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523]]]
db4_H = [[[-0.23037781330885523, 0.7148465705525415, -0.6308807679295904, -0.02798376941698385,
           0.18703481171888114, 0.030841381835986965, -0.032883011666982945, -0.010597401784997278]]]
L = [[[0.7071067811865476, 0.7071067811865476]]]
H = [[[-0.7071067811865476, 0.7071067811865476]]]

if __name__ == "__main__":

    ds_path = '/home/administrator/Data/VCTK-Corpus'

    shared_params = {'ds_path': ds_path,
                     'audio_rate': 48000,
                     'random_crop': True,
                     'normalize': True}
    batch_size = 100
    sigma = 0.03
    trainloader = LoadDataset(mode='train',
                              inputLength=0.8,
                              sigma=sigma,
                              **shared_params)
    dataloader_train = torch.utils.data.DataLoader(trainloader, batch_size,
                                                   shuffle=True,
                                                   pin_memory=False)
    testloader = LoadDataset(mode='test',
                             inputLength=0.8,
                             sigma=sigma,
                             **shared_params)
    dataloader_test = torch.utils.data.DataLoader(testloader, batch_size,
                                                  shuffle=True,
                                                  pin_memory=False)
    now = time.time()
    noised_signal, signal = next(iter(dataloader_train))
    print(time.time() - now)
    noised_signal.shape, noised_signal.shape, len(dataloader_train), len(dataloader_test)
    net = Denoiser(6, 8, sigma, thresholding_algorithm='hard', threshold_mode='level_dependent',
                   signal_length=noised_signal.size(-1))
    # net.wavelet.lo = nn.Parameter(torch.Tensor(db4_L))
    # net.wavelet.hi = nn.Parameter(torch.Tensor(db4_H))
    best_score = float("inf")

    name = 'Denoising_VCTK_DWT_Vanilla'

    trainer = Trainer(net, batch_size, name, C=0.001, lambda_reg=0, lr=0.0000001)

    trainer.train(dataloader_train, dataloader_test, 3)


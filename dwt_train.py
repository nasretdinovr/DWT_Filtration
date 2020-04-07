import time
import torch
from argparse import ArgumentParser

from dataloader import LoadDataset_VCTK as LoadDataset
from denoise import Denoiser
from trainer import Trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--ds_path', type=str, default='/home/administrator/Data/VCTK-Corpus')
    parser.add_argument('--log_path', type=str, default='Denoising_VCTK_DWT_Vanilla')

    parser.add_argument('--audio_rate', type=int, default=48000)
    parser.add_argument('--random_crop', type=bool, default=True)
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--inputLength', type=float, default=2.4)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--sigma', type=float, default=0.03)

    parser.add_argument('--num_wavelet_levels', type=int, default=6)
    parser.add_argument('--wavelet_size', type=int, default=8)
    parser.add_argument('--wavelet_reg', type=float, default=0.001)
    parser.add_argument('--net_reg', type=float, default=0)
    parser.add_argument('--wavelet_name', type=str, default='db4')
    parser.add_argument('--lr', type=float, default=0.0000001)
    parser.add_argument('--thresholding_parameter', type=float, default=0.1)

    parser.add_argument('--thresholding_algorithm', type=str, default='hard')
    parser.add_argument('--threshold_mode', type=str, default='level_dependent')

    parser.add_argument('--epochs', type=int, default=3)

    hparams = parser.parse_args()


    dataloader_params = {'ds_path': hparams.ds_path,
                     'audio_rate': hparams.audio_rate,
                     'random_crop': hparams.random_crop,
                     'normalize': hparams.normalize,
                     'inputLength': hparams.inputLength,
                     'sigma': hparams.sigma}

    trainloader = LoadDataset(mode='train', **dataloader_params)

    dataloader_train = torch.utils.data.DataLoader(trainloader,
                                                   hparams.batch_size,
                                                   shuffle=True,
                                                   pin_memory=False)
    testloader = LoadDataset(mode='test',  **dataloader_params)

    dataloader_test = torch.utils.data.DataLoader(testloader,
                                                  hparams.batch_size,
                                                  shuffle=True,
                                                  pin_memory=False)
    now = time.time()
    noised_signal, signal = next(iter(dataloader_train))
    print("Batch uploading time : {}".format(time.time() - now))
    signal_length = noised_signal.size(-1)

    net = Denoiser(hparams.num_wavelet_levels,
                   hparams.wavelet_size,
                   hparams.sigma,
                   thresholding_algorithm=hparams.thresholding_algorithm,
                   threshold_mode=hparams.threshold_mode,
                   signal_length=signal_length,
                   thresholding_parameter=hparams.thresholding_parameter,
                   wavelet_name=hparams.wavelet_name)

    trainer = Trainer(net,
                      hparams.batch_size,
                      hparams.log_path,
                      wavelet_reg=hparams.wavelet_reg,
                      net_reg=hparams.net_reg,
                      lr=hparams.lr)

    trainer.train(dataloader_train, dataloader_test, hparams.epochs)


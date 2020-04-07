import os
import io
from collections import OrderedDict
from scipy import fft
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm
from Wavelet_loss import Wavelet_loss
from denoise import rmse_loss, signal_noise_ratie
from graphviz import render
from torchviz import make_dot, make_dot_from_trace

class Trainer:
    def __init__(self, net, batch_size, name, C, lambda_reg, lr, optimizer='sgd',
                 best_score=float("inf")):
        """
        The classifier used for training and launching predictions
        Args:
            net (nn.Module): The neural net module containing the definition of your model
            batch_size (int): batch size
            name (string): path to save logs
            C (float): wavelet regularization
            lambda_reg(float): l2 regularization
            lr (float): learning rate
        """
        self.net = net
        for par in net.parameters():
            print(par)
        self.epochs = None
        self.global_step = 0
        self.epoch_counter = 0
        self.batch_size = batch_size
        self.best_score = best_score
        self.name = os.path.join(name, net.thresholding_algorithm, net.threshold_mode, optimizer,
                                 'C={}_lambda_reg={}_lr={}'.format(C, lambda_reg, lr))
        self.C = C
        self.lambda_reg = lambda_reg
        self.lr = lr
        self.criterion = Wavelet_loss(C=self.C, lambda_reg=self.lambda_reg)
        if optimizer == 'sgd':
            self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        else:
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.7,
                                                              patience=3, verbose=True, threshold=0.001)

        if torch.cuda.is_available():
            self.device_train = torch.device('cuda:0')
            self.device_val = torch.device('cuda:1')
        else:
            self.device_train = torch.device('cpu')
            self.device_val = torch.device('cpu')

        self.net.to(self.device_train)

        self.writer_train = SummaryWriter(os.path.join('runs', 'VCTK', self.name, 'train'))
        self.writer_val = SummaryWriter(os.path.join('runs', 'VCTK', self.name, 'val'))

    def drow_filters_fft(self):
        hi_f = np.abs(fft(self.net.wavelet.hi[0, 0, :].cpu().data.numpy()))
        lo_f = np.abs(fft(self.net.wavelet.lo[0, 0, :].cpu().data.numpy()))
        n = hi_f.shape[-1]
        plt.plot(np.arange(n // 2 + 1) / (n // 2), lo_f[:n // 2 + 1])
        plt.plot(np.arange(n // 2 + 1) / (n // 2), hi_f[:n // 2 + 1])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.clf()
        return buf

    def _train_epoch(self, train_loader, optimizer, criterion):
        losses = []
        it_count = len(train_loader)
        with tqdm(total=it_count,
                  desc="Epochs {}/{}".format(self.epoch_counter + 1, self.epochs),
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{remaining}{postfix}]'
                  ) as pbar:
            for teration, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self.device_train), targets.to(self.device_train)
                rmse_before = rmse_loss(targets, inputs)
                snr_before = signal_noise_ratie(targets, inputs)
                d = make_dot(self.net(inputs), params=dict(self.net.named_parameters()))
                d.render(format='pdf')
                outputs = self.net(inputs)
                optimizer.zero_grad()

                loss, wavelet_loss, mse_loss = criterion(outputs, targets, self.net)

                loss.backward(retain_graph=True)
                optimizer.step()

                rmse_after = rmse_loss(targets, outputs)
                snr_after = signal_noise_ratie(targets, outputs)

                self.writer_train.add_scalar("loss", loss.item(), global_step=self.global_step)
                self.writer_train.add_scalar("Wavelet_loss", wavelet_loss.item(), global_step=self.global_step)
                self.writer_train.add_scalar("MSE_loss", mse_loss.item(), global_step=self.global_step)
                self.writer_train.add_scalars("RMSE", {'before': rmse_before.item(),
                                                       'after': rmse_after.item()}, global_step=self.global_step)
                self.writer_train.add_scalars("SNR", {'before': snr_before.item(),
                                                      'after': snr_after.item()}, global_step=self.global_step)

                plot_buf = self.drow_filters_fft()
                image = np.array(Image.open(plot_buf))
                image = np.transpose(image, [2, 0, 1])
                self.writer_train.add_image("wavelet", image, global_step=self.global_step)

                self.global_step += 1
                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(loss.item()),
                                             wavelet_loss='{0:1.5f}'.format(wavelet_loss.item()),
                                             mse_loss='{0:1.5f}'.format(mse_loss.item()),
                                             snr='{0:1.5f}'.format(snr_after.item())))
                pbar.update(1)

        return loss.item(), wavelet_loss.item(), mse_loss.item()

    def _validate_epoch(self, val_loader, criterion):
        it_count = len(val_loader)
        mean_mse_loss, mean_snr, mean_rmse = 0, 0, 0
        with tqdm(total=it_count, desc="Validating", leave=False) as pbar:
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device_val), targets.to(self.device_val)

                rmse_before = rmse_loss(targets, inputs)
                snr_before = signal_noise_ratie(targets, inputs)

                outputs = self.net(inputs)

                _, _, mse_loss = criterion(outputs, targets, self.net)

                rmse_after = rmse_loss(targets, outputs)
                snr_after = signal_noise_ratie(targets, outputs)

                mean_mse_loss += mse_loss.item()
                mean_rmse += rmse_after.item()
                mean_snr += snr_after.item()

                pbar.set_postfix(OrderedDict(loss='{0:1.5f}'.format(mse_loss.item()),
                                             snr='{0:1.5f}'.format(snr_after.item())))
                pbar.update(1)

        self.writer_val.add_audio('Noised_audio', inputs[0], sample_rate=48000,
                                  global_step=self.epoch_counter)
        self.writer_val.add_audio('Denoised_audio', outputs[0], sample_rate=48000,
                                  global_step=self.epoch_counter)

        self.writer_val.add_scalar("MSE_loss", mean_mse_loss / it_count, global_step=self.global_step)
        self.writer_val.add_scalars("RMSE", {'after': mean_rmse / it_count},
                                    global_step=self.global_step)
        self.writer_val.add_scalars("SNR", {'after': mean_snr / it_count},
                                    global_step=self.global_step)

        return mean_mse_loss / it_count, mean_rmse / it_count, mean_snr / it_count

    def _run_epoch(self, train_loader, val_loader,
                   optimizer, criterion, lr_scheduler):

        # switch to train mode
        self.net.train()
        self.net.to(self.device_train)

        # Run a train pass on the current epoch
        train_loss, wavelet_loss, mse_loss = self._train_epoch(train_loader, optimizer, criterion)

        # switch to evaluate mode
        self.net.eval()
        self.net.to(self.device_val)

        # Run the validation pass
        mean_val_loss, _, _ = self._validate_epoch(val_loader, criterion)

        # Reduce learning rate when needed
        lr_scheduler.step(mean_val_loss, self.epoch_counter)

        if self.best_score > mean_val_loss:
            self.best_score = mean_val_loss
            directory = os.path.join('saves', self.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.net, os.path.join(directory, 'best'))
        else:
            directory = os.path.join('saves', self.name)
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save(self.net, os.path.join(directory, 'latest'))

        self.epoch_counter += 1

    def evaluate(self, val_loader):
        mean_val_loss, mean_rmse, mean_snr = self._validate_epoch(iter(val_loader), self.criterion)
        return mean_val_loss, mean_rmse, mean_snr

    def train(self, train_loader, val_loader, epochs):

        if self.epochs != None:
            self.epochs += epochs
        else:
            self.epochs = epochs
            self.epoch_counter = 0

        #         self.writer_train.add_graph(self.net)

        for epoch in range(epochs):
            self._run_epoch(iter(train_loader), iter(val_loader), self.optimizer,
                            self.criterion, self.scheduler)

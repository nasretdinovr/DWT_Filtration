import os
import glob
import scipy.io.wavfile as sci_wav
import pandas as pd
import numpy as np
import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import bc_utils as U


class LoadDataset_VoxCeleb(Dataset):
    def __init__(self, 
                 ds_path,
                 mode,
                 sigma,
                 audio_rate=16000,
                 inputLength=1,
                 random_crop=True,
                 normalize=True):
        
        self.ds_path = ds_path
        self.sigma = sigma
        self.audio_rate = audio_rate
        self.inputLength = inputLength
        self.random_crop = random_crop
        self.normalize = normalize
 
        assert mode in ["dev","test"], 'Incorrect mode: ' + mode
        self.mode = mode

        f_path = os.path.join(self.ds_path, self.mode +'.csv')
        self.ds_path = os.path.join(self.ds_path, self.mode)    
        self.files = pd.read_csv(f_path, index_col = '№')['path'].values
        
        self._preprocess_setup()
        
    def __getitem__(self, index):
        noised_signal, signal = self.pull_item(index)
        return noised_signal, signal

    def __len__(self):
        return len(self.files)
    
    def add_noise(self, signal): 
        noise = torch.tensor(np.random.normal(0, 1, size=(signal.size())), dtype=torch.float32)
        return (self.sigma*noise + signal)
    
    def fname_to_wav(self, fname):
        """Retrive wav data from fname
        """
        fname = os.path.splitext(fname)[0] + '.wav'  
        fpath = os.path.join(self.ds_path, fname)
        wav_data = torch.tensor(sci_wav.read(fpath)[1], dtype=torch.float32)
        return wav_data
    
    def _preprocess_setup(self):
        """Apply desired pre_processing to the input
        """
        self.preprocess_funcs = []
        if self.random_crop:
            self.preprocess_funcs.append(
                U.random_crop(int(self.inputLength * self.audio_rate)))

        if self.normalize:
            self.preprocess_funcs.append(U.normalize(2147483648.0))
        
    def preprocess(self, audio):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        audio: array 
            audio signal to be preprocess
        """
        for f in self.preprocess_funcs:
            audio = f(audio)

        return audio
    
    def pull_item(self, index):

        signal = self.fname_to_wav(self.files[index])
        signal = self.preprocess(signal).unsqueeze(0)
        noised_signal = self.add_noise(signal)
        
        return noised_signal, signal
    
class LoadDataset_VCTK(Dataset):
    def __init__(self, 
                 ds_path,
                 mode,
                 sigma,
                 audio_rate=48000,
                 inputLength=1,
                 random_crop=True,
                 normalize=True,
                 test_size=0.1):
        
        self.ds_path = ds_path
        self.sigma = sigma
        self.audio_rate = audio_rate
        self.inputLength = inputLength
        self.random_crop = random_crop
        self.normalize = normalize
        self.test_size = test_size
 
        assert mode in ["train", "test"], 'Incorrect mode: ' + mode
        self.mode = mode
        
        self.files = glob.glob(self.ds_path+'/wav48/*/*.wav')
        
        self._train_test_split()
        self._preprocess_setup()
    
    def _train_test_split(self):
        """Divides dataset at train and test part
        """
        dataset_size = len(self.files)
        if self.mode == 'train':
            self.files = self.files[:-int(dataset_size*self.test_size)]
        else:
            self.files = self.files[-int(dataset_size*self.test_size):]
        
    def __getitem__(self, index):
        noised_signal, signal = self.pull_item(index)
        return noised_signal, signal

    def __len__(self):
        return len(self.files)
    
    def add_noise(self, signal): 
        noise = torch.tensor(np.random.normal(0, 1, size=(signal.size())), dtype=torch.float32)
        return (self.sigma*noise + signal)
    
    def fname_to_wav(self, fname):
        """Retrive wav data from fname
        """
        fname = os.path.splitext(fname)[0] + '.wav'  
        fpath = os.path.join(self.ds_path, fname)
        return torch.tensor(sci_wav.read(fpath)[1], dtype=torch.float32)
    
    def _preprocess_setup(self):
        """Apply desired pre_processing to the input
        """
        self.preprocess_funcs = []
        if self.random_crop:
            self.preprocess_funcs.append(
                U.random_crop(int(self.inputLength * self.audio_rate)))

        if self.normalize:
            self.preprocess_funcs.append(U.normalize(2.0**15))
        
    def preprocess(self, audio):
        """Apply desired pre_processing to the input

        Parameters
        ----------
        audio: array 
            audio signal to be preprocess
        """
        for f in self.preprocess_funcs:
            audio = f(audio)

        return audio
    
    def pull_item(self, index):
        signal = self.fname_to_wav(self.files[index])
        if signal.size(-1) < int(self.inputLength * self.audio_rate):
            to_pad = (int(self.inputLength * self.audio_rate) - signal.size(-1))
            signal = F.pad(signal, (0, to_pad))
        signal = self.preprocess(signal).unsqueeze(0)
        noised_signal = self.add_noise(signal)
        return noised_signal, signal


class LoadDataset_VOiCES(Dataset):
    def __init__(self, ds_path, mode):
        assert mode in ["test", "train"], 'Incorrect mode, must be train or test'
        csv_path = os.path.join(ds_path, mode + ".csv")

        df = pd.read_csv(csv_path)
        self.files = [x for x in df["path"]]
        self.targets = [x for x in df["source"]]

    def __getitem__(self, index):
        im, lb = self.pull_item(index)
        return im, lb

    def __len__(self):
        return len(self.files)

    def pull_item(self, index):
        x, _ = librosa.load(self.files[index])
        y, _ = librosa.load(self.targets[index])
        return torch.FloatTensor(x).unsqueeze(0), torch.FloatTensor(y).unsqueeze(0)

if __name__ == "__main__":
    LoadDataset_VOiCES("")

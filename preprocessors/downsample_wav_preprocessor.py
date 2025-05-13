import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats
from scipy.signal import resample_poly


class DownsampleWavPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(DownsampleWavPreprocessor, self).__init__()
        self.cfg = cfg

    def forward(self, wav):
        # return torch.Tensor(wav[:,::8][:,:1500])#TODO hardcode
        signal_poly_resampled = resample_poly(wav, axis=1, up=250, down=2048)     # Polyphase
        return torch.Tensor(signal_poly_resampled[:, :250]) #TODO hardcode

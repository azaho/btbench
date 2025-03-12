import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats

class DownsampleWavPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(DownsampleWavPreprocessor, self).__init__()
        self.cfg = cfg

    def forward(self, wav):
        return torch.Tensor(wav[:,::8][:,:1500])#TODO hardcode

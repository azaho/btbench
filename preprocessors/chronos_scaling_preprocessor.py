import numpy as np
import torch
import torch.nn as nn
from scipy import signal, stats

class ChronosScalingPreprocessor(nn.Module):
    def __init__(self, cfg):
        super(ChronosScalingPreprocessor, self).__init__()
        self.cfg = cfg

    def forward(self, embed):
        return embed*10 + 0.4

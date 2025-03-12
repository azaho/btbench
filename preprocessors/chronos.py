from .stft import STFTPreprocessor
from .morelet_preprocessor import MoreletPreprocessor
from .superlet_preprocessor import SuperletPreprocessor
import torch
import torch.nn as nn
import models
import os
import numpy as np
import pandas as pd
import torch


#This preprocssor combines a spectrogram preprocessor with a feature extracter (transformer)

class Chronos(nn.Module):
    def __init__(self, cfg):
        super(Chronos, self).__init__()
        from chronos import ChronosPipeline

        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            cfg.path_to_chronos_pretrained,
            #"amazon/chronos-t5-small",
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )

        self.pipeline = pipeline


        self.cfg = cfg

        self.pipeline.model.to('cuda')#TODO hardcode

    def forward(self, wav, spec_preprocessed=None):
        '''
            wav is [n_electrodes, n_time]
            output is [n_electrodes, n_embed]
        '''
        wav = wav[:,::2]#TODO hardcode
        wav = torch.Tensor(wav)
        outputs, _ = self.pipeline.embed(wav)
        #outputs is [n_elec, n_time, d]
        middle = int(outputs.shape[1]/2)
        out = outputs[:,:]#TODO hardcode
        
        #weight = np.arange(0.1, 0.6, 0.1)
        #weight = np.concatenate([weight, weight[::-1]])
        #weight = weight[np.newaxis,:,np.newaxis]
        #weight = torch.FloatTensor(weight)

        if "pool" in self.cfg and self.cfg.pool=="max":
            out, _ = out.max(axis=1)
        elif "pool" in self.cfg and self.cfg.pool=="raw":
            out = out
        else:
            out = out.mean(axis=1)
        out = out.float().cpu() #TODO hardcode
        return out

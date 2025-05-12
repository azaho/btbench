from models import register_model
import torch.nn.functional as F
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("deep_linear_wav_baseline")
class DeepLinearWavModel(BaseModel):
    '''
    Takes as input a raw wave form for one electrode
    '''
    def __init__(self):
        super(DeepLinearWavModel, self).__init__()

    def forward(self, inputs):
        hidden = F.relu(self.linear1(inputs))
        hidden = F.relu(self.linear2(hidden))
        hidden = F.relu(self.linear3(hidden))
        hidden = F.relu(self.linear4(hidden))
        out = (self.linear_out(hidden))
        return out

    def build_model(self, cfg, input_dim):
        self.cfg = cfg
        self.linear1 = nn.Linear(in_features=input_dim, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=256)
        self.linear4 = nn.Linear(in_features=256, out_features=128)
        self.linear_out = nn.Linear(in_features=128, out_features=1) #TODO hardcode out_features
        #TODO hardcode in_features 

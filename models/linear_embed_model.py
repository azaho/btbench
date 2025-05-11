from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("linear_embed")
class LinearEmbed(BaseModel):
    def __init__(self):
        '''
        Takes as input a tensor of BrainBERT embeddings
        '''
        super(LinearEmbed, self).__init__()

    def forward(self, inputs):
        flattened_input = inputs.flatten(start_dim=1)
        #flattened_input = inputs.mean(axis=1) #Take the mean instead of concat
        out = self.linear_out(flattened_input)
        return out

    def build_model(self, cfg):
        self.cfg = cfg
        self.linear_out = nn.Linear(in_features=cfg.input_dim, out_features=1)


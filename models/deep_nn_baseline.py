from models import register_model
import torch.nn as nn
import torch
from models.base_model import BaseModel
from models.transformer_encoder_input import TransformerEncoderInput

@register_model("deep_nn_baseline")
class DeepNNEmbed(BaseModel):
    def __init__(self):
        '''
        Takes as input a tensor of BrainBERT embeddings
        '''
        super(DeepNNEmbed, self).__init__()

    def forward(self, inputs):
        flattened_input = inputs.flatten(start_dim=1)
        #flattened_input = inputs.mean(axis=1) #Take the mean instead of concat
        out = self.linear_out(flattened_input)
        out = self.act_0(out)
        out = self.linear_out_1(out)
        out = self.act_1(out)
        out = self.linear_out_2(out)
        out = self.act_2(out)
        out = self.linear_out_3(out)
        out = self.act_3(out)
        out = self.linear_out_4(out)
        return out

    def build_model(self, cfg):
        self.cfg = cfg
        self.linear_out = nn.Linear(in_features=cfg.input_dim, out_features=512)
        self.act_0 = nn.GELU()
        self.linear_out_1 = nn.Linear(in_features=512, out_features=512)
        self.act_1 = nn.GELU()
        self.linear_out_2 = nn.Linear(in_features=512, out_features=512)
        self.act_2 = nn.GELU()
        self.linear_out_3 = nn.Linear(in_features=512, out_features=512)
        self.act_3 = nn.GELU()
        self.linear_out_4 = nn.Linear(in_features=512, out_features=1)



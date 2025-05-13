import torch
from .base_criterion import BaseCriterion
from torch import nn
from criterions import register_criterion
import torch.nn.functional as F


import numpy as np

@register_criterion("brant_feature_extract_criterion")
class BrantFeatureExtractCriterion(BaseCriterion):
    def __init__(self):
        super(BrantFeatureExtractCriterion, self).__init__()
        pass

    def build_criterion(self, cfg):
        self.cfg = cfg
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()
        if 'loss_fn' in cfg and cfg.loss_fn == "mse":
            self.loss_fn = nn.MSELoss()
        elif 'loss_fn' in cfg and cfg.loss_fn == "cce":
            self.loss_fn = nn.CrossEntropyLoss(reduction="mean")
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, model, batch, device, return_predicts=False):
        #TODO fix the dataset here. 
        inputs = batch["input"].to(device) #potentially don't move to device if dataparallel
        if self.cfg.get("use_power", False):
            power = batch["power"].to(device)
            output = model.forward(inputs, power)
        else:
            output = model.forward(inputs)
        
        if self.cfg.get("loss_fn", "default") == "cce": 
            labels = torch.LongTensor(np.array(batch["labels"])).to(output.device)
        else :
            labels = torch.FloatTensor(np.array(batch["labels"])).to(output.device)
        
        output = output.squeeze(-1)
        self.loss_fn = self.loss_fn.to(output.device) # Move ContrastiveLoss to output device
        loss = self.loss_fn(output, labels)
        if return_predicts:
            if 'loss_fn' in self.cfg and self.cfg.loss_fn == "mse":
                predicts = output.squeeze().detach().cpu().numpy()
            elif 'loss_fn' in self.cfg and (self.cfg.loss_fn == "cce"):
                if output.shape[-1] == 2:
                    if output.shape[0] == 1:
                        predicts = F.softmax(output, dim=len(output.shape) - 1).detach().cpu().numpy()[:, 1]  # if only one item in the batch, don't squeeze
                    else: 
                        predicts = F.softmax(output, dim=len(output.shape) - 1).squeeze().detach().cpu().numpy()[:, 1]
                else:
                    predicts = F.softmax(output, dim=len(output.shape) - 1).squeeze().detach().cpu().numpy()
            else:
                predicts = self.sigmoid(output).squeeze().detach().cpu().numpy()
            logging_output = {"loss": loss.item(),
                              "predicts": predicts}
        else:
            logging_output = {"loss": loss.item()}
            
        return loss, logging_output


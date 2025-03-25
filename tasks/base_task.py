from torch.utils.data import Subset
import models
import criterions
from torch.utils import data
import torch
from datasets import build_dataset
from tasks.utils import split_dataset_idxs

class BaseTask():
    def __init__(self, cfg):
        self.cfg = cfg

    def build_model(self, cfg):
        return models.build_model(cfg)

    def load_datasets(self, data_cfg, preprocessor_cfg):
        pass

    def train_step(self, batch, model, criterion, optimizer, scheduler, device, grad_clip=None):
        #print(batch["input"].shape)
        loss, logging_out = criterion(model, batch, device)
        loss.backward(loss)
        if grad_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        logging_out["grad_norm"] = grad_norm.item()
        return logging_out

    def build_criterion(self, cfg):
        return criterions.build_criterion(cfg)

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def get_valid_outs():
        raise NotImplementedError

    def save_model_weights(self, model, states, multi_gpu):
        #expects a new state with "models" key
        if multi_gpu:
            return model.module.save_model_weights(states)
        return model.save_model_weights(states)

    def load_model_weights(self, model, states, multi_gpu):
        if multi_gpu:
            model.module.load_weights(states)
        else:
            model.load_weights(states)

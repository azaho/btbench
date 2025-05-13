import copy
from tqdm.contrib.logging import logging_redirect_tqdm
import numpy as np
import torch.nn as nn
import os
from tqdm import tqdm
import torch
import tasks
import torch.multiprocessing as mp
import torch.distributed as dist
import logging
from tensorboardX import SummaryWriter
from schedulers import build_scheduler
import torch_optimizer as torch_optim
import json
from pathlib import Path

log = logging.getLogger(__name__)

class Runner():
    def __init__(self, cfg, task, criterion, model_cfg):
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.task = task
        self.evaluator = None
        self.device = cfg.device
        self.criterion = criterion
        self.exp_dir = os.getcwd()
        self.output_tb = cfg.get("output_tb", True)
        self.logger = None
        if self.output_tb:
            exp_dir = cfg.get('results_dir', self.exp_dir)
            self.logger = SummaryWriter(exp_dir)

        if 'start_from_ckpt' in cfg:
            self.load_from_ckpt()

    def load_from_ckpt(self, model):
        ckpt_path = self.cfg.start_from_ckpt
        init_state = torch.load(ckpt_path)
        self.task.load_model_weights(model, init_state['model'], self.cfg.multi_gpu)
        self.optim.load_state_dict(init_state["optim"])
        self.scheduler.load_state_dict(init_state["optim"])

    def _init_optim(self, cfg, model):
        if cfg.optim == "SGD":
            optim = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum = 0.9)
        elif cfg.optim == 'Adam':
            optim = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0.01)
        elif cfg.optim == 'AdamW':
            optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
        elif cfg.optim == 'AdamW_finetune':
            upstream_params = model.upstream.parameters() if not self.cfg.multi_gpu else model.module.upstream.parameters()
            upstream_params = list(upstream_params)
            ignored_params = list(map(id, upstream_params))
            linear_params = filter(lambda p: id(p) not in ignored_params,
                                 model.parameters())
            linear_params = list(linear_params)

            optim = torch.optim.AdamW([
                        {'params': upstream_params},
                        {'params': linear_params, 'lr': cfg.lr}
                    ], lr=cfg.lr*0.1)
        elif cfg.optim == 'Adam_brant_finetune': 
            optim = torch.optim.Adam(
                [{'params': list(model.encoder_t.parameters()),   'lr': cfg.ft_lr},
                {'params': list(model.encoder_ch.parameters()),   'lr': cfg.ft_lr},
                {'params': list(model.final_module.parameters()),        'lr': cfg.lr}],
                betas=(0.9, 0.999),
                eps=1e-8,
            )
        elif cfg.optim == 'LAMB':
            optim = torch_optim.Lamb(model.parameters(), lr=cfg.lr)
        else:
            print("no valid optim name")
        return optim

    def output_logs(self, progress, scheduler, train_logging_outs, val_logging_outs):
        global_step = progress.n
        train_logging_outs['lr'] = scheduler.get_lr()
        standard_metrics = ["lr", "loss", "grad_norm"]
        all_standard_metrics = {}
        def add_prefix(prefix, outs):
            for k,v in outs.items():
                if k in standard_metrics:
                    all_standard_metrics[f'{prefix}_{k}'] = v
        add_prefix('train', train_logging_outs)
        add_prefix('val', val_logging_outs)

        log.info(all_standard_metrics)

        if self.logger is not None:
            for k,v in all_standard_metrics.items():
                self.logger.add_scalar(k, v, global_step=global_step)
        self.task.output_logs(train_logging_outs, val_logging_outs, self.logger, global_step)

    def get_valid_outs(self, model, valid_loader):
        valid_logging_outs = self.task.get_valid_outs(model, valid_loader, self.criterion, self.device) 
        return valid_logging_outs

    def save_checkpoint_last(self, states, best_val=False):
        cwd = os.getcwd()
        if best_val:
            save_path = os.path.join(cwd, 'checkpoint_best.pth')
        else:
            save_path = os.path.join(cwd, 'checkpoint_last.pth')
        log.info(f'Saving checkpoint to {save_path}')
        torch.save(states, save_path)
        log.info(f'Saved checkpoint to {save_path}')

    def save_checkpoints(self, model, optim, scheduler, best_val=False):
        if 'save_checkpoints' in self.cfg and not self.cfg.save_checkpoints:#the default is to save the checkpoints. so this only triggers if the argument is deliberately false.
            return
        all_states = {}
        all_states = self.task.save_model_weights(all_states, self.cfg.multi_gpu)
        all_states['optim'] = optim.state_dict()
        all_states['scheduler'] = scheduler.get_state_dict()
        if self.cfg.multi_gpu:
            all_states['model_cfg'] = model.module.cfg
        else:
            all_states['model_cfg'] = model.cfg
        self.save_checkpoint_last(all_states)
        if best_val:
            self.save_checkpoint_last(all_states, best_val)
        
    def run_epoch(self, model, optim, scheduler, train_loader, valid_loader, progress, total_loss, best_state):
        epoch_loss = []
        best_model, best_val = best_state
        for batch in train_loader:
            if progress.n >= progress.total:
                break
            model.train()
            logging_out = self.task.train_step(batch, model, self.criterion, optim, scheduler, self.device, self.cfg.grad_clip)
            total_loss.append(logging_out["loss"])
            epoch_loss.append(logging_out["loss"])
            log_step = progress.n % self.cfg.log_step == 0 or progress.n == progress.total - 1

            ckpt_step = False
            if self.cfg.checkpoint_step > -1:
                ckpt_step = progress.n % self.cfg.checkpoint_step == 0 or progress.n == progress.total - 1

            valid_logging_outs = {}
            if ckpt_step or log_step:
                model.eval()
                valid_logging_outs = self.get_valid_outs(model, valid_loader)
            if log_step:
                logging_out["loss"] = np.mean(total_loss)
                self.output_logs(progress, scheduler, logging_out, valid_logging_outs)
                total_loss = []
            if ckpt_step:
                better = False

                metric = "loss"
                better = valid_logging_outs[metric] < best_val[metric]

                if better:
                    self.save_checkpoints(model, optim, scheduler, best_val=True)
                    best_val = valid_logging_outs
                    best_model = copy.deepcopy(model)
                else:
                    self.save_checkpoints(model, optim, scheduler)
            progress.update(1)
        return total_loss, (best_model, best_val)

    def scheduler_step(self):
        pass

    def cross_val(self):
        folds = []
        k_fold = len(self.task.train_datasets)
        for i in range(k_fold):
            model = self.task.build_model(self.model_cfg)

            if self.cfg.multi_gpu:
                model = torch.nn.DataParallel(model)
                log.info(f'Use {torch.cuda.device_count()} GPUs')
            assert not(self.cfg.device=='cpu' and self.cfg.multi_gpu)
            model.to(self.device)
            optim = self._init_optim(self.cfg, model)
            scheduler = build_scheduler(self.cfg.scheduler, optim)

            train_loader = self.get_batch_iterator(self.task.train_datasets[i], self.cfg.train_batch_size, shuffle=self.cfg.shuffle, num_workers=self.cfg.num_workers, persistent_workers=self.cfg.num_workers>0)
            val_loader = self.get_batch_iterator(self.task.val_datasets[i], self.cfg.valid_batch_size, shuffle=self.cfg.shuffle)
            best_model = self.train(model, optim, scheduler, train_loader, val_loader)

            test_loader = self.get_batch_iterator(self.task.test_datasets[i], self.cfg.valid_batch_size, shuffle=self.cfg.shuffle)

            test_results = self.test(best_model, test_loader)
            train_results = self.test(best_model, train_loader)

            fold_results = {f"test_{k}":v for k,v in test_results.items()}

            folds.append(fold_results)

        Path(self.cfg.results_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(self.cfg.results_dir, "results.json"), "w") as f:
            json.dump(folds, f)

    def train(self, model, optim, scheduler, train_loader, valid_loader):
        total_loss = []
        best_val = {"loss": float("inf"), "roc_auc": 0}
        best_model = None
        best_state = (best_model, best_val)
        total_steps = self.cfg.total_steps
        progress = tqdm(total=total_steps, dynamic_ncols=True, desc="overall")
        with logging_redirect_tqdm():
            if self.cfg.checkpoint_step > -1:
                self.save_checkpoints(model, optim, scheduler)
            while progress.n < progress.total:
                total_loss, best_state = self.run_epoch(model, optim, scheduler, train_loader, valid_loader, 
                                                        progress, total_loss, best_state)
                best_model, best_val = best_state
            progress.close()

        return best_model
                
    def format_test_outs(self, test_outs):
        new_test_outs = {}
        for k,v in test_outs.items():
            if k not in ["predicts", "labels"]:
                new_test_outs[k] = v
        return new_test_outs

    def test(self, best_model, test_loader):
        test_outs = self.task.get_valid_outs(best_model, test_loader, self.criterion, self.device)
        formatted = self.format_test_outs(test_outs)
        log.info(f"test_results {formatted}")
        return test_outs

    def get_batch_iterator(self, dataset, batch_size, **kwargs):
        return self.task.get_batch_iterator(dataset, batch_size, **kwargs)

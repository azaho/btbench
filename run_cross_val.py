#example
#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=16 +data=masked_spec +model=debug_model +data.data=/storage/czw/self_supervised_seeg/all_electrode_data/manifests
from omegaconf import DictConfig, OmegaConf, open_dict
import hydra
import models
import tasks
from runner import Runner
import logging
import os
import json
import pandas as pd
from btbench_config import BTBENCH_LITE_ELECTRODES

log = logging.getLogger(__name__)

def cross_val(cfg):
    log.info("cross val loop")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')
    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    if "brant" in cfg.model.name:
        cfg['model']['num_electrodes'] = len(task.train_datasets[0].dataset.ordered_electrodes[cfg.data.subject])
    criterion = task.build_criterion(cfg.criterion)
    model_cfg = cfg.model
    runner = Runner(cfg.exp.runner, task, criterion, model_cfg)
    runner.cross_val()

@hydra.main(config_path="conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    with open(cfg.data_prep.electrodes, "r") as f:
        all_electrodes = json.load(f)

    #with open(cfg.data_prep.brain_runs, "r") as f:
    #    subj_brain_runs = json.load(f)
    eval_name = cfg.data.eval_name
    model_name = cfg.model.name    
    brain_run = f"trial{cfg.data.brain_run:03}"
    subject = f"sub_{cfg.data.subject}"
    electrodes = all_electrodes[subject]

    #order the electrodes
    dataset_dir = cfg.data.raw_brain_data_dir
    regions_file = os.path.join(dataset_dir, f'localization/{subject}/depth-wm.csv')
    localization_df = pd.read_csv(regions_file)
    localization_df.Electrode = [x.replace('*','').replace('#','') for x in localization_df.Electrode]

    labels = list(localization_df.Electrode)
    for e in electrodes:
       assert e in labels

    #NOTE that the braintreebank_subject is sensitive to the ordering that the electrodes are passed in
    selected_electrodes = [e for i,e in enumerate(labels) if e in electrodes] 

    data_cfg_copy = cfg.data.copy()
    data_cfg_copy['subject'] = subject
    data_cfg_copy['brain_runs'] = [brain_run]
    data_cfg_copy['electrodes'] = selected_electrodes
    data_cfg_copy['eval_name'] = eval_name
    cfg["data"] = data_cfg_copy

    results_dir_root = cfg.exp.runner.results_dir_root
    results_dir = os.path.join(results_dir_root, data_cfg_copy.split_type, f"{model_name}_{subject}_{brain_run}_{eval_name}")
    with open_dict(cfg):
        cfg.exp.runner.results_dir=results_dir
        cfg['model']['output_dim'] = 1
        
    cross_val(cfg)

if __name__ == "__main__":
    main()

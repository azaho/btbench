#example
#python3 run_train.py +exp=spec2vec ++exp.runner.device=cuda ++exp.runner.multi_gpu=True ++exp.runner.num_workers=16 +data=masked_spec +model=debug_model +data.data=/storage/czw/self_supervised_seeg/all_electrode_data/manifests
from omegaconf import DictConfig, OmegaConf
import hydra
import models
import tasks
from runner import Runner
import logging
import os
import json
import pandas as pd

log = logging.getLogger(__name__)

def cross_val(cfg):
    log.info("cross val loop")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')
    task = tasks.setup_task(cfg.task)
    task.load_datasets(cfg.data, cfg.preprocessor)
    criterion = task.build_criterion(cfg.criterion)
    model_cfg = cfg.model
    runner = Runner(cfg.exp.runner, task, criterion, model_cfg)
    runner.cross_val()

@hydra.main(config_path="conf", version_base="1.1")
def main(cfg: DictConfig) -> None:
    with open(cfg.data_prep.electrodes, "r") as f:
        all_electrodes = json.load(f)

    with open(cfg.data_prep.brain_runs, "r") as f:
        subj_brain_runs = json.load(f)

    eval_tasks = ["frame_brightness", "global_flow", "local_flow", "global_flow_angle", "local_flow_angle", "face_num", "volume", "pitch", "delta_volume", "delta_pitch", "speech", "onset", "gpt2_surprisal", "word_length", "word_gap", "word_index", "word_head_pos", "word_part_speech", "speaker"]
    for eval_name in eval_tasks:
        for subject, brain_runs in subj_brain_runs.items():
            for brain_run in brain_runs:
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
                results_dir = os.path.join(results_dir_root, f"{subject}_{brain_run}_{eval_name}")
                cfg.exp.runner.results_dir=results_dir
                cross_val(cfg)
                break #TODO
            break #TODO
        break

if __name__ == "__main__":
    main()


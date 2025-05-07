from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
import json
import numpy as np
from tqdm import tqdm as tqdm
from scipy.stats import zscore
from .h5_data_reader import H5DataReader
from .trial_data import TrialData
import h5py

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    with open(cfg.data_prep.brain_runs, "r") as f:
        brain_runs = json.load(f)

    with open(cfg.data_prep.electrodes, "r") as f:
        electrodes = json.load(f)

    for subject in brain_runs:
        data_cfg_template = cfg.data.copy()
        log.info(f'Writing features for {subject}')
        log.info(electrodes[subject])
        log.info(brain_runs[subject])
        data_cfg_template["subject"] = subject
        data_cfg_template["electrodes"] = electrodes[subject]

        data_cfg_template_copy = data_cfg_template.copy()

        for brain_run in brain_runs[subject]:
            log.info(f'Writing features for {brain_run}')
            data_cfg_template_copy["brain_runs"] = [brain_run]

            log.info(f'Obtaining brain data and labels {brain_run}')

            input_file_path = os.path.join(cfg.data.raw_brain_data_dir, "all_subject_data", f"{subject}_{brain_run}.h5")
            with h5py.File(input_file_path, "r") as f:
                n_electrodes = len(f['data'].keys())

            trial_data = TrialData(subject, brain_run, data_cfg_template_copy)
            labels = trial_data.get_brain_region_localization()

            if subject != "sub_1":
                assert n_electrodes == len(labels)
            else:
                assert n_electrodes-1 == len(labels)

            print(subject, brain_run)
            out_file_dir = os.path.join(cfg.data_prep.out_dir, "all_subject_data")
            Path(out_file_dir).mkdir(exist_ok=True, parents=True)

            out_file_path = os.path.join(out_file_dir, f"{subject}_{brain_run}.h5")
            with h5py.File(out_file_path, "w") as out_f:

                for (i,electrode) in tqdm(enumerate(labels)):
                    if electrode in electrodes[subject]:
                        data_cfg_template_electrode_copy = data_cfg_template_copy.copy()
                        data_cfg_template_electrode_copy["electrodes"] = [electrode]
                        trial_data = TrialData(subject, brain_run, data_cfg_template_electrode_copy)
                        reader = H5DataReader(trial_data, data_cfg_template_electrode_copy)
                        channel_data = reader.get_filtered_data().squeeze()
                    else:
                        with h5py.File(input_file_path, "r") as f:
                            channel_data = f['data'][f'electrode_{i}'][:]

                    out_f.create_dataset(f'data/electrode_{i}', data=channel_data)        

if __name__ == "__main__":
    main()

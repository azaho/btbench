import pandas as pd
import torch
from braintreebank_subject import BrainTreebankSubject
import btbench_train_test_splits
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
import btbench_config

# Make sure the config ROOT_DIR is set correctly
print("Expected braintreebank data at:", btbench_config.ROOT_DIR)
print("Sampling rate:", btbench_config.SAMPLING_RATE, "Hz")

from omegaconf import DictConfig, OmegaConf
import hydra
import logging
from pathlib import Path
import os
from datasets import build_dataset
from preprocessors import build_preprocessor
from data.subject_data import SubjectData
from data.multi_electrode_subj_data import MultiElectrodeSubjectData
from data.speech_nonspeech_subject_data import WordOnsetSubjectData, SentenceOnsetSubjectData
import json
import numpy as np
from tqdm import tqdm as tqdm
import csv
from scipy.stats import zscore

log = logging.getLogger(__name__)

def write_outputs(dataset, extracter, output_path):
    for item in tqdm(dataset):
        raw_neural_data, label, subject_id, trial_id, est_idx, est_end_idx = item
        raw_neural_data = raw_neural_data.cpu().numpy()
        all_embeddings = extracter(raw_neural_data).numpy()

        fname = f"sub_{subject_id}_trial_{trial_id}_s_{est_idx}_e_{est_end_idx}"
        save_path = os.path.join(output_path, f'{fname}.npy')
        np.save(save_path, all_embeddings)

def write_trial_data(subject_id, brain_run, extracter, data_cfg_template_copy, cfg):
    ### START BTBENCH WORK HERE
    subject_id = int(subject_id[len("sub_"):])

    # use cache=True to load this trial's neural data into RAM, if you have enough memory!
    # It will make the loading process faster.
    subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
    print("Electrode labels:", subject.electrode_labels) # list of electrode labels

    # Optionally, subset the electrodes to a specific set of electrodes.
    selected_electrodes = ['F3aOFa2', 'F3aOFa3', 'F3aOFa4', 'F3aOFa7']

    dataset_dir = cfg.data.raw_brain_data_dir
    regions_file = os.path.join(dataset_dir, f'localization/sub_{subject_id}/depth-wm.csv')
    localization_df = pd.read_csv(regions_file)
    localization_df.Electrode = [x.replace('*','').replace('#','') for x in localization_df.Electrode]

    labels = list(localization_df.Electrode)
    for e in selected_electrodes:
       assert e in labels

    #NOTE that the braintreebank_subject is sensitive to the ordering that the electrodes are passed in
    selected_electrodes = [e for i,e in enumerate(labels) if e in selected_electrodes] 

    subject.set_electrode_subset(selected_electrodes) # if you change this line when using cache=True, you need to clear the cache after: subject.clear_neural_data_cache()
    print("Electrode labels after subsetting:", subject.electrode_labels)

    trial_id = int(brain_run[len("trial"):])

    subject.load_neural_data(trial_id)
    window_from = None
    window_to = None # if None, the whole trial will be loaded

    print("All neural data shape:")
    print(subject.get_all_electrode_data(trial_id, window_from=window_from, window_to=window_to).shape) # (n_electrodes, n_samples). To get the data for a specific electrode, use subject.get_electrode_data(trial_id, electrode_label)

    print("\nElectrode coordinates:")
    print(subject.get_electrode_coordinates()) # L, P, I coordinates of the electrodes

    popt_coords = localization_df[localization_df.Electrode.isin(selected_electrodes)][['L','P','I']]
    btbench_coords = subject.get_electrode_coordinates()
    
    print(btbench_coords.cpu().numpy())
    print(popt_coords.to_numpy())
    assert (btbench_coords.cpu().numpy() == popt_coords.to_numpy()).all()

    # Options for eval_name (from the BTBench paper):
    #   frame_brightness, global_flow, local_flow, global_flow_angle, local_flow_angle, face_num, volume, pitch, delta_volume, 
    #   delta_pitch, speech, onset, gpt2_surprisal, word_length, word_gap, word_index, word_head_pos, word_part_speech, speaker
    eval_name = cfg.data_prep.task_name

    # if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
    # if False, the dataset will output the neural data directly
    output_indices = False

    start_neural_data_before_word_onset = 0 # the number of samples to start the neural data before each word onset
    end_neural_data_after_word_onset = btbench_config.SAMPLING_RATE * 1 # the number of samples to end the neural data after each word onset -- here we use 1 second

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=eval_name, output_indices=output_indices, 
                                                        start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset)

    print("Items in the dataset:", len(dataset), "\n")
    print("The first item:", dataset[0][0], f"label = {dataset[0][1]}", sep="\n")

    Path(cfg.data_prep.output_directory).mkdir(exist_ok=True, parents=True)
    write_outputs(dataset, extracter, cfg.data_prep.output_directory) 

    localization_dir = os.path.join(cfg.data_prep.output_directory, "localization")
    sub_loc_path = os.path.join(localization_dir, f"sub_{subject_id}.csv")
    if os.path.exists(sub_loc_path):
        loaded_sub_loc = pd.read_csv(sub_loc_path, index_col=False)
        if not (loaded_sub_loc==localization_df).all().all():
            raise RuntimeError("Localization did not match stored localization. Please clear saved embed directory and re-run")
    else:
        Path(localization_dir).mkdir(exist_ok=True, parents=True)
        localization_df.to_csv(sub_loc_path, index=False)

    ordered_electrodes_path = os.path.join(cfg.data_prep.output_directory, f"all_ordered_electrodes.json")
    loaded_elecs = {}
    if os.path.exists(ordered_electrodes_path):
        with open(ordered_electrodes_path) as f:
            loaded_elecs = json.load(f)
    if not f"sub_{subject_id}" in loaded_elecs: 
        loaded_elecs[f"sub_{subject_id}"] = selected_electrodes 
    elif not (loaded_elecs[f"sub_{subject_id}"] == selected_electrodes):
        raise RuntimeError("Localization did not match stored localization. Please clear saved embed directory and re-run")
    with open(ordered_electrodes_path, "w") as f:
        json.dump(loaded_elecs,f)

@hydra.main(version_base=None, config_path="../conf")
def main(cfg: DictConfig) -> None:
    log.info("Writing data to disk")
    log.info(OmegaConf.to_yaml(cfg, resolve=True))
    log.info(f'Working directory {os.getcwd()}')

    extracter = build_preprocessor(cfg.preprocessor)

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
            write_trial_data(subject, brain_run, extracter, data_cfg_template_copy, cfg)
            log.info(f'Obtained brain data and labels {brain_run}')

if __name__ == "__main__":
    main()

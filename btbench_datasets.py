import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import glob
import random
from btbench_config import *
from braintreebank_subject import Subject

class BrainTreebankSubjectTrialBenchmarkDataset(Dataset):
    def __init__(self, subject, trial_id, dtype, eval_name):
        """
        Args:
            eval_name (str): can be "pitch" or "rms" (rms for volume) or "onset" or "speech"
        """
        single_float_variables = ["pitch", "rms", "mean_pixel_brightness"]
        all_tasks = single_float_variables + ["onset", "speech"]
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"
        self.subject = subject
        self.subject_id = subject.subject_id
        self.trial_id = trial_id
        self.eval_name = eval_name
        self.dtype = dtype

        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_words_df.csv")
        nonverbal_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{self.trial_id}_nonverbal_df.csv")
        self.all_words_df = pd.read_csv(words_df_path)
        self.nonverbal_df = pd.read_csv(nonverbal_df_path)
        self.subject.load_neural_data(self.trial_id)

        if eval_name in ["pitch", "rms"]:
            # Get indices for words in top and bottom quartiles
            all_labels = self.all_words_df[self.eval_name].to_numpy()
            label_percentiles = np.array([np.mean(all_labels < x) for x in all_labels])
            self.extreme_indices = np.where((label_percentiles > 0.75) | (label_percentiles < 0.25))[0]
            self.extreme_labels = torch.from_numpy((label_percentiles[self.extreme_indices] > 0.75).astype(int))
            self.n_samples = len(self.extreme_indices)
            self.__getitem__ = self._pitch_rms__getitem__
        else:
            self.positive_indices = np.where(self.all_words_df["is_onset"].to_numpy() == 1)[0] if eval_name == "onset" else np.arange(len(self.all_words_df))
            self.negative_indices = np.arange(len(self.nonverbal_df))
            min_len = min(len(self.positive_indices), len(self.negative_indices)) # make sure we have an equal number of positive and negative samples
            self.positive_indices = np.sort(np.random.choice(self.positive_indices, size=min_len, replace=False))
            self.negative_indices = np.sort(np.random.choice(self.negative_indices, size=min_len, replace=False))
            self.n_samples = len(self.positive_indices) + len(self.negative_indices)
            self.__getitem__ = self._onset_speech__getitem__

    def _pitch_rms__getitem__(self, idx):
        word_index = self.extreme_indices[idx]
        row = self.all_words_df.iloc[word_index]
        est_idx = int(row['est_idx']) - int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE)
        est_end_idx = int(row['est_idx']) + int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)

        input = self.subject.get_all_electrode_data(self.trial_id, window_from=est_idx, window_to=est_end_idx)
        input = torch.from_numpy(input).to(dtype=self.dtype)
        return input, self.extreme_labels[idx]

    def _onset_speech__getitem__(self, idx):
        if idx % 2 == 0: # even indices are positive samples
            word_index = self.positive_indices[idx//2]
            row = self.all_words_df.iloc[word_index]
            est_idx = int(row['est_idx']) - int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE)
            est_end_idx = int(row['est_idx']) + int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=est_idx, window_to=est_end_idx)
            input = torch.from_numpy(input).to(dtype=self.dtype)
            return input, 1
        else: # odd indices are negative samples
            item_index = self.negative_indices[idx//2]
            row = self.nonverbal_df.iloc[item_index]
            est_idx = int(row['est_idx'])
            est_end_idx = int(row['est_end_idx']) # == est_idx + NEURAL_DATA_NONVERBAL_WINDOW_SIZE * SAMPLING_RATE
            input = self.subject.get_all_electrode_data(self.trial_id, window_from=est_idx, window_to=est_end_idx)
            input = torch.from_numpy(input).to(dtype=self.dtype)
            return input, 0
        
    def __len__(self):
        return self.n_samples
    def __getitem__(self, idx):
        if self.eval_name in ["pitch", "rms"]:
            return self._pitch_rms__getitem__(idx)
        else:
            return self._onset_speech__getitem__(idx)
        
if __name__ == "__main__":
    subject = Subject(3, cache=False)
    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, 0, dtype=torch.float32, eval_name="speech")
    print(len(dataset))
    print(dataset[1][0].shape)

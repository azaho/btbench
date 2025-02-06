import h5py
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
from scipy import signal, stats
import numpy as np
from btbench_config import *

class Subject:
    """ 
        This class is used to load the neural data for a given subject and trial.
        It also contains methods to get the data for a given electrode and trial, and to get the spectrogram for a given electrode and trial.
    """
    def __init__(self, subject_id, sampling_rate=SAMPLING_RATE, allow_corrupted=False, cache=True):
        self.subject_id = subject_id
        self.sampling_rate = sampling_rate
        self.localization_data = self._load_localization_data()
        self.neural_data = {}
        self.h5f_files = {}
        self.cache = cache
        if self.cache: self.neural_data_cache = {}
        self.electrode_labels = self._get_all_electrode_names()
        self.h5_neural_data_keys = {e:"electrode_"+str(i) for i, e in enumerate(self.electrode_labels)}
        self.corrupted_electrodes = self._get_corrupted_electrodes("corrupted_elec.json")
        if not allow_corrupted:
            self.electrode_labels = [e for e in self.electrode_labels if e not in self.corrupted_electrodes]
            for e in self.corrupted_electrodes:
                assert e in self.h5_neural_data_keys, f"Corrupted electrode {e} not found in electrode labels"
                del self.h5_neural_data_keys[e]
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}
        self.laplacian_electrodes, self.electrode_neighbors = self._get_all_laplacian_electrodes()

    def _clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')

    def _get_corrupted_electrodes(self, corrupted_electrodes_file):
        corrupted_electrodes_file = os.path.join(ROOT_DIR, corrupted_electrodes_file)
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[f'sub_{self.subject_id}']]
        # add electrodes that start with "DC" to corrupted electrodes, because they don't have brain signal, instead are used for triggers
        corrupted_electrodes += [e for e in self.electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG")) and e not in corrupted_electrodes] 
        return corrupted_electrodes

    def _get_all_electrode_names(self):
        electrode_labels_file = os.path.join(ROOT_DIR, f'electrode_labels/sub_{self.subject_id}/electrode_labels.json')
        electrode_labels = json.load(open(electrode_labels_file))
        electrode_labels = [self._clean_electrode_label(e) for e in electrode_labels]
        return electrode_labels

    def _get_all_laplacian_electrodes(self, verbose=False):
        """
            Get all laplacian electrodes for a given subject. This function is originally from
            https://github.com/czlwang/BrainBERT repository (Wang et al., 2023)
        """
        def stem_electrode_name(name):
            #names look like 'O1aIb4', 'O1aIb5', 'O1aIb6', 'O1aIb7'
            #names look like 'T1b2
            reverse_name = reversed(name)
            found_stem_end = False
            stem, num = [], []
            for c in reversed(name):
                if c.isalpha():
                    found_stem_end = True
                if found_stem_end:
                    stem.append(c)
                else:
                    num.append(c)
            return ''.join(reversed(stem)), int(''.join(reversed(num)))
        def has_neighbors(stem, stems):
            (x,y) = stem
            return ((x,y+1) in stems) and ((x,y-1) in stems)
        def get_neighbors(stem):
            (x,y) = stem
            return [f'{x}{y}' for (x,y) in [(x,y+1), (x,y-1)]]
        stems = [stem_electrode_name(e) for e in self.electrode_labels]
        laplacian_stems = [x for x in stems if has_neighbors(x, stems)]
        electrodes = [f'{x}{y}' for (x,y) in laplacian_stems]
        neighbors = {e: get_neighbors(stem_electrode_name(e)) for e in electrodes}
        return electrodes, neighbors

    def load_neural_data(self, trial_id):
        if trial_id in self.neural_data: return
        neural_data_file = os.path.join(ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        h5f = h5py.File(neural_data_file, 'r', locking=False)
        self.h5f_files[trial_id] = h5f
        self.neural_data[trial_id] = h5f['data']

        if self.cache:
            neural_data_key = self.h5_neural_data_keys[self.electrode_labels[0]]
            self.neural_data_cache[trial_id] = np.zeros((len(self.electrode_labels), self.neural_data[trial_id][neural_data_key].shape[0]))
            for electrode_label, electrode_id in self.electrode_ids.items():
                neural_data_key = self.h5_neural_data_keys[electrode_label]
                self.neural_data_cache[trial_id][electrode_id] = self.neural_data[trial_id][neural_data_key][:]
            h5f.close() # if cache is True, we don't need the h5f file anymore

    def _load_localization_data(self):
        """Load localization data for this electrode's subject from depth-wm.csv"""
        loc_file = os.path.join(ROOT_DIR, f'localization/sub_{self.subject_id}/depth-wm.csv')
        df = pd.read_csv(loc_file)
        df['Electrode'] = df['Electrode'].apply(self._clean_electrode_label)
        return df
    
    def get_electrode_coordinates(self, laplacian_rereferenced=False):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) array of coordinates (L, I, P) without any preprocessing of the coordinates
                All coordinates are in between 50mm and 200mm for this dataset (check braintreebank_utils.ipynb for statistics)
        """
        # Load the brain regions file for this subject
        regions_df = self.localization_data
        # Create array of coordinates in same order as electrode_labels
        coordinates = np.zeros((self.get_n_electrodes(laplacian_rereferenced), 3))
        for i, label in enumerate(self.electrode_labels if not laplacian_rereferenced else self.laplacian_electrodes):
            assert label in regions_df['Electrode'].values, f"Electrode {label} not found in regions file of subject {self.subject_id}"
            row = regions_df[regions_df['Electrode'] == label].iloc[0]
            coordinates[i] = [row['L'], row['I'], row['P']]
        return coordinates
    def get_n_electrodes(self, laplacian_rereferenced=False):
        return len(self.laplacian_electrodes if laplacian_rereferenced else self.electrode_labels)

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        """
        Get the data for a given electrode for a given trial.
        If cache is True, all of the data is cached in self.neural_data_cache[trial_id][electrode_label] (not just the window)
        """
        neural_data_key = self.h5_neural_data_keys[electrode_label]
        if trial_id not in self.neural_data: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.neural_data[trial_id][neural_data_key].shape[0]
        if self.cache:
            electrode_id = self.electrode_ids[electrode_label]
            return self.neural_data_cache[trial_id][electrode_id][window_from:window_to]
        else: return self.neural_data[trial_id][neural_data_key][window_from:window_to]
    def get_all_electrode_data(self, trial_id, window_from=None, window_to=None):
        if window_from is None: window_from = 0
        if window_to is None: 
            neural_data_key = self.h5_neural_data_keys[self.electrode_labels[0]]
            window_to = self.neural_data[trial_id][neural_data_key].shape[0]
        if self.cache: return self.neural_data_cache[trial_id][:, window_from:window_to]

        all_electrode_data = np.zeros((len(self.electrode_labels), window_to-window_from))
        for electrode_label, electrode_id in self.electrode_ids.items():
            neural_data_key = self.h5_neural_data_keys[electrode_label]
            all_electrode_data[electrode_id] = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)
        return all_electrode_data

    def get_laplacian_rereferenced_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        if electrode_label not in self.laplacian_electrodes:
            raise ValueError(f"Electrode {electrode_label} does not have neighbors")
        neighbors = self.electrode_neighbors[electrode_label]
        neighbor_data = [self.get_electrode_data(n, trial_id, window_from=window_from, window_to=window_to) for n in neighbors]
        return self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)-np.mean(neighbor_data, axis=0)
    def get_spectrogram(self, electrode_label, trial_id, window_from=None, window_to=None, 
                        normalizing_params=None, laplacian_rereferenced=False, return_power=True, 
                        normalize_per_freq=False, nperseg=256, noverlap=0, power_smoothing_factor=1e-5,
                        min_freq=1, max_freq=None): # min_freq=1 to avoid 0 Hz, which has 0 std in some cases.
        if laplacian_rereferenced: 
            data = self.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)
        else: data = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)

        f, t, Sxx = signal.spectrogram(data, fs=self.sampling_rate, nperseg=nperseg, noverlap=noverlap, window='boxcar')
        mask = np.ones(f.shape, dtype=bool)
        if min_freq is not None:
            mask = (f>=min_freq)
        if max_freq is not None:
            mask = (f<=max_freq) & mask
        f, Sxx = f[mask], Sxx[mask] # only keep frequencies up to some frequency
        if return_power: Sxx = 10 * np.log10(Sxx + power_smoothing_factor) # puts a lower bound of -50 on the power with the default power_smoothing_factor
        if normalize_per_freq: 
            if normalizing_params is None: 
                normalizing_params = np.mean(Sxx, axis=1), np.std(Sxx, axis=1)
            Sxx = (Sxx - normalizing_params[0][:, None])/normalizing_params[1][:, None]
        return f, t, Sxx
    def get_electrode_data_partial(self, electrode_label, trial_id, window_from, window_to):
        data = self.get_electrode_data(electrode_label, trial_id)
        return data[window_from:window_to]
    def get_spectrogram_normalizing_params(self, electrode_label, trial_id, laplacian_rereferenced=False):
        f, t, Sxx = self.get_spectrogram(electrode_label, trial_id, laplacian_rereferenced=laplacian_rereferenced)
        return np.mean(Sxx, axis=1), np.std(Sxx, axis=1)
    def get_electrode_data_normalizing_params(self, electrode_label, trial_id, laplacian_rereferenced=False):
        if laplacian_rereferenced:
            data = self.get_laplacian_rereferenced_electrode_data(electrode_label, trial_id)
        else:
            data = self.get_electrode_data(electrode_label, trial_id)
        return np.mean(data, axis=0), np.std(data, axis=0)
    def close_all_files(self):
        for h5f in self.h5f_files.values():
            h5f.close()
        self.neural_data = {}
        self.neural_data_cache = {}
        self.h5f_files = {}

if __name__ == "__main__":
    subject = Subject(3, cache=False)
    subject.load_neural_data(0)
    print(subject.get_all_electrode_data(0, 0, 100).shape)

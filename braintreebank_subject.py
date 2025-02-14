import h5py
import os
import json
import pandas as pd
import numpy as np

from btbench_config import *

class BrainTreebankSubject:
    """ 
        This class is used to load the neural data for a given subject and trial.
        It also contains methods to get the data for a given electrode and trial, and to get the spectrogram for a given electrode and trial.
    """
    def __init__(self, subject_id, allow_corrupted=True, cache=False, dtype=np.float32):
        self.subject_id = subject_id
        self.subject_identifier = f'btbank{subject_id}'
        self.allow_corrupted = allow_corrupted
        self.cache = cache
        self.dtype = dtype  # Store dtype as instance variable

        self.localization_data = self._load_localization_data()
        self.electrode_labels = self._get_all_electrode_names()

        self.h5_neural_data_keys = {e:"electrode_"+str(i) for i, e in enumerate(self.electrode_labels)} # only used for accessing the neural data in h5 files
        self.electrode_labels = self._filter_electrode_labels()
        self.electrode_ids = {e:i for i, e in enumerate(self.electrode_labels)}

        self.electrode_data_length = {}
        self.neural_data_cache = {} # structure: {trial_id: NumPy Array of shape (n_electrodes, n_samples)}
        self.h5_files = {} # structure: {trial_id: h5py.File}

        # For downstream Laplacian rereferencing
        self.laplacian_electrodes, self.electrode_neighbors = self._get_all_laplacian_electrodes()

    def get_n_electrodes(self):
        return len(self.electrode_labels)
    def _load_localization_data(self):
        """Load localization data for this electrode's subject from depth-wm.csv"""
        loc_file = os.path.join(ROOT_DIR, f'localization/sub_{self.subject_id}/depth-wm.csv')
        df = pd.read_csv(loc_file)
        df['Electrode'] = df['Electrode'].apply(self._clean_electrode_label)
        return df
    def _get_all_electrode_names(self):
        electrode_labels_file = os.path.join(ROOT_DIR, f'electrode_labels/sub_{self.subject_id}/electrode_labels.json')
        electrode_labels = json.load(open(electrode_labels_file))
        electrode_labels = [self._clean_electrode_label(e) for e in electrode_labels]
        return electrode_labels
    def _clean_electrode_label(self, electrode_label):
        return electrode_label.replace('*', '').replace('#', '')
    def _get_corrupted_electrodes(self, corrupted_electrodes_file):
        corrupted_electrodes_file = os.path.join(ROOT_DIR, corrupted_electrodes_file)
        corrupted_electrodes = json.load(open(corrupted_electrodes_file))
        corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[f'sub_{self.subject_id}']]
        return corrupted_electrodes
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
    def _filter_electrode_labels(self):
        """
            Filter the electrode labels to remove corrupted electrodes and electrodes that don't have brain signal
        """
        filtered_electrode_labels = self.electrode_labels
        # Step 1. Remove corrupted electrodes
        if not self.allow_corrupted:
            corrupted_electrodes_file = os.path.join(ROOT_DIR, "corrupted_elec.json")
            corrupted_electrodes = json.load(open(corrupted_electrodes_file))
            corrupted_electrodes = [self._clean_electrode_label(e) for e in corrupted_electrodes[f'sub_{self.subject_id}']]
            filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in corrupted_electrodes]
        # Step 2. Remove trigger electrodes
        trigger_electrodes = [e for e in self.electrode_labels if (e.upper().startswith("DC") or e.upper().startswith("TRIG"))]
        filtered_electrode_labels = [e for e in filtered_electrode_labels if e not in trigger_electrodes]
        return filtered_electrode_labels
    
    def cache_neural_data(self, trial_id):
        assert self.cache, "Cache is not enabled; not able to cache neural data."
        if trial_id in self.neural_data_cache: return  # no need to cache again
        
        # Open file with context manager to ensure proper closing
        neural_data_file = os.path.join(ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        with h5py.File(neural_data_file, 'r', locking=False) as f:
            # Get data length first
            self.electrode_data_length[trial_id] = f['data'][self.h5_neural_data_keys[self.electrode_labels[0]]].shape[0]
            # Pre-allocate array with specific dtype to reduce memory
            self.neural_data_cache[trial_id] = np.zeros((len(self.electrode_labels), 
                                                        self.electrode_data_length[trial_id]), 
                                                        dtype=self.dtype)
            # Load data
            for electrode_label, electrode_id in self.electrode_ids.items():
                neural_data_key = self.h5_neural_data_keys[electrode_label]
                self.neural_data_cache[trial_id][electrode_id] = f['data'][neural_data_key][:].astype(self.dtype)

    def _calculate_laplacian_rereferencing_addon(self, trial_id):
        assert trial_id in self.neural_data_cache, "Trial data must be cached before Laplacian rereferencing can be applied."
        neural_data_cache = self.neural_data_cache[trial_id]
        neighbors_data = np.zeros((len(self.electrode_labels), neural_data_cache.shape[1]))
        for electrode_label, electrode_id in self.electrode_ids.items():
            assert electrode_label in self.electrode_neighbors, f"Neighbor electrodes for electrode {electrode_label} not found in electrode_neighbors. A mistake upstream must have happened."
            neighbors = self.electrode_neighbors[electrode_label]
            neighbors_data[electrode_id] = np.mean([neural_data_cache[self.electrode_ids[n]][:] for n in neighbors], axis=0)
        return neighbors_data
    def clear_neural_data_cache(self, trial_id=None):
        if trial_id is None:
            self.neural_data_cache = {}
            self.h5_files = {}
        else:
            if trial_id in self.neural_data_cache: del self.neural_data_cache[trial_id]
            if trial_id in self.h5_files: del self.h5_files[trial_id]
    def open_neural_data_file(self, trial_id):
        assert not self.cache, "Cache is enabled; Use cache_neural_data() instead."
        if trial_id in self.h5_files: return
        neural_data_file = os.path.join(ROOT_DIR, f'sub_{self.subject_id}_trial{trial_id:03}.h5')
        self.h5_files[trial_id] = h5py.File(neural_data_file, 'r', locking=False)
        self.electrode_data_length[trial_id] = self.h5_files[trial_id]['data'][self.h5_neural_data_keys[self.electrode_labels[0]]].shape[0]
    def load_neural_data(self, trial_id):
        if self.cache: self.cache_neural_data(trial_id)
        else: self.open_neural_data_file(trial_id)
    
    def get_electrode_coordinates(self):
        """
            Get the coordinates of the electrodes for this subject
            Returns:
                coordinates: (n_electrodes, 3) array of coordinates (L, I, P) without any preprocessing of the coordinates
                All coordinates are in between 50mm and 200mm for this dataset (check braintreebank_utils.ipynb for statistics)
        """
        # Create array of coordinates in same order as electrode_labels
        coordinates = np.zeros((len(self.electrode_labels), 3))
        for i, label in enumerate(self.electrode_labels):
            row = self.get_electrode_metadata(label)
            coordinates[i] = [row['L'], row['I'], row['P']]
        return coordinates
    def get_electrode_metadata(self, electrode_label):
        """
            Get the metadata for a given electrode.
        """
        return self.localization_data[self.localization_data['Electrode'] == electrode_label].iloc[0]
    def get_all_electrode_metadata(self):
        filtered_df = self.localization_data[self.localization_data['Electrode'].isin(self.electrode_labels)]
        ordered_df = pd.DataFrame([filtered_df[filtered_df['Electrode'] == label].iloc[0] for label in self.electrode_labels])
        return ordered_df.reset_index(drop=True)

    def get_electrode_data(self, electrode_label, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache:
            if trial_id not in self.neural_data_cache: self.cache_neural_data(trial_id)
            electrode_id = self.electrode_ids[electrode_label]
            return self.neural_data_cache[trial_id][electrode_id][window_from:window_to]
        else:
            if trial_id not in self.h5_files: self.open_neural_data_file(trial_id)
            neural_data_key = self.h5_neural_data_keys[electrode_label]
            return self.h5_files[trial_id]['data'][neural_data_key][window_from:window_to].astype(self.dtype)

    def get_all_electrode_data(self, trial_id, window_from=None, window_to=None):
        if trial_id not in self.electrode_data_length: self.load_neural_data(trial_id)
        if window_from is None: window_from = 0
        if window_to is None: window_to = self.electrode_data_length[trial_id]
        if self.cache: 
            return self.neural_data_cache[trial_id][:, window_from:window_to]
        else:
            all_electrode_data = np.zeros((len(self.electrode_labels), window_to-window_from), dtype=self.dtype)
            for electrode_label, electrode_id in self.electrode_ids.items():
                all_electrode_data[electrode_id] = self.get_electrode_data(electrode_label, trial_id, window_from=window_from, window_to=window_to)
            return all_electrode_data


if __name__ == "__main__":
    # This code checks the memory usage of the Subject class for a given trial.
    # Subject 3 trial 0 weights aeound 3.5GB on the disk, which is achieved when using bfloat16 dtype.
    # When using np.float32, the numpy array weighs around 6-7GB.
    import psutil
    from ml_dtypes import bfloat16
    import sys
    process = psutil.Process()

    def print_memory_usage(label):
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"{label}: {memory_mb:.2f} MB")

    print_memory_usage("Initial RAM usage")
    
    subject = BrainTreebankSubject(3, cache=True, dtype=bfloat16)
    print_memory_usage("After creating Subject")
    
    # Print size of major attributes
    print("\nSize of major attributes:")
    for attr_name, attr_value in subject.__dict__.items():
        if isinstance(attr_value, (dict, list, np.ndarray)):
            size_mb = sys.getsizeof(attr_value) / 1024 / 1024
            if isinstance(attr_value, dict):
                # For dictionaries, also check values
                for v in attr_value.values():
                    if isinstance(v, np.ndarray):
                        size_mb += v.nbytes / 1024 / 1024
            elif isinstance(attr_value, np.ndarray):
                size_mb = attr_value.nbytes / 1024 / 1024
            print(f"{attr_name}: {size_mb:.2f} MB")
    
    subject.load_neural_data(0)
    print_memory_usage("\nAfter loading neural data")
    
    data = subject.get_all_electrode_data(0, 0, 100)
    print_memory_usage("After getting electrode data")
    
    # Print shape and memory info of the cached data
    if subject.neural_data_cache:
        for trial_id, array in subject.neural_data_cache.items():
            print(f"\nTrial {trial_id} cache info:")
            print(f"Shape: {array.shape}")
            print(f"Dtype: {array.dtype}")
            print(f"Memory usage: {array.nbytes / 1024 / 1024:.2f} MB")

    import gc
    gc.collect()
    print_memory_usage("After garbage collection")

    print(f"Data shape: {data.shape}")

Subject = BrainTreebankSubject # alias for convenience
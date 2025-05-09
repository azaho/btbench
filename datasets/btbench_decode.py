from btbench_config import  BTBENCH_LITE_ELECTRODES
import random
import os
import torch
from tqdm import tqdm as tqdm
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils import data
from datasets import register_dataset
from pathlib import Path
import logging
import csv
import json
import glob
import pandas as pd
from preprocessors import build_preprocessor

log = logging.getLogger(__name__)

@register_dataset(name="btbench_decode")
class BTBenchDecodingDataset(data.Dataset):
    def __init__(self, cfg, btbench_dataset, preprocessor_cfg=None):
        super().__init__()
        self.btbench_dataset = btbench_dataset
        self.extracter = build_preprocessor(preprocessor_cfg)
        self.cfg = cfg

        btbench_cache_path = self.cfg.btbench_cache_path

        self.manifest, self.labels = self.create_manifest_and_labels(btbench_dataset)

        electrodes_path = os.path.join(btbench_cache_path, "all_ordered_electrodes.json")
        assert os.path.exists(electrodes_path)
        with open(electrodes_path, 'r') as f:
            ordered_electrodes = json.load(f)

        #check that electrode ordering is the same
        uniq_subjects = set([t[1] for t in self.manifest])
        for subject in uniq_subjects:
            assert ordered_electrodes[subject] == cfg.electrodes#the RHS is what was used in subsetting the electrodes

        elec2absolute_id = {subj:{elec:idx for idx,elec in enumerate(elecs)} for subj,elecs in ordered_electrodes.items()}

        localization_root = os.path.join(btbench_cache_path, "localization")
        all_localization_dfs = {}
        for fpath in glob.glob(f'{localization_root}/*'):
            subject = os.path.split(fpath)[1].split(".")[0]
            df = pd.read_csv(fpath)
            selected_electrodes = ordered_electrodes[subject]
            df = df[df.Electrode.isin(selected_electrodes)] #These are all the electrodes, so let's trim them so they are just the ones that we've actually cached data for
            all_localization_dfs[subject] = df

        for subject in ordered_electrodes:
            sub_id = subject[len("sub_"):]
            key = f"btbank{sub_id}"
            if key in BTBENCH_LITE_ELECTRODES:
                lite_electrodes = BTBENCH_LITE_ELECTRODES[key]
                sub_ordered_electrodes = ordered_electrodes[subject]
                lite_sub_ordered = [e for e in sub_ordered_electrodes if e in lite_electrodes]
                ordered_electrodes[subject] = lite_sub_ordered

        if "sub_sample_electrodes" in cfg:
            sub_sample_electrodes_path = cfg.sub_sample_electrodes
            with open(sub_sample_electrodes_path, 'r') as f:
                sub_sample_electrodes = json.load(f)
            ordered_electrodes, all_localization_dfs = self.make_sub_sample(ordered_electrodes, all_localization_dfs, sub_sample_electrodes)
        else:
            ordered_electrodes, all_localization_dfs = self.make_sub_sample(ordered_electrodes, all_localization_dfs, ordered_electrodes)

        self.ordered_electrodes = ordered_electrodes

        self.all_localization_dfs = all_localization_dfs

        label2idx_dict = {}
        uniq_labels = set(self.labels)
        for idx, l in enumerate(uniq_labels):
            label2idx_dict[l] = idx
        self.label2idx_dict = label2idx_dict
        self.idx2label_dict = {k:v for v,k in label2idx_dict.items()}

        self.absolute_id = {subj: [elec2absolute_id[subj][elec] for elec in elecs] for subj,elecs in self.ordered_electrodes.items()} #A map from subject to a list of indices of the sub sampled channels

        self.region2id = {}
        all_dk_regions = set()
        for subj, df in self.all_localization_dfs.items():
            dk_regions = self.all_localization_dfs[subj]["DesikanKilliany"]
            all_dk_regions.update(dk_regions)
        all_dk_regions = sorted(list(all_dk_regions))
        self.region2id = {r:i for i,r in enumerate(all_dk_regions)}
        self.id2region = {i:r for r,i in self.region2id.items()} 

        self.coords_dict = {}
        for subject in self.ordered_electrodes.keys():
            self.coords_dict[subject] = self.all_localization_dfs[subject][["L", "I", "P"]].to_numpy()

        if self.cfg.get("region_coords", False):
            raise ValueError("Not implemented")


    def create_manifest_and_labels(self, btbench_dataset):
        btbench_cache_path = self.cfg.btbench_cache_path
        manifest, labels = [], []
        for item in tqdm(btbench_dataset):
            raw_neural_data, label, subject_id, trial_id, est_idx, est_end_idx = item

            fname = f"sub_{subject_id}_trial_{trial_id}_s_{est_idx}_e_{est_end_idx}.npy"

            fpath = os.path.join(btbench_cache_path, fname)
            if not os.path.exists(fpath):
                print(fpath)
                raise RuntimeError("Need to create brainbert embeds first")

            manifest.append((fpath, f"sub_{subject_id}"))
            labels.append(label)
        return manifest, labels

    def make_sub_sample(self, ordered_electrodes, all_localization_dfs, sub_sample):
        '''
            ordered_electrodes is {<subject>: [<elec>]}
            sub_sample is {<subject>: [<elec>]} but not necessarily all the subjects
        '''
        assert set(sub_sample.keys()).issubset(set(ordered_electrodes.keys()))
        assert set(sub_sample.keys()).issubset(set(all_localization_dfs.keys()))
        new_ordered_electrodes, new_all_localization_dfs = {}, {}
        for subj, elecs in sub_sample.items():
            ordered = ordered_electrodes[subj]
            print(elecs, ordered)
            assert set(elecs).issubset(set(ordered))
            new_ordered_electrodes[subj] = [e for e in ordered if e in elecs] #makes sure order is preserved

            df = all_localization_dfs[subj]
            assert set(elecs).issubset(set(df.Electrode))
            new_all_localization_dfs[subj] = df[df.Electrode.isin(elecs)]

        return new_ordered_electrodes, new_all_localization_dfs

    def get_input_dim(self):
        item = self.__getitem__(0)
        return item["input"].shape[-1]

    def get_output_size(self):
        return 1 #single logit

    def __len__(self):
        return len(self.manifest)

    def label2idx(self, label):
        return self.label2idx_dict[label]
        
    def __getitem__(self, idx: int):
        fpath, subject = self.manifest[idx]
        input_x = np.load(fpath)
        input_x = self.extracter(input_x)

        input_x = torch.FloatTensor(input_x[self.absolute_id[subject],:])#sub sample the channels based on selection

        embed_dim = input_x.shape[-1]
        cls_token = torch.ones(1,embed_dim)

        input_x = torch.concatenate([cls_token,input_x])

        coords = self.coords_dict[subject]
        coords = torch.LongTensor(coords)

        seq_len = input_x.shape[0] - 1
        seq_id = torch.LongTensor([0]*seq_len)
        return {
                "input" : input_x,
                "wav": np.zeros(3),#TODO get rid of this
                "length": 1+input_x.shape[0], 
                "coords": coords,
                "label": self.label2idx(self.labels[idx]),
                "seq_id": seq_id
               }


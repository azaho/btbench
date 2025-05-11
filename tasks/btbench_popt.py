import os
import json
import logging
import numpy as np
import models
from torch.utils import data
import torch
from tasks import register_task
from tasks.base_task import BaseTask
from tasks.batch_utils import pt_feature_extract_coords_collator
from util.tensorboard_utils import plot_tensorboard_line
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tasks.utils import split_dataset_idxs
from torch.utils.data import Subset

log = logging.getLogger(__name__)

@register_task(name="btbench_popt")
class BTBenchPopTTask(BaseTask):
    def __init__(self, cfg):
        super(BTBenchPopTTask, self).__init__(cfg)

    def load_datasets(self, data_cfg, preprocessor_cfg):
        import btbench_config
        from braintreebank_subject import BrainTreebankSubject

        subject_id = int(data_cfg.subject[len("sub_"):])

        # use cache=True to load this trial's neural data into RAM, if you have enough memory!
        # It will make the loading process faster.
        subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
        #print("Electrode labels:", subject.electrode_labels) # list of electrode labels

        # Optionally, subset the electrodes to a specific set of electrodes.
        selected = data_cfg.electrodes

        subject.set_electrode_subset(selected) # if you change this line when using cache=True, you need to clear the cache after: subject.clear_neural_data_cache()
        #print("Electrode labels after subsetting:", subject.electrode_labels)

        assert len(data_cfg.brain_runs)==1
        trial_id = int(data_cfg.brain_runs[0][len("trial"):])

        window_from = None
        window_to = None # if None, the whole trial will be loaded

        #print("All neural data shape:")
        #print(subject.get_all_electrode_data(trial_id, window_from=window_from, window_to=window_to).shape) # (n_electrodes, n_samples). To get the data for a specific electrode, use subject.get_electrode_data(trial_id, electrode_label)

        #print("\nElectrode coordinates:")
        #print(subject.get_electrode_coordinates()) # L, P, I coordinates of the electrodes

        #TODO move all the below somewhere sensible
        from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset

        # Options for eval_name (from the BTBench paper):
        #   frame_brightness, global_flow, local_flow, global_flow_angle, local_flow_angle, face_num, volume, pitch, delta_volume, 
        #   delta_pitch, speech, onset, gpt2_surprisal, word_length, word_gap, word_index, word_head_pos, word_part_speech, speaker
        eval_name = data_cfg.eval_name

        # if True, the dataset will output the indices of the samples in the neural data in a tuple: (index_from, index_to); 
        # if False, the dataset will output the neural data directly
        output_indices = True

        start_neural_data_before_word_onset = 0 # the number of samples to start the neural data before each word onset
        end_neural_data_after_word_onset = btbench_config.SAMPLING_RATE * 1 # the number of samples to end the neural data after each word onset -- here we use 1 second

        import btbench_train_test_splits
        if data_cfg.split_type=="SS_SM":
            bt_train_datasets, bt_test_datasets = btbench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, k_folds=data_cfg.k_fold, dtype=torch.float32,
                            # Put the dataset parameters here
                        output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                        lite=True)
        elif data_cfg.split_type=="SS_DM":
            bt_train_datasets, bt_test_datasets = btbench_train_test_splits.generate_splits_SS_DM(subject, trial_id, eval_name, dtype=torch.float32, 
                            # Put the dataset parameters here
                        output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                        lite=True)
            bt_train_datasets = [bt_train_datasets]
            bt_test_datasets = [bt_test_datasets]
        elif data_cfg.split_type=="DS_DM":
            bt_train_datasets, bt_test_datasets = btbench_train_test_splits.generate_splits_DS_DM(subject, trial_id, eval_name, dtype=torch.float32,
                            # Put the dataset parameters here
                        output_indices=output_indices, start_neural_data_before_word_onset=start_neural_data_before_word_onset, end_neural_data_after_word_onset=end_neural_data_after_word_onset,
                        lite=True)
            bt_train_datasets = [bt_train_datasets]
            bt_test_datasets = [bt_test_datasets]
            import pdb; pdb.set_trace()

        from datasets.btbench_decode import BTBenchDecodingDataset

        train_datasets, val_datasets = [], []
        for bt_train_dataset in bt_train_datasets:
            train_idxs, val_idxs, test_idxs = split_dataset_idxs(bt_train_dataset, data_cfg)
            train_dataset = BTBenchDecodingDataset(data_cfg, bt_train_dataset, preprocessor_cfg=preprocessor_cfg)
            train_datasets.append(Subset(train_dataset, train_idxs))
            val_datasets.append(Subset(train_dataset, val_idxs))

        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = [BTBenchDecodingDataset(data_cfg, bt_test_dataset, preprocessor_cfg=preprocessor_cfg) for bt_test_dataset in bt_test_datasets]

    def build_model(self, cfg):
        return models.build_model(cfg)
        
    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)

    def get_valid_outs(self, model, valid_loader, criterion, device):
        model.eval()
        all_outs = {"loss":0}
        predicts, labels = [], []
        with torch.no_grad():
            for batch in valid_loader:
                batch["input"] = batch["input"].to(device)
                _, valid_outs = criterion(model, batch, device, return_predicts=True)

                predicts.append(valid_outs["predicts"])
                labels.append(batch["labels"])
                all_outs["loss"] += valid_outs["loss"]
        labels = np.array([x for y in labels for x in y])
        predicts = [np.array([p]) if len(p.shape)==0 else p for p in predicts]
        predicts = np.concatenate(predicts)

        roc_auc = roc_auc_score(labels, predicts)
        f1 = f1_score(labels, np.round(predicts))

        all_outs["roc_auc"] = roc_auc
        all_outs["f1"] = f1

        accuracy = accuracy_score(labels, np.round(predicts))
        all_outs["loss"] /= len(valid_loader)
        all_outs["accuracy"] = accuracy
        all_outs["predicts"] = predicts.tolist()
        all_outs["labels"] = labels.tolist()
        return all_outs

    def get_batch_iterator(self, dataset, batch_size, shuffle=True, **kwargs):
        return data.DataLoader(dataset, batch_size=batch_size, collate_fn=pt_feature_extract_coords_collator, **kwargs)

    def output_logs(self, train_logging_outs, val_logging_outs, writer, global_step):
        val_auc_roc = val_logging_outs["roc_auc"]
        val_f1 = val_logging_outs["f1"]
        val_accuracy = val_logging_outs["accuracy"]
        if writer is not None:
            writer.add_scalar("valid_roc_auc", val_auc_roc, global_step)
            writer.add_scalar("valid_f1", val_f1, global_step)
        log.info(f'valid_roc_auc: {val_auc_roc}, valid_f1: {val_f1}, valid_accuracy: {val_accuracy}')

        #image = train_logging_outs["images"]["wav"]
        #label = train_logging_outs["images"]["wav_label"]
        #tb_image = plot_tensorboard_line(image, title=label)
        #if writer is not None:
        #    writer.add_image("raw_wave", tb_image, global_step)


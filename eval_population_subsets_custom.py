from braintreebank_subject import BrainTreebankSubject
import btbench_train_test_splits, btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time, psutil
import gc  # Add at top with other imports


preprocess_options = [
    'none', # no preprocessing, just raw voltage
]
splits_options = [
    'SS_SM', # same subject, same trial
    'SS_DM', # same subject, different trial
]

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--preprocess', type=str, choices=preprocess_options, default='none', help=f'Preprocessing to apply to neural data ({", ".join(preprocess_options)})')
parser.add_argument('--splits_type', type=str, choices=splits_options, default='SS_SM', help=f'Type of splits to use ({", ".join(splits_options)})')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--nperseg', type=int, default=256, help='Length of each segment for FFT calculation')
parser.add_argument('--only_1second', action='store_true', help='Whether to only evaluate on 1 second after word onset')
parser.add_argument('--lite', action='store_true', help='Whether to use the lite eval for BTBench (which is the default)')

parser.add_argument('--n_electrode_subset', type=int, default=-1, help='Number of electrodes to use in subset (-1 for all)')
parser.add_argument('--n_data', type=int, default=-1, help='Number of examples to use (-1 for all)')

parser.add_argument('--model_name', type=str, default='', help='Name of the model to use')
parser.add_argument('--model_dir', type=str, default='', help='Directory path where the model is stored')

args = parser.parse_args()

eval_names = args.eval_name.split(',') if ',' in args.eval_name else [args.eval_name]
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
preprocess = args.preprocess
splits_type = args.splits_type
seed = args.seed
nperseg = args.nperseg
only_1second = bool(args.only_1second)
lite = bool(args.lite)
model_name = args.model_name
model_dir = args.model_dir

n_electrode_subset = args.n_electrode_subset
n_data = args.n_data

electrode_subset_suffix = f"_ne{n_electrode_subset}" if n_electrode_subset != -1 else ""
data_subset_suffix = f"_nd{n_data}" if n_data != -1 else ""
seed_suffix = f"_seed{seed}" if n_electrode_subset != -1 or n_data != -1 else ""
total_suffix = f"{electrode_subset_suffix}{data_subset_suffix}{seed_suffix}"

# Set random seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

bins_start_before_word_onset_seconds = 0.5# if not only_1second else 0 Since we're using preprocessed data, we use a signel total window
bins_end_after_word_onset_seconds = 1.5# if not only_1second else 1
bin_size_seconds = 0.25
bin_step_size_seconds = 0.125

bin_starts = []
bin_ends = []
if not only_1second:
    for bin_start in np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds-bin_size_seconds, bin_step_size_seconds):
        bin_end = bin_start + bin_size_seconds
        if bin_end > bins_end_after_word_onset_seconds: break

        bin_starts.append(bin_start)
        bin_ends.append(bin_end)
    bin_starts += [-bins_start_before_word_onset_seconds]
    bin_ends += [bins_end_after_word_onset_seconds]
bin_starts += [0]
bin_ends += [1]


max_log_priority = -1 if not verbose else 4
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] {' '*4*indent}{message}")


def preprocess_data(data):
    return data # preprocess_data is not used in this script

# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=False, dtype=torch.float32)
all_electrode_labels = subject.electrode_labels if not lite else btbench_config.BTBENCH_LITE_ELECTRODES[subject.subject_identifier]

electrode_subset = np.random.choice(all_electrode_labels, size=n_electrode_subset, replace=False) if n_electrode_subset != -1 else all_electrode_labels
all_electrode_labels = list(electrode_subset)

class DatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        (window_from, window_to), label = self.dataset[idx]

        filename = f"{subject.subject_identifier}_{trial_id}_s{window_from}_e{window_to}.npy"

        middle_dirname = "btbench_saved_features"
        if lite: middle_dirname += "_lite"
        middle_dirname += total_suffix
        data = np.load(f"{model_dir}/{middle_dirname}/{filename}") # shape: (n_channels, n_samples)
        return data, label


for eval_name in eval_names:
    file_save_dir = f"{save_dir}/{model_name}_{preprocess if preprocess != 'none' else 'voltage'}{'_nperseg' + str(nperseg) if nperseg != 256 else ''}"
    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist
    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}{total_suffix}_{eval_name}.json"
    if os.path.exists(file_save_path):
        log(f"Skipping {file_save_path} because it already exists", priority=0)
        continue


    results_population = {
        "time_bins": [],
    }

    # Load all electrodes at once
    # subject.clear_neural_data_cache()
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    if verbose:
        log("Subject loaded", priority=0)

    # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
    if splits_type == "SS_SM":
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32, 
                                                                                        output_indices=True, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
    elif splits_type == "SS_DM":
        train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_DM(subject, trial_id, eval_name, max_other_trials=3, dtype=torch.float32, 
                                                                                        output_indices=True, 
                                                                                        start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                        end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE),
                                                                                        lite=lite, allow_partial_cache=True)
        train_datasets = [train_datasets]
        test_datasets = [test_datasets]


    for bin_start, bin_end in zip(bin_starts, bin_ends):
        data_idx_from = int((bin_start+bins_start_before_word_onset_seconds)*btbench_config.SAMPLING_RATE)
        data_idx_to = int((bin_end+bins_start_before_word_onset_seconds)*btbench_config.SAMPLING_RATE)

        bin_results = {
            "time_bin_start": float(bin_start),
            "time_bin_end": float(bin_end),
            "folds": []
        }

        # Loop over all folds
        for fold_idx in range(len(train_datasets)):
            train_dataset = train_datasets[fold_idx]
            test_dataset = test_datasets[fold_idx]

            # Create a subset of the training data if n_data is specified
            n_samples = len(train_dataset)
            if n_data > 0:
                # Calculate the number of samples to keep
                n_samples_to_keep = min(n_data, n_samples)
                if verbose:
                    log(f"Using {n_samples_to_keep}/{n_samples} training samples", 
                        priority=1, indent=1)
                # Randomly select indices without replacement
                indices = np.random.choice(n_samples, size=n_samples_to_keep, replace=False)
            else:
                indices = np.arange(n_samples)

            log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
            log("Preparing and preprocessing data...", priority=2, indent=1)

            # Convert PyTorch dataset to numpy arrays for scikit-learn
            X_train = np.array([preprocess_data(item[0][:, data_idx_from:data_idx_to].float().numpy()) for item_i, item in enumerate(train_dataset) if item_i in indices])
            y_train = np.array([item[1] for item_i, item in enumerate(train_dataset) if item_i in indices])
            X_test = np.array([preprocess_data(item[0][:, data_idx_from:data_idx_to].float().numpy()) for item in test_dataset])
            y_test = np.array([item[1] for item in test_dataset])
            gc.collect()  # Collect after creating large arrays

            # Flatten the data after preprocessing in-place
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            log(f"Standardizing data...", priority=2, indent=1)

            # Standardize the data in-place
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            gc.collect()  # Collect after standardization

            log(f"Training model...", priority=2, indent=1)

            # Train logistic regression
            clf = LogisticRegression(random_state=seed, max_iter=10000, tol=1e-3)
            clf.fit(X_train, y_train)

            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)

            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)
            gc.collect()  # Collect after predictions

            # Filter test samples to only include classes that were in training
            valid_class_mask = np.isin(y_test, clf.classes_)
            y_test_filtered = y_test[valid_class_mask]
            test_probs_filtered = test_probs[valid_class_mask]

            # Convert y_test to one-hot encoding
            y_test_onehot = np.zeros((len(y_test_filtered), len(clf.classes_)))
            for i, label in enumerate(y_test_filtered):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_test_onehot[i, class_idx] = 1

            y_train_onehot = np.zeros((len(y_train), len(clf.classes_)))
            for i, label in enumerate(y_train):
                class_idx = np.where(clf.classes_ == label)[0][0]
                y_train_onehot[i, class_idx] = 1

            # For multiclass ROC AUC, we need to calculate the score for each class
            n_classes = len(clf.classes_)
            if n_classes > 2:
                train_roc = roc_auc_score(y_train_onehot, train_probs, multi_class='ovr', average='macro')
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered, multi_class='ovr', average='macro')
            else:
                train_roc = roc_auc_score(y_train_onehot, train_probs)
                test_roc = roc_auc_score(y_test_onehot, test_probs_filtered)

            fold_result = {
                "train_accuracy": float(train_accuracy),
                "train_roc_auc": float(train_roc),
                "test_accuracy": float(test_accuracy),
                "test_roc_auc": float(test_roc)
            }
            bin_results["folds"].append(fold_result)
            
            # Clean up variables no longer needed
            del X_train, y_train, X_test, y_test, train_probs, test_probs
            del y_test_filtered, test_probs_filtered, y_test_onehot, y_train_onehot
            del clf, scaler
            gc.collect()  # Collect after cleanup

            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds and not only_1second:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results


    results = {
        "model_name": model_name,
        "author": "Andrii Zahorodnii",
        "description": f"Simple linear regression using all electrodes (features from {model_name}).",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        },

        "nperseg": nperseg,
        "only_1second": only_1second,
        "seed": seed
    }

    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Results saved to {file_save_path}", priority=0)

    # Clean up at end of each eval_name loop
    del train_datasets, test_datasets
    gc.collect()
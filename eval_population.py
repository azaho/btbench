from braintreebank_subject import BrainTreebankSubject
import btbench_train_test_splits, btbench_config

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import torch, numpy as np
import argparse, json, os, time, psutil

parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name(s) (e.g. onset, gpt2_surprisal). If multiple, separate with commas.')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')
parser.add_argument('--verbose', action='store_true', help='Whether to print progress')
parser.add_argument('--save_dir', type=str, default='eval_results', help='Directory to save results')
parser.add_argument('--preprocess', type=str, choices=['fft', 'spectrogram', 'none'], default='', help='Preprocessing to apply to neural data (fft or spectrogram)')
args = parser.parse_args()

eval_names = args.eval_name.split(',') if ',' in args.eval_name else [args.eval_name]
subject_id = args.subject
trial_id = args.trial 
verbose = bool(args.verbose)
save_dir = args.save_dir
preprocess = args.preprocess

bins_start_before_word_onset_seconds = 0.5
bins_end_after_word_onset_seconds = 2.5
bin_size_seconds = 0.125

# Loop over all time bins
bin_starts = np.arange(-bins_start_before_word_onset_seconds, bins_end_after_word_onset_seconds, bin_size_seconds)
bin_ends = bin_starts + bin_size_seconds
# Add a time bin for the whole window and for 1 second after the word onset
bin_starts = [0, -bins_start_before_word_onset_seconds][::-1] + list(bin_starts)
bin_ends = [1, bins_end_after_word_onset_seconds][::-1] + list(bin_ends)


max_log_priority = -1 if not verbose else 1
def log(message, priority=0, indent=0):
    if priority > max_log_priority: return

    current_time = time.strftime("%H:%M:%S")
    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0
    process = psutil.Process()
    ram_usage = process.memory_info().rss / 1024**3
    print(f"[{current_time} gpu {gpu_memory_reserved:04.1f}G ram {ram_usage:05.1f}G] ({priority}) {' '*4*indent}{message}")

def calculate_fft(electrode_data, spectrogram=True):
    """Calculate FFT features for electrode data.
    
    Args:
        electrode_data (np.ndarray): Array of shape (n_samples, n_electrodes, n_timepoints)
    
    Returns:
        np.ndarray: FFT features of shape (n_samples, n_electrodes, n_frequency_bins)
    """
    n_samples, n_electrodes, n_timepoints = electrode_data.shape
    
    # Reshape to 2D for FFT calculation
    x = electrode_data.reshape(-1, n_timepoints)
    
    # Calculate FFT
    x = np.fft.rfft(x, axis=-1)
    
    # Get number of frequency bins
    n_freq = x.shape[1]
            
    # Reshape back to 3D
    x = x.reshape(n_samples, n_electrodes, n_freq)
    
    if spectrogram:
        # Calculate magnitude and convert to log power
        x = np.abs(x)
        x = np.log(x + 1e-5)
    else:
        # Stack real and imaginary parts
        x_real = np.real(x)
        x_imag = np.imag(x)
        x = np.concatenate([x_real, x_imag], axis=-1)
    
    return x



# use cache=True to load this trial's neural data into RAM, if you have enough memory!
# It will make the loading process faster.
subject = BrainTreebankSubject(subject_id, allow_corrupted=False, cache=True, dtype=torch.float32)
all_electrode_labels = subject.electrode_labels

for eval_name in eval_names:
    results_population = {
        "time_bins": [],
    }

    # Load all electrodes at once
    subject.clear_neural_data_cache()
    subject.set_electrode_subset(all_electrode_labels)  # Use all electrodes
    subject.load_neural_data(trial_id)
    if verbose:
        log("All electrodes loaded", priority=0)

    # train_datasets and test_datasets are arrays of length k_folds, each element is a BrainTreebankSubjectTrialBenchmarkDataset for the train/test split
    train_datasets, test_datasets = btbench_train_test_splits.generate_splits_SS_SM(subject, trial_id, eval_name, add_other_trials=False, k_folds=5, dtype=torch.float32, 
                                                                                    output_indices=False, 
                                                                                    start_neural_data_before_word_onset=int(bins_start_before_word_onset_seconds*btbench_config.SAMPLING_RATE), 
                                                                                    end_neural_data_after_word_onset=int(bins_end_after_word_onset_seconds*btbench_config.SAMPLING_RATE))

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

            log(f"Fold {fold_idx+1}, Bin {bin_start}-{bin_end}")
            log("Preparing data...", priority=0, indent=1)

            # Convert PyTorch dataset to numpy arrays for scikit-learn
            X_train = np.array([item[0][:, data_idx_from:data_idx_to] for item in train_dataset])
            y_train = np.array([item[1] for item in train_dataset])
            X_test = np.array([item[0][:, data_idx_from:data_idx_to] for item in test_dataset])
            y_test = np.array([item[1] for item in test_dataset])

            if preprocess in ['fft', 'spectrogram']:
                log(f"Calculating {preprocess}...", priority=0, indent=1)
                X_train = calculate_fft(X_train, spectrogram=preprocess == 'spectrogram')
                X_test = calculate_fft(X_test, spectrogram=preprocess == 'spectrogram')

            # Flatten the data after preprocessing in-place
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)

            log(f"Standardizing data...", priority=0, indent=1)

            # Standardize the data in-place
            scaler = StandardScaler(copy=False)
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            log(f"Training model...", priority=0, indent=1)

            # Train logistic regression
            clf = LogisticRegression(random_state=42, max_iter=10000, tol=1e-3)
            clf.fit(X_train, y_train)

            # Evaluate model
            train_accuracy = clf.score(X_train, y_train)
            test_accuracy = clf.score(X_test, y_test)

            # Get predictions - for multiclass classification
            train_probs = clf.predict_proba(X_train)
            test_probs = clf.predict_proba(X_test)

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
            if verbose: 
                log(f"Population, Fold {fold_idx+1}, Bin {bin_start}-{bin_end}: Train accuracy: {train_accuracy:.3f}, Test accuracy: {test_accuracy:.3f}, Train ROC AUC: {train_roc:.3f}, Test ROC AUC: {test_roc:.3f}", priority=0, indent=0)

        if bin_start == -bins_start_before_word_onset_seconds and bin_end == bins_end_after_word_onset_seconds:
            results_population["whole_window"] = bin_results # whole window results
        elif bin_start == 0 and bin_end == 1:
            results_population["one_second_after_onset"] = bin_results # one second after onset results
        else:
            results_population["time_bins"].append(bin_results) # time bin results

    results = {
        "model_name": "Logistic Regression",
        "author": "Andrii Zahorodnii",
        "description": f"Simple linear regression using all electrodes ({preprocess if preprocess != 'none' else 'voltage'}).",
        "organization": "MIT",
        "organization_url": "https://azaho.org/",
        "timestamp": time.time(),

        "evaluation_results": {
            f"{subject.subject_identifier}_{trial_id}": {
                "population": results_population
            }
        }
    }

    file_save_dir = f"{save_dir}/linear_{preprocess if preprocess != 'none' else 'voltage'}"
    os.makedirs(file_save_dir, exist_ok=True) # Create save directory if it doesn't exist
    file_save_path = f"{file_save_dir}/population_{subject.subject_identifier}_{trial_id}_{eval_name}.json"

    with open(file_save_path, "w") as f:
        json.dump(results, f, indent=4)
    if verbose:
        log(f"Results saved to {file_save_path}", priority=0)
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
from btbench_config import *
from braintreebank_subject import Subject
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

single_float_variables_name_remapping = {
    "pitch": "pitch",
    "volume": "rms",
    "frame_brightness": "mean_pixel_brightness",
    "global_flow": "max_global_magnitude",
    "local_flow": "max_vector_magnitude",
    "delta_volume": "delta_rms",
    "delta_pitch": "delta_pitch",
    "gpt2_surprisal": "gpt2_surprisal",
    "word_length": "word_length",
}
four_way_cardinal_directions_name_remapping = {
    "global_flow_angle": "max_global_angle",
    "local_flow_angle": "max_vector_angle",
}
classification_variables_name_remapping = {
    "word_head_pos": "bin_head",
    "word_part_speech": "pos"
}
single_float_variables = list(single_float_variables_name_remapping.values()) + list(single_float_variables_name_remapping.keys())
four_way_cardinal_direction_variables = list(four_way_cardinal_directions_name_remapping.values()) + list(four_way_cardinal_directions_name_remapping.keys())
classification_variables = list(classification_variables_name_remapping.values()) + list(classification_variables_name_remapping.keys())
all_tasks = single_float_variables + four_way_cardinal_direction_variables + ["onset", "speech"] + ["face_num", "word_gap", "word_index", "speaker"] + classification_variables
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

# all possible evaluations for now
all_evaluations = ["pitch", "rms", "onset", "speech"]

# the evaluation pairs of (subject_id, trial_id) based on the Population Transformer paper
all_eval_subject_trials = [(1, 2), (2, 6), (3, 0), (6, 4), (7, 0), (4, 1), (10, 0)] # made to match PopT paper
class SingleElectrodeDataset(Dataset):
    def __init__(self, subject, trial_ids, dtype, eval_name, electrode_label):
        """
        Args:
            eval_name (str): can be "pitch", "rms", "onset", "speech", etc.
            electrode_label (str): The electrode to extract data from.
        """
        assert eval_name in all_tasks, f"eval_name must be one of {all_tasks}, not {eval_name}"

        self.trial_ids = trial_ids
        self.subject = subject
        self.subject_id = subject.subject_id
        self.eval_name = eval_name
        self.dtype = dtype
        self.electrode_label = electrode_label  # Store electrode label

        # Ensure electrode exists
        assert self.electrode_label in self.subject.electrode_labels, f"Electrode {self.electrode_label} not found."

        print("Preloading neural data for trials...")
        for trial_id in self.trial_ids:
            self.subject.load_neural_data(trial_id)
        print("Neural data preloading complete!")

    def get_single_word_data(self, trial_id, row):
        """ Extracts data from the specified electrode for a given word index in a given trial """

      
        word_onset = int(row['est_idx'])
        
        est_idx = word_onset - int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE)
        est_end_idx = word_onset + int(END_NEURAL_DATA_AFTER_WORD_ONSET * SAMPLING_RATE)
        word_electrode_data = self.subject.get_electrode_data(self.electrode_label, trial_id, window_from=est_idx, window_to=est_end_idx)
        return word_electrode_data


    def get_bins_for_single_word(self, trial_id, row, num_bins=10, bin_size=int(0.150 * SAMPLING_RATE)):
        """Extracts time-binned data from the specified electrode for a given word index in a given trial."""

        word_onset = int(row['est_idx'])
        start_idx = word_onset - int(0.500 * SAMPLING_RATE)  # 500ms before onset
        end_idx = start_idx + (num_bins * bin_size)

        word_electrode_data = self.subject.get_electrode_data(self.electrode_label, trial_id, window_from=start_idx, window_to=end_idx)

        # Store bins in a dictionary
        binned_data = {f"bin_{bin_idx}": word_electrode_data[bin_idx * bin_size : (bin_idx+1) * bin_size] for bin_idx in range(num_bins)}

        return binned_data


    def get_all_word_data(self, trial_id):
        """Extracts data for all words from a specific trial for this electrode."""
        words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{self.subject_id}_trial{trial_id}_words_df.csv")
        words_df = pd.read_csv(words_df_path)
        print('trial', trial_id, 'words df collected!')

        # Initialize dictionary where each bin stores a list of all words' data
        binned_data_per_trial = {f"bin_{i}": [] for i in range(10)}

        for idx, row in words_df.iterrows():
            if idx % 100 == 0: 
                print(idx)
            word_bins = self.get_bins_for_single_word(trial_id, row)  # ✅ Fixed missing args
            for bin_id, data in word_bins.items():
                binned_data_per_trial[bin_id].append(data)

        return binned_data_per_trial


    def get_all_words_all_trials(self):
        """Extracts data for all words across the specified trials and organizes it in a dictionary."""
        all_trials_data = {}
        for trial_id in self.trial_ids:
            print('trial', trial_id, 'process beginning')
            trial_word_data = self.get_all_word_data(trial_id)
            all_trials_data[trial_id] = trial_word_data  # Store bins grouped by trial
            print('trial', trial_id, 'process ended')
        return all_trials_data


    def __getitem__(self, trial_id, idx):
        return self.get_all_word_data(self, trial_id)  # Return just the electrode data for a trial

def print_dict_structure(d, indent=0):
    """Recursively prints the structure and lengths of a nested dictionary."""
    for key, value in d.items():
        if isinstance(value, dict):  # If it's another dictionary, go deeper
            print("  " * indent + f"{key}: (dict)")
            print_dict_structure(value, indent + 1)
        elif isinstance(value, list):  # If it's a list, print its length
            print("  " * indent + f"{key}: (list of length {len(value)})")
        elif isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):  # If it's an array, print shape
            print("  " * indent + f"{key}: (tensor of shape {value.shape})")
        else:  # Otherwise, just print type
            print("  " * indent + f"{key}: ({type(value).__name__})")

def get_train_test_splits(all_trials_data, bin_id):
    """Create train-test splits for a specific bin using Leave-One-Out CV (k=3)."""
    splits = []
    trial_ids = list(all_trials_data.keys())  # List of trials

    for test_trial in trial_ids:
        train_trials = [t for t in trial_ids if t != test_trial]
        
        # Get train data: Concatenate all words from the two training trials
        train_data = np.concatenate([all_trials_data[t][bin_id] for t in train_trials])

        # Get test data: Take all words from the test trial
        test_data = np.array(all_trials_data[test_trial][bin_id])
        print(f"For bin {bin_id}, Test Trial: {test_trial} | Train Size: {train_data.shape}, Test Size: {test_data.shape}")

        splits.append((train_data, test_data))

    return splits  # List of (train_data, test_data) tuples





def run_classification(dataset, all_data):
    """
    Runs logistic regression classification on the dataset, using a leave-one-out approach for trials.
    Handles class imbalance, feature scaling, and optimizer convergence issues.
    """
    bin_accuracies = {}  # Store final averaged accuracies

    for bin_id in all_data[0]:  # Iterate over bins
        print(f"\nProcessing {bin_id}...")

        fold_accuracies = []

        # Leave-one-out cross-validation (train on two trials, test on the third)
        for test_trial in dataset.trial_ids:
            print(f"\nFor {bin_id}, Test Trial: {test_trial}")

            # Train on all trials except the test trial
            train_trials = [t for t in dataset.trial_ids if t != test_trial]
            X_train, y_train = [], []
            X_test, y_test = [], []

            for trial_id in dataset.trial_ids:
                words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{dataset.subject_id}_trial{trial_id}_words_df.csv")
                words_df = pd.read_csv(words_df_path)
                y_labels = words_df["is_onset"].values  # Get labels
                
                # Get binned neural data
                bin_data = np.array(all_data[trial_id][bin_id])  # Shape: (num_words, num_features)

                if trial_id == test_trial:
                    X_test.extend(bin_data)
                    y_test.extend(y_labels)
                else:
                    X_train.extend(bin_data)
                    y_train.extend(y_labels)

            # Convert lists to numpy arrays
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_test, y_test = np.array(X_test), np.array(y_test)

            # Print class distribution (check for imbalance)
            unique, counts = np.unique(y_train, return_counts=True)
            print(f"Training class distribution (trial {test_trial} excluded): {dict(zip(unique, counts))}")

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Handle class imbalance
            class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
            class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

            # Train Logistic Regression Model
            clf = LogisticRegression(random_state=42, max_iter=5000, class_weight=class_weight_dict)
            clf.fit(X_train, y_train)

            # Evaluate accuracy
            accuracy = clf.score(X_test, y_test)
            print(f"For {bin_id}, Test Trial: {test_trial} | Accuracy: {accuracy:.3f}")

            fold_accuracies.append(accuracy)

        # Compute average accuracy across folds
        avg_accuracy = np.mean(fold_accuracies)
        bin_accuracies[bin_id] = avg_accuracy

    # Print final results
    print("\n=== FINAL AVERAGE ACCURACIES ACROSS FOLDS ===")
    for bin_id, avg_acc in bin_accuracies.items():
        print(f"{bin_id}: Average Test Accuracy = {avg_acc:.3f}")

    return bin_accuracies







import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def run_regression(dataset, all_data, task_label="pitch"):
    """
    Runs regression for a given task on preloaded binned neural data.

    Args:
        dataset: The SingleElectrodeDataset instance.
        all_data (dict): Dictionary containing neural data for all bins & trials.
        task_label (str): The label to predict (e.g., "pitch").

    Returns:
        final_scores (dict): Average scores across folds for each bin.
    """
    
    bin_scores = {f"bin_{i}": [] for i in range(10)}  # Store scores per bin

    for bin_id in all_data[0].keys():  # Iterate over bins
        print(f"\nProcessing {bin_id}...\n")

        for test_trial in dataset.trial_ids:
            # Initialize train & test sets
            X_train, y_train, X_test, y_test = [], [], [], []

            for trial_id in dataset.trial_ids:
                bin_data = all_data[trial_id][bin_id]  # Extract bin data
                words_df_path = os.path.join(SAVE_SUBJECT_TRIAL_DF_DIR, f"subject{dataset.subject_id}_trial{trial_id}_words_df.csv")
                words_df = pd.read_csv(words_df_path)

                labels = words_df[task_label].to_numpy()  # Extract labels for task

                if trial_id == test_trial:
                    X_test.extend(bin_data)
                    y_test.extend(labels)
                else:
                    X_train.extend(bin_data)
                    y_train.extend(labels)

            # Convert lists to NumPy arrays
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_test, y_test = np.array(X_test), np.array(y_test)

            print(f"For bin {bin_id}, Test Trial: {test_trial} | Train Size: {X_train.shape}, Test Size: {X_test.shape}")

            # Train regression model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Evaluate performance
            train_score = model.score(X_train, y_train)  # R² score
            test_score = model.score(X_test, y_test)    # R² score
            bin_scores[bin_id].append(test_score)  # Store fold accuracy for averaging

            print(f"Train R²: {train_score:.3f} | Test R²: {test_score:.3f}")

    # **Compute Averages**
    final_scores = {bin_id: np.mean(scores) for bin_id, scores in bin_scores.items()}

    print("\n=== FINAL AVERAGE ACCURACIES ACROSS FOLDS ===")
    for bin_id, avg_score in final_scores.items():
        print(f"{bin_id}: Average Test R² = {avg_score:.3f}")

    return final_scores  # Return dictionary of averaged scores







if __name__ == "__main__":
    # Initialize subject
    subject_id = 3
    subject = Subject(subject_id=subject_id, cache=True)

    # Get list of available trial IDs (movies)
    trial_ids = [0, 1, 2]  # Example: Modify this to include actual trial numbers

    # Choose an electrode
    electrode_label = subject.electrode_labels[0]

    # Create dataset
    dataset = SingleElectrodeDataset(subject, trial_ids, dtype=torch.float32, eval_name="pitch", electrode_label=electrode_label)
    all_data = dataset.get_all_words_all_trials()
    # Check dataset size
    print(f"Dataset contains {len(all_data.keys()), len(all_data[0]), len(all_data[0]['bin_0'])} words from multiple movies.")
    print('')
    print_dict_structure(all_data)
    # Get first word's neural data
    #word_data = dataset[0][0]
    #print(f"Neural data shape for first word: {word_data.shape}")
    for bin_id in range(10):
        bin_name = f"bin_{bin_id}"
        print(f"\nProcessing {bin_name}...\n")
        train_test_splits = get_train_test_splits(all_data, bin_name)
    
    trial_id = random.choice(list(all_data.keys()))  # Random trial
    bin_id = random.choice(list(all_data[trial_id].keys()))  # Random bin
    word_idx = random.randint(0, len(all_data[trial_id][bin_id]) - 1)  # Random word

    sample_data = all_data[trial_id][bin_id][word_idx]
    print(f"Random Sample from Trial {trial_id}, Bin {bin_id}, Word {word_idx}:")
    print(f"Shape: {sample_data.shape} → First 10 values: {sample_data[:10]}")
    print('-'*80)
    print('-'*80)
    print('-'*80)
    regs = run_classification(dataset, all_data)
    print(regs)
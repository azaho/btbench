import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
import numpy as np


def generate_splits_DS_DT(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32):

    _all_subject_trials = [
        (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
        (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4),
        (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)
    ]

    movie_subject_mapping = {}
    for subject_id, trial_id in _all_subject_trials:
        if trial_id not in movie_subject_mapping:
            movie_subject_mapping[trial_id] = []
        movie_subject_mapping[trial_id].append(subject_id)

    # ✅ Step 2: Select ONE specific movie for LOSO cross-validation
    selected_movie = 0  # Change this to pick a different movie
    subject_list = movie_subject_mapping[selected_movie]

    print(f"\n🎬 Selected Movie {selected_movie} - Subjects: {subject_list}")

    # ✅ Step 3: Run Leave-One-Subject-Out for this movie
    for test_subject in subject_list:
        print(f"\n🚀 Training on all subjects except {test_subject}, testing on Subject {test_subject}")

        # ✅ Step 4: Load datasets
        train_datasets = []
        test_dataset = None

        for subject_id in subject_list:
            subject = Subject(subject_id, cache=True)
            dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, selected_movie, dtype=torch.float32, eval_name="rms")

            if subject_id == test_subject:
                test_dataset = dataset  # Leave one subject out
            else:
                train_datasets.append(dataset)  # Use others for training

        # ✅ Step 5: Combine training datasets
        train_dataset = ConcatDataset(train_datasets)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32)

        print(f"✅ Train Subjects: {len(train_datasets)} - Train Samples: {len(train_dataset)}")
        print(f"✅ Test Subject: {test_subject} - Test Samples: {len(test_dataset)}")



def generate_splits_SS_DT(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32):
    """Generate train/test splits for Single Subject Different Trials (SS-DT) evaluation.
    
    This function creates train/test splits by using one trial as the test set and all other
    trials from the same subject as the training set. Unlike SS-ST, this does not perform
    k-fold cross validation since trials are already naturally separated.

    Args:
        subject (Subject): Subject object containing brain recording data
        trial_id (int): ID of the trial/movie to use as test set
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        k_folds (int, optional): Not used in this function but kept for API consistency. Defaults to 5.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.

    Returns:
        tuple: A tuple containing:
            - train_dataset (ConcatDataset): Combined dataset of all training trials
            - test_dataset (Dataset): Dataset for the test trial
    """
    # Define subject-to-trials mapping
    _subject_trials = {
        1: [0, 1, 2],
        2: [0, 1, 2, 3, 4, 5, 6],
        3: [0, 1, 2],
        4: [0, 1, 2],
        5: [0],
        6: [0, 1, 4],
        7: [0, 1],
        8: [0],
        9: [0],
        10: [0, 1],
    }

    # Choose subject
    subject_id = subject.subject_id  # Change this to test different subjects
    if subject_id not in _subject_trials: raise ValueError(f"Subject {subject_id} not found in dataset.")
    for test_trial in _subject_trials[subject_id]:
        train_trials = [t for t in _subject_trials[subject_id] if t != test_trial]
        if len(train_trials) == 0: raise ValueError(f"Subject {subject_id} has no training trials.")
        
        test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, test_trial, dtype=dtype, eval_name=eval_name)
        # Load training datasets dynamically
        train_datasets = []
        for train_trial in train_trials:
            train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, train_trial, dtype=dtype, eval_name=eval_name)
            train_datasets.append(train_dataset)
        train_dataset = ConcatDataset(train_datasets)
    return train_dataset, test_dataset


def generate_splits_SS_ST(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32, gap_length=300):
    """Generate train/test splits for Single Subject Single Trial (SS-ST) evaluation.
    
    This function performs k-fold cross validation on data from a single subject and trial,
    ensuring temporal gaps of at least 300 seconds between train and test sets to avoid
    temporal correlation. It also maintains a test/train ratio between 0.18 and 0.22.

    Args:
        subject (Subject): Subject object containing brain recording data
        trial_id (int): ID of the trial/movie to use
        eval_name (str): Name of the evaluation metric to use (e.g. "rms")
        k_folds (int, optional): Number of folds for cross validation. Defaults to 5.
        dtype (torch.dtype, optional): Data type for tensors. Defaults to torch.float32.
        verbose (bool, optional): Whether to print detailed split information. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - train_datasets (list): List of k training dataset splits
            - test_datasets (list): List of k test dataset splits, which correspond to the train datasets in the array above
    """
    train_datasets = []
    test_datasets = []

    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=dtype, eval_name=eval_name)
    kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important to avoid correlated train/test splits!

    # Get word start times & texts from dataset
    word_start_times = dataset.all_words_df["start"].to_numpy()
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        # Skip empty splits
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue

        # Get initial test boundaries
        test_earliest_idx = test_idx[0]
        test_latest_idx = test_idx[-1]
        test_start_time = word_start_times[test_earliest_idx]
        test_end_time = word_start_times[test_latest_idx]

        # Create gaps by filtering train indices
        train_idx = np.array([
            i for i in train_idx 
            if word_start_times[i] <= test_start_time - gap_length  # Before test set with gap
            or word_start_times[i] >= test_end_time + gap_length    # After test set with gap
        ])
        
        train_datasets.append(Subset(dataset, train_idx))
        test_datasets.append(Subset(dataset, test_idx))

    return train_datasets, test_datasets


if __name__ == "__main__":
    subject = Subject(3, cache=False)
    train_datasets, test_datasets = generate_splits_SS_ST(subject, 0, "rms", verbose=False)
    print(train_datasets)
    print(test_datasets)
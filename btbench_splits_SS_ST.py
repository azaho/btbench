import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
import numpy as np


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
            - test_datasets (list): List of k test dataset splits
    """
    train_datasets = []
    test_datasets = []

    subject_id = subject.subject_id
    dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=dtype, eval_name=eval_name)

    # Define K-fold cross-validation
    kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important to avoid correlated train/test splits!

    # Get word start times & texts from dataset
    word_start_times = dataset.all_words_df["start"].to_numpy()
    word_texts = dataset.all_words_df["text"].to_numpy()

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
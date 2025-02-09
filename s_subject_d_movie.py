import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset

# Define subject-to-trials mapping
subject_trials = {
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

def generate_splits_SS_DT(subject, trial_id, eval_name, k_folds=5, dtype=torch.float32):
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

    # Choose subject
    subject_id = subject.subject_id  # Change this to test different subjects
    if subject_id not in subject_trials: raise ValueError(f"Subject {subject_id} not found in dataset.")

    for test_trial in subject_trials[subject_id]:
        train_trials = [t for t in subject_trials[subject_id] if t != test_trial]
        if len(train_trials) == 0: raise ValueError(f"Subject {subject_id} has no training trials.")



        test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, test_trial, dtype=dtype, eval_name=eval_name)

        # Load training datasets dynamically
        train_datasets = []
        for train_trial in train_trials:
            train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, train_trial, dtype=dtype, eval_name=eval_name)
            train_datasets.append(train_dataset)

        # Combine training datasets
        if train_datasets:
            train_dataset = ConcatDataset(train_datasets)
        else:
            train_dataset = None  # No training data available

        # Print dataset sizes
        train_size = len(train_dataset) if train_dataset else 0
            print(f"Subject {subject_id} - Leave-One-Trial-Out CV")
        print(f"Train Trials: {train_trials}")
        print(f"Test Trial: {test_trial}")
        print(f"Train Size: {train_size}, Test Size: {len(test_dataset)}")
        print("-" * 80)

        # Define DataLoaders (if needed for training)
        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

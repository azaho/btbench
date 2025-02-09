import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
import numpy as np

subject_id, trial_id = 2, 6
subject = Subject(subject_id, cache=True)
dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name="rms")

# Define K-fold cross-validation
k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=False)  # shuffle=False is important!

# Get word start times & texts from dataset
word_start_times = dataset.all_words_df["start"].to_numpy()
word_texts = dataset.all_words_df["text"].to_numpy()
total_time = word_start_times[-1]  # Total duration of the movie
MAX_ITERS = 1000  # Prevent infinite loops

for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):
    train_dataset = Subset(dataset, train_idx)
    test_dataset = Subset(dataset, test_idx)

    if len(test_idx) > 0 and len(train_idx) > 0:
        test_earliest_idx = test_idx[0]
        test_latest_idx = test_idx[-1]
        train_latest_before_test_idx = max([i for i in train_idx if word_start_times[i] < word_start_times[test_earliest_idx]], default=None)
        train_earliest_after_test_idx = min([i for i in train_idx if word_start_times[i] > word_start_times[test_latest_idx]], default=None)

        iterations = 0  # Prevent infinite loops
        while iterations < MAX_ITERS:
            test_ratio = len(test_idx) / len(train_idx) if len(train_idx) > 0 else float("inf")
            gap_before = (word_start_times[test_earliest_idx] - word_start_times[train_latest_before_test_idx]) if train_latest_before_test_idx is not None else float("inf")
            gap_after = (word_start_times[train_earliest_after_test_idx] - word_start_times[test_latest_idx]) if train_earliest_after_test_idx is not None else float("inf")

            # ✅ Exit loop when both conditions are met
            if gap_before >= 300 and gap_after >= 300 and 0.18 <= test_ratio <= 0.22:
                break  

            # ✅ If ratio is too big, pop from test set
            if test_ratio >= 0.2:
                if gap_before < 300 and len(test_idx) > 1:
                    test_idx = test_idx[1:]  # Remove early test words
                    test_earliest_idx = test_idx[0] if len(test_idx) > 0 else None

                if gap_after < 300 and len(test_idx) > 1:
                    test_idx = test_idx[:-1]  # Remove late test words
                    test_latest_idx = test_idx[-1] if len(test_idx) > 0 else None

            # ✅ If ratio is too small, pop from train set instead
            else:
                if gap_before < 300 and len(train_idx) > 1:
                    train_idx = train_idx[:-1]  # Remove last train word
                    train_latest_before_test_idx = max([i for i in train_idx if word_start_times[i] < word_start_times[test_earliest_idx]], default=None)

                if gap_after < 300 and len(train_idx) > 1:
                    train_idx = train_idx[1:]  # Remove first train word
                    train_earliest_after_test_idx = min([i for i in train_idx if word_start_times[i] > word_start_times[test_latest_idx]], default=None)

            iterations += 1  # Prevent infinite loops

        # ✅ If we failed to make a good split, shift test set instead of shrinking it too much
        if iterations >= MAX_ITERS:
            mid_point = len(train_idx) // 2
            test_idx = train_idx[mid_point:mid_point + len(train_idx) // k_folds]
            train_idx = np.array([i for i in train_idx if i not in test_idx])
    
        # ✅ Print Results
        print(f"Fold {fold + 1}")

        if fold > 0 and train_latest_before_test_idx is not None:
            print(f"Gap before test set: '{word_texts[train_latest_before_test_idx]}' at {word_start_times[train_latest_before_test_idx]:.2f}s → '{word_texts[test_earliest_idx]}' at {word_start_times[test_earliest_idx]:.2f}s")

        if fold < k_folds - 1 and train_earliest_after_test_idx is not None:
            print(f"Gap after test set: '{word_texts[test_latest_idx]}' at {word_start_times[test_latest_idx]:.2f}s → '{word_texts[train_earliest_after_test_idx]}' at {word_start_times[train_earliest_after_test_idx]:.2f}s")
    # ✅ Convert updated indices back into PyTorch Subset datasets
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=32, shuffle=True)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32)

    # ✅ Print train-test split sizes
    print(f"Train size: {len(train_loader)}, Test size: {len(test_loader)}, Train-Test Ratio: {len(test_dataset) / len(train_dataset):.3f}")
    print("-" * 80)
    # ✅ Print train-test split sizes
    # print(f"Train size: {len(train_idx)}, Test size: {len(test_idx)}, Train-Test Ratio: {len(test_idx) / len(train_idx):.3f}")
    # print("-" * 80)

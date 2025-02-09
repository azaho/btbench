import torch
from torch.utils.data import DataLoader, ConcatDataset
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset

# ✅ Step 1: Define all (subject, movie) pairs
all_subject_trials = [
    (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4),
    (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)
]

# ✅ Step 2: Iterate over (subject, movie) pairs for Leave-One-Subject-One-Movie-Out CV
for test_subject, test_movie in all_subject_trials:
    # Train on all other (subject, movie) pairs
    train_pairs = [(s, m) for (s, m) in all_subject_trials if s != test_subject and m != test_movie]

    print(f"\n🎬 **DS-DM Leave-One-Subject-One-Movie-Out CV**")
    print(f"🎯 **Test Subject:** {test_subject}, **Test Movie:** {test_movie}")
    print(f"📽 **Train Pairs:** {len(train_pairs)} subject-movie combinations")

    # ✅ Load Test Dataset (1 Subject, 1 Movie)
    test_subject_obj = Subject(test_subject, cache=False)
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(test_subject_obj, test_movie, dtype=torch.float32, eval_name="rms")

    # ✅ Load and Combine Training Datasets
    train_datasets = []
    for train_subject, train_movie in train_pairs:
        train_subject_obj = Subject(train_subject, cache=False)
        train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(train_subject_obj, train_movie, dtype=torch.float32, eval_name="rms")
        train_datasets.append(train_dataset)

    # ✅ Combine datasets into DataLoader
    train_dataset = ConcatDataset(train_datasets)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # ✅ Print dataset sizes
    print(f"📊 Train Size: {len(train_dataset)}, Test Size: {len(test_dataset)}")
    print("-" * 80)
    
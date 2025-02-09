import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset

# ✅ Step 1: Define the mapping of movies to subjects
all_subject_trials = [
    (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
    (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4),
    (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)
]

movie_subject_mapping = {}
for subject_id, trial_id in all_subject_trials:
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

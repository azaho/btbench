import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset

# Define subject-to-movies mapping
subject_movies = {
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
subject_id = 2  # Change this to test different subjects

if subject_id not in subject_movies:
    raise ValueError(f"Subject {subject_id} not found in dataset.")

for test_movie in subject_movies[subject_id]:
    train_movies = [m for m in subject_movies[subject_id] if m != test_movie]

    print(f"Subject {subject_id} - Leave-One-Movie-Out CV")
    print(f"Train Movies: {train_movies}")
    print(f"Test Movie: {test_movie}")

    # Load test dataset (Only 1 movie at a time to save memory)
    subject = Subject(subject_id, cache=False)  # Avoid caching to reduce memory usage
    test_dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, test_movie, dtype=torch.float32, eval_name="rms")

    # Load training datasets dynamically
    train_datasets = []
    for train_movie in train_movies:
        train_dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, train_movie, dtype=torch.float32, eval_name="rms")
        train_datasets.append(train_dataset)

    # Combine training datasets
    if train_datasets:
        train_dataset = ConcatDataset(train_datasets)
    else:
        train_dataset = None  # No training data available

    # Print dataset sizes
    train_size = len(train_dataset) if train_dataset else 0
    print(f"Train Size: {train_size}, Test Size: {len(test_dataset)}")
    print("-" * 80)

    # Define DataLoaders (if needed for training)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) if train_dataset else None
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

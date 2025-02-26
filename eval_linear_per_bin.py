# %%
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import json
import glob
import random
import argparse
from btbench_config import *
from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--eval_name', type=str, default='onset', help='Evaluation name (e.g. onset, gpt2_surprisal)')
parser.add_argument('--fold', type=int, choices=[1,2,3,4,5], required=True, help='Fold number (1-5)')
parser.add_argument('--subject', type=int, required=True, help='Subject ID')
parser.add_argument('--trial', type=int, required=True, help='Trial ID')

args = parser.parse_args()

# all possible pairs of (subject_id, trial_id)
all_subject_trials = [(1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 0), (3, 1), (3, 2), (4, 0), (4, 1), (4, 2), (5, 0), (6, 0), (6, 1), (6, 4), (7, 0), (7, 1), (8, 0), (9, 0), (10, 0), (10, 1)]

print(f"Running evaluation for subject {args.subject}, trial {args.trial}, eval {args.eval_name}, fold {args.fold}")

# Check if output files already exist
output_files = [
    f"eval_results_ss_sm/per_bin_linear_voltage_{args.eval_name}_subject{args.subject}_trial{args.trial}_fold{args.fold}_train.npy",
    f"eval_results_ss_sm/per_bin_linear_voltage_{args.eval_name}_subject{args.subject}_trial{args.trial}_fold{args.fold}_test.npy"
]
for file in output_files:
    if os.path.exists(file):
        print(f"Output file {file} already exists. Skipping...")
        exit(0)

print("Loading the dataset...")
# %%
subject_id, trial_id = args.subject, args.trial
subject = Subject(subject_id, cache=True) # use cache=True to load this trial's neural data into RAM, if you have enough memory!
dataset = BrainTreebankSubjectTrialBenchmarkDataset(subject, trial_id, dtype=torch.float32, eval_name=args.eval_name)

# %%
print("Items in the dataset:", len(dataset))
print("Shape of the first item: features.shape =", dataset[0][0].shape, "label =", dataset[0][1])

# %%
from scipy import signal
def compute_spectrogram(data, fs=2048, max_freq=2000):
    """Compute spectrogram for a single trial of data.
    
    Args:
        data (numpy.ndarray): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
    
    Returns:
        numpy.ndarray: Spectrogram representation
    """
    # For 1 second of data at 2048Hz, we'll use larger window
    nperseg = 512  # 250ms window
    noverlap = 0  # 0% overlap
    
    f, t, Sxx = signal.spectrogram(
        data, 
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window='boxcar'
    )
    
    return np.log10(Sxx[:, (f<max_freq) & (f>0)] + 1e-10)

# %%
word_onset_idx = int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE) # 1024
word_onset_idx_end = int((START_NEURAL_DATA_BEFORE_WORD_ONSET + 1) * SAMPLING_RATE) # 3072
word_onset_idx, word_onset_idx_end = 512, 3072
spectrogram = False

print("Loading dataset...")
# Convert PyTorch dataset to numpy arrays for scikit-learn
X = []
y = []

for i in range(len(dataset)):
    features, label = dataset[i]
    features = features.numpy()[:, word_onset_idx:word_onset_idx_end]
    if spectrogram:
        features = compute_spectrogram(features)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Normalize each electrode (axis 1) across all samples and timepoints
print("Normalizing dataset...")

mean = X.mean(axis=(0,2), keepdims=True)  # Mean across samples and time
std = X.std(axis=(0,2), keepdims=True)    # Std across samples and time
X = (X - mean) / (std + 1e-10)          # Add small constant to avoid div by 0

print("Dataset loaded")
print("Shape of X:", X.shape, "shape of y:", y.shape)

# %%
# Train logistic regression with 5-fold CV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=False)

# Create arrays to store ROC AUC scores for each fold and time bin
n_bins = 20
bin_size = 128

train_roc_scores = np.zeros((1, n_bins))
test_roc_scores = np.zeros((1, n_bins))

print("Training and evaluating model for each fold and time bin...")

# %%

print("\nProcessing dataset...")
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
    # Skip folds that don't match the specified fold number
    if fold_idx + 1 != args.fold:
        continue
        
    print(f"\nProcessing fold {fold_idx + 1}/{n_folds}")
    
    # Split data for this fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    for bin_idx in range(n_bins):
        # Reshape data for current time bin
        if spectrogram:
            X_train_bin = X_train[:, :, :, bin_idx].reshape(X_train.shape[0], -1)
            X_test_bin = X_test[:, :, :, bin_idx].reshape(X_test.shape[0], -1)
        else:
            start_idx = bin_idx * bin_size
            end_idx = start_idx + bin_size
            X_train_bin = X_train[:, :, start_idx:end_idx].reshape(X_train.shape[0], -1)
            X_test_bin = X_test[:, :, start_idx:end_idx].reshape(X_test.shape[0], -1)
        
        # Train model
        clf = LogisticRegression(random_state=42, max_iter=1000, n_jobs=5, solver='lbfgs', verbose=1, tol=1e-3)
        clf.fit(X_train_bin, y_train)
        
        # Get predictions - modified for multiclass
        train_probs = clf.predict_proba(X_train_bin)
        test_probs = clf.predict_proba(X_test_bin)
        
        # Calculate and store ROC AUC scores
        # For multiclass, roc_auc_score will handle the OvR calculation internally
        train_roc_scores[0, bin_idx] = roc_auc_score(y_train, train_probs, multi_class='ovr')
        test_roc_scores[0, bin_idx] = roc_auc_score(y_test, test_probs, multi_class='ovr')
        
        print(f"Bin {bin_idx}: Train ROC AUC = {train_roc_scores[0, bin_idx]:.3f}, "
              f"Test ROC AUC = {test_roc_scores[0, bin_idx]:.3f}")

print("\nCompleted evaluation for all folds and time bins")

# Save ROC AUC scores to files
print("\nSaving results...")
np.save(f"eval_results_ss_sm/per_bin_linear_voltage_{args.eval_name}_subject{args.subject}_trial{args.trial}_fold{args.fold}_train.npy", train_roc_scores)
np.save(f"eval_results_ss_sm/per_bin_linear_voltage_{args.eval_name}_subject{args.subject}_trial{args.trial}_fold{args.fold}_test.npy", test_roc_scores)
print("Results saved successfully")

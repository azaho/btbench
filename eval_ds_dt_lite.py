
import numpy as np
import torch
import h5py
import pandas as pd
from tqdm import tqdm
import gc
import argparse
import os
import json
from datetime import datetime
import random

from braintreebank_subject import Subject
from btbench_datasets import BrainTreebankSubjectTrialBenchmarkDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PREPROCESSED_DIR = "preprocessed_regions_desikan_killiany"
FIXED_TRAIN_SUBJECT = 2
FIXED_TRAIN_TRIAL = 4

def extract_features_and_labels(subject_id, trial_id, eval_name, region_h5_path, lite=True):
    """
    Uses the Dataset's output_indices mode to extract features and labels from preprocessed region data.
    lite parameter to use BTBench-Lite.
    """
    with h5py.File(region_h5_path, 'r') as f:
        region_data = f['region_data'][:]  # shape: (n_regions, n_timepoints)
    subject = Subject(subject_id, cache=False)
    dataset = BrainTreebankSubjectTrialBenchmarkDataset(
        subject, trial_id, dtype=np.float32, eval_name=eval_name, 
        output_indices=True, lite=lite,
 # Add metadata_dir parameter
    )
    X_features = []
    y_labels = []
    for (start_idx, end_idx), label in dataset:
        segment = region_data[:, int(start_idx):int(end_idx)]
        X_features.append(segment.flatten())
        y_labels.append(label)
    return np.array(X_features), np.array(y_labels)

def evaluate_fixed_train(test_subject_id, test_trial_id, eval_name, lite=True):
    """
    Evaluate using fixed training data (S2T4) on test data, using BTBench-Lite if specified.
    """
    split_type = 'DS-DT-Fixed-Lite' if lite else 'DS-DT-Fixed'
    print(f"\nEvaluating {split_type} using FIXED TRAIN (S{FIXED_TRAIN_SUBJECT} T{FIXED_TRAIN_TRIAL}) "
          f"-> Test (S{test_subject_id} T{test_trial_id}), Task: {eval_name}")

    if test_subject_id == FIXED_TRAIN_SUBJECT:
        print(f"ERROR: Cannot test on Subject {FIXED_TRAIN_SUBJECT} as it's the fixed training subject.")
        return {'error': f'Test subject cannot be {FIXED_TRAIN_SUBJECT}'}

    # --- Load Training Data ---
    train_region_h5 = f"{PREPROCESSED_DIR}/sub_{FIXED_TRAIN_SUBJECT}_trial{FIXED_TRAIN_TRIAL:03}_regions.h5"
    X_train, y_train = extract_features_and_labels(
        FIXED_TRAIN_SUBJECT, FIXED_TRAIN_TRIAL, eval_name, train_region_h5, lite=lite
    )
    if X_train is None or len(X_train) == 0:
        print("CRITICAL ERROR: Failed to load fixed training data. Aborting.")
        return {'error': 'Failed to load fixed training data'}
    print(f"Loaded training data. Shape: {X_train.shape}")

    # --- Load Test Data ---
    test_region_h5 = f"{PREPROCESSED_DIR}/sub_{test_subject_id}_trial{test_trial_id:03}_regions.h5"
    X_test, y_test = extract_features_and_labels(
        test_subject_id, test_trial_id, eval_name, test_region_h5, lite=lite
    )
    if X_test is None or len(X_test) == 0:
        print("CRITICAL ERROR: Failed to load or empty test data. Aborting.")
        return {'error': 'Failed/Empty test data'}
    print(f"Loaded test data. Shape: {X_test.shape}")

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Train and Evaluate ---
    model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    accuracy = model.score(X_test_scaled, y_test)
    auroc = np.nan
    y_scores = model.predict_proba(X_test_scaled)
    if len(np.unique(y_test)) == 2:
        auroc = roc_auc_score(y_test, y_scores[:, 1])
    elif len(np.unique(y_test)) > 2:
        try:
            auroc = roc_auc_score(y_test, y_scores, multi_class='ovr')
        except Exception as e:
            print(f"Could not compute multiclass AUROC: {e}")
            auroc = np.nan

    final_results = {
        'train_subject': FIXED_TRAIN_SUBJECT, 
        'train_trial': FIXED_TRAIN_TRIAL,
        'test_subject': test_subject_id, 
        'test_trial': test_trial_id,
        'eval_name': eval_name, 
        'split_type': split_type,
        'n_train_samples': len(X_train),
        'n_test_samples': len(X_test),
        'final_accuracy': float(accuracy),
        'final_auroc': float(auroc) if not np.isnan(auroc) else None,
        'lite': lite
    }
    return final_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FIXED TRAIN (S2T4) cross-subject evaluation using BTBench-Lite.")
    parser.add_argument('--subject', type=int, required=True, help="Test Subject ID (Cannot be 2)")
    parser.add_argument('--trial', type=int, required=True, help="Test Trial ID")
    parser.add_argument('--eval_name', type=str, required=True)
    parser.add_argument('--no_lite', action='store_true', help="Disable BTBench-Lite (use full dataset)")
    args = parser.parse_args()

    # Call the evaluation function
    results = evaluate_fixed_train(
        args.subject, 
        args.trial, 
        args.eval_name,
        lite=not args.no_lite  # Use lite by default unless --no_lite is specified
    )

    # Save results
    output_dir = 'eval_results_ds_dt_lite_desikan_killiany'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{output_dir}/DS-DT-FixedTrain-Lite_{args.eval_name}_test_S{args.subject}T{args.trial}_{timestamp}.json"
    print(f"Saving results to {filename}...")
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)
    print("Fixed Train evaluation script finished.")
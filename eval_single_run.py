import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from btbench_train_test_splits import generate_splits_SS_ST
from braintreebank_subject import Subject
from sklearn.metrics import roc_auc_score
import os
import json
from datetime import datetime


def run_linear_classification(subject_id, trial_id, eval_name, k_folds=5):
    """Run linear classification for a given subject, trial, and eval_name.
    
    Args:
        subject_id (int): Subject ID
        trial_id (int): Trial ID
        eval_name (str): eval_name name (e.g., "rms" for volume classification)
        k_folds (int): Number of cross-validation folds
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = 'eval_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load subject data
    print(f"Loading subject {subject_id}...")
    subject = Subject(subject_id, cache=True)
    subject.load_neural_data(trial_id)
    
    # Generate train/test splits
    print(f"Generating train/test splits for subject {subject_id}...")
    train_datasets, test_datasets = generate_splits_SS_ST(
        test_subject=subject,
        test_trial_id=trial_id,
        eval_name=eval_name,
        k_folds=k_folds
    )
    
    # Store results for each fold
    fold_accuracies = []
    fold_results = []
    
    for fold, (train_data, test_data) in enumerate(zip(train_datasets, test_datasets)):
        print(f"\nProcessing fold {fold + 1}/{len(train_datasets)}")

        print(f"Train data shape: {train_data[0][0].shape}")
        print(f"Test data shape: {test_data[0][0].shape}")
        print(f"Length of train data: {len(train_data)}")
        print(f"Length of test data: {len(test_data)}")
        
        # Convert dataset to numpy arrays
        # Convert to numpy arrays in one pass
        print("Processing X_train...")
        X_train = np.array([sample[0].numpy() for sample in train_data])
        print("Processing y_train...")
        y_train = np.array([sample[1].numpy() for sample in train_data])
        print("Processing X_test...")
        X_test = np.array([sample[0].numpy() for sample in test_data], dtype=int)
        print("Processing y_test...")
        y_test = np.array([sample[1].numpy() for sample in test_data], dtype=int)

        # Flatten the electrode data if needed
        if len(X_train.shape) > 2:
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        # Train logistic regression
        print("Training logistic regression...")
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy and AUROC
        print("Calculating accuracy and AUROC...")
        accuracy = accuracy_score(y_test, y_pred)
        auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        fold_accuracies.append(accuracy)
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'n_train_samples': len(y_train),
            'n_test_samples': len(y_test),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        })
        
        # Print fold results
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
    
    # Calculate and print overall results
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    print("\nOverall Results:")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Std Accuracy: {std_accuracy:.4f}")
    
    # Save results to JSON
    results = {
        'subject_id': subject_id,
        'trial_id': trial_id,
        'eval_name': eval_name,
        'k_folds': k_folds,
        'mean_accuracy': float(mean_accuracy),
        'std_accuracy': float(std_accuracy),
        'fold_results': fold_results
    }
    
    results_file = os.path.join(output_dir, f'run_{timestamp}_subject{subject_id}_trial{trial_id}_{eval_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear classification on brain data")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID")
    parser.add_argument("--trial", type=int, required=True, help="Trial ID")
    parser.add_argument("--eval_name", type=str, required=True, help="eval_name name (e.g., 'rms')")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    run_linear_classification(
        subject_id=args.subject,
        trial_id=args.trial,
        eval_name=args.eval_name,
        k_folds=args.folds
    )
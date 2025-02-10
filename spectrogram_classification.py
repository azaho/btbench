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
from scipy import signal

def compute_spectrogram(data, fs=2048):
    """Compute spectrogram for a single trial of data.
    
    Args:
        data (numpy.ndarray): Input voltage data of shape (n_samples,)
        fs (int): Sampling frequency in Hz
    
    Returns:
        numpy.ndarray: Spectrogram representation
    """
    # For 1 second of data at 2048Hz, we'll use larger window
    nperseg = 256  # 125ms window
    noverlap = 128  # 50% overlap
    
    f, t, Sxx = signal.spectrogram(
        data, 
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window='boxcar'
    )
    
    # Return log power (adding small constant to avoid log(0))
    return np.log10(Sxx + 1e-10)

def run_spectrogram_classification(subject_id, trial_id, eval_name, k_folds=5):
    """Run linear classification on spectrogram features for a given subject, trial, and eval_name.
    
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
        
        # Convert dataset to numpy arrays and compute spectrograms
        print("Processing training data...")
        X_train_raw = np.array([sample[0].numpy() for sample in train_data])
        y_train = np.array([sample[1] for sample in train_data])
        
        print("Processing test data...")
        X_test_raw = np.array([sample[0].numpy() for sample in test_data])
        y_test = np.array([sample[1] for sample in test_data])
        
        # Compute spectrograms for each sample
        print("Computing spectrograms for training data...")
        X_train_specs = np.array([
            compute_spectrogram(x.flatten()) for x in X_train_raw
        ])
        
        print("Computing spectrograms for test data...")
        X_test_specs = np.array([
            compute_spectrogram(x.flatten()) for x in X_test_raw
        ])
        
        # Flatten the spectrograms for linear classification
        n_train = X_train_specs.shape[0]
        n_test = X_test_specs.shape[0]
        X_train = X_train_specs.reshape(n_train, -1)
        X_test = X_test_specs.reshape(n_test, -1)
        
        # Train logistic regression
        print("Training logistic regression...")
        clf = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        clf.fit(X_train, y_train)
        
        # Make predictions
        print("Making predictions...")
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy and AUROC
        print("Calculating accuracy and AUROC...")
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate AUROC for multi-class using one-vs-rest approach
        n_classes = len(np.unique(y_test))
        if n_classes == 2:
            auroc = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        else:
            # For multi-class, calculate AUROC for each class vs rest
            auroc_per_class = []
            y_score = clf.predict_proba(X_test)
            for i in range(n_classes):
                # Create binary labels for current class
                y_binary = (y_test == i).astype(int)
                auroc_per_class.append(roc_auc_score(y_binary, y_score[:, i]))
            auroc = np.mean(auroc_per_class)
        
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
        
        if n_classes > 2:
            fold_results[-1]['auroc_per_class'] = {f'class_{i}': float(auc) for i, auc in enumerate(auroc_per_class)}
        
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
        'fold_results': fold_results,
        'n_classes': int(n_classes)
    }
    
    results_file = os.path.join(output_dir, f'spectrogram_linear_{timestamp}_subject{subject_id}_trial{trial_id}_{eval_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear classification on spectrogram features")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID")
    parser.add_argument("--trial", type=int, required=True, help="Trial ID")
    parser.add_argument("--eval_name", type=str, required=True, help="eval_name name (e.g., 'rms')")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    
    args = parser.parse_args()
    
    run_spectrogram_classification(
        subject_id=args.subject,
        trial_id=args.trial,
        eval_name=args.eval_name,
        k_folds=args.folds
    ) 
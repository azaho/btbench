import argparse
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from btbench_train_test_splits import generate_splits_SS_SM
from braintreebank_subject import Subject
from scipy import signal
from sklearn.metrics import roc_auc_score
import os
import json
from datetime import datetime
import gc
import psutil
from btbench_config import START_NEURAL_DATA_BEFORE_WORD_ONSET, END_NEURAL_DATA_AFTER_WORD_ONSET, SAMPLING_RATE

def compute_spectrogram(data, fs=2048, max_freq=2000, min_freq=0):
    """Compute spectrogram for a single trial of data.
    
    Args:
        data (numpy.ndarray): Input voltage data of shape (n_channels, n_samples) or (batch_size, n_channels, n_samples)
        fs (int): Sampling frequency in Hz
    
    Returns:
        numpy.ndarray: Spectrogram representation
    """
    # For 1 second of data at 2048Hz, we'll use larger window
    nperseg = 256  # 125ms window
    noverlap = 0  # 0% overlap
    
    f, t, Sxx = signal.spectrogram(
        data, 
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        window='boxcar'
    )
    
    return np.log10(Sxx[:, (f<=max_freq) & (f>=min_freq)] + 1e-5)

def run_linear_classification(subject_id, trial_id, eval_name, k_folds=5, spectrogram=False, normalize=False, solver='lbfgs'):
    """Run linear classification for a given subject, trial, and eval_name.
    
    Args:
        subject_id (int): Subject ID
        trial_id (int): Trial ID
        eval_name (str): eval_name name (e.g., "rms" for volume classification)
        k_folds (int): Number of cross-validation folds
    """
    suffix = '_spectrogram' if spectrogram else '_voltage'
    if normalize:
        suffix += '_normalized'
    # Check if results file already exists
    output_dir = 'eval_results_ss_sm_enhanced_pitch'
    # Check if any matching results files exist using glob
    import glob
    results_pattern = os.path.join(output_dir, f'linear{suffix}_*subject{subject_id}_trial{trial_id}_{eval_name}.json')
    matching_files = glob.glob(results_pattern)
    if matching_files:
        print(f"Results file already exists for this configuration. Skipping...")
        return
    process = psutil.Process() # for tracking ram usage

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load subject data
    print(f"Loading subject {subject_id}... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
    subject = Subject(subject_id, cache=True)
    subject.load_neural_data(trial_id)
    
    # Generate train/test splits
    print(f"Generating train/test splits for subject {subject_id}... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
    train_datasets, test_datasets = generate_splits_SS_SM(
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

        sample_time_from, sample_time_to = int(START_NEURAL_DATA_BEFORE_WORD_ONSET * SAMPLING_RATE), int((START_NEURAL_DATA_BEFORE_WORD_ONSET + 1) * SAMPLING_RATE) # get the first second of neural data after word onset
        
        # Convert dataset to numpy arrays
        # Convert to numpy arrays in one pass
        print(f"Processing train data... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        X_train = []
        y_train = []
        for i in range(len(train_data)):
            features, label = train_data[i]
            features = features.numpy()[:, sample_time_from:sample_time_to]
            if spectrogram: features = compute_spectrogram(features)
            X_train.append(features)#.flatten())
            y_train.append(label)
        X_train = np.array(X_train)
        y_train = np.array(y_train, dtype=int)
        print("Train dataset loaded")
        if normalize:
            train_means = X_train.mean(axis=(0,2) if spectrogram else (0,2), keepdims=True)
            train_stds = X_train.std(axis=(0,2) if spectrogram else (0,2), keepdims=True)
            X_train = (X_train - train_means) / (train_stds + 1e-5)
        X_train = X_train.reshape(len(X_train), -1)


        print(f"Processing test data... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        X_test = []
        y_test = []
        for i in range(len(test_data)):
            features, label = test_data[i]
            features = features.numpy()[:, sample_time_from:sample_time_to]
            if spectrogram: features = compute_spectrogram(features)
            X_test.append(features)#.flatten())
            y_test.append(label)
        X_test = np.array(X_test)
        y_test = np.array(y_test, dtype=int)
        print("Test dataset loaded")
        if normalize:
            X_test = (X_test - train_means) / (train_stds + 1e-10)
        X_test = X_test.reshape(len(X_test), -1)
        
        # Train logistic regression with optimized parameters
        print(f"Training logistic regression... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        clf = LogisticRegression(
            max_iter=10000,
            random_state=42,
            solver=solver,  
            n_jobs=4,     # Use 4
            tol=1e-3,      # Slightly less strict convergence criterion
            verbose=2
        )
        clf.fit(X_train, y_train)
        
        # Make predictions
        print(f"Making predictions... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        y_pred = clf.predict(X_test)
        
        # Calculate accuracy and AUROC
        print(f"Calculating accuracy and AUROC... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
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
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f} (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # clean up memory
        del X_train, y_train, X_test, y_test, y_pred
        del clf, accuracy, auroc
        if n_classes > 2:
            del auroc_per_class, y_score, y_binary
        gc.collect()  # Force garbage collection
    
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
    
    results_file = os.path.join(output_dir, f'linear{suffix}_subject{subject_id}_trial{trial_id}_{eval_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear classification on brain data")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID")
    parser.add_argument("--trial", type=int, required=True, help="Trial ID")
    parser.add_argument("--eval_name", type=str, required=True, help="eval_name name (e.g., 'rms')")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--spectrogram", type=int, default=0, help="Whether to compute spectrogram")
    parser.add_argument("--normalize", type=int, default=0, help="Whether to normalize features")
    parser.add_argument("--solver", type=str, default='lbfgs', help="Solver to use for logistic regression")
    
    args = parser.parse_args()
    
    run_linear_classification(
        subject_id=args.subject,
        trial_id=args.trial,
        eval_name=args.eval_name,
        k_folds=args.folds,
        spectrogram=(args.spectrogram==1),
        normalize=(args.normalize==1),
        solver=args.solver
    )
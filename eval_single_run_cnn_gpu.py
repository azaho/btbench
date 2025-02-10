import argparse
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
from btbench_train_test_splits import generate_splits_SS_ST
from braintreebank_subject import Subject
from scipy import signal
from sklearn.metrics import roc_auc_score
import os
import json
from datetime import datetime
import gc
import psutil

def compute_spectrogram(data, fs=2048, max_freq=2000):
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
    
    return np.log10(Sxx[:, (f<max_freq) & (f>0)] + 1e-10)

class MLPClassifierGPU(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)

def run_mlp_classification(subject_id, trial_id, eval_name, k_folds=5, spectrogram=False):
    """Run MLP classification for a given subject, trial, and eval_name.
    
    Args:
        subject_id (int): Subject ID
        trial_id (int): Trial ID
        eval_name (str): eval_name name (e.g., "rms" for volume classification)
        k_folds (int): Number of cross-validation folds
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    suffix = '_spectrogram' if spectrogram else '_voltage'
    # Check if results file already exists
    output_dir = 'eval_results'
    # Check if any matching results files exist using glob
    import glob
    results_pattern = os.path.join(output_dir, f'mlp{suffix}_*_subject{subject_id}_trial{trial_id}_{eval_name}.json')
    matching_files = glob.glob(results_pattern)
    if matching_files:
        print(f"Results file already exists for this configuration. Skipping...")
        return
    process = psutil.Process() # for tracking ram usage

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load subject data
    print(f"Loading subject {subject_id}... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
    subject = Subject(subject_id, cache=True)
    subject.load_neural_data(trial_id)
    
    # Generate train/test splits
    print(f"Generating train/test splits for subject {subject_id}... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
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

        sample_time_from, sample_time_to = 1024, 3072 # get the first second of neural data after word onset
        
        # Convert dataset to tensors
        print(f"Processing train data... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        X_train = []
        y_train = []
        for i in range(len(train_data)):
            features, label = train_data[i]
            features = features[:, sample_time_from:sample_time_to]
            if spectrogram: 
                features = torch.from_numpy(compute_spectrogram(features.numpy())).float()
            X_train.append(features.flatten())
            y_train.append(label)
        X_train = torch.stack(X_train)
        y_train = torch.tensor(y_train, dtype=torch.long)

        print(f"Processing test data... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        X_test = []
        y_test = []
        for i in range(len(test_data)):
            features, label = test_data[i]
            features = features[:, sample_time_from:sample_time_to]
            if spectrogram:
                features = torch.from_numpy(compute_spectrogram(features.numpy())).float()
            X_test.append(features.flatten())
            y_test.append(label)
        X_test = torch.stack(X_test)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # Move data to GPU
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        X_test = X_test.to(device)
        y_test = y_test.to(device)

        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Initialize model
        n_classes = len(torch.unique(y_train))
        model = MLPClassifierGPU(
            input_size=X_train.shape[1],
            hidden_sizes=[256, 128],
            num_classes=n_classes
        ).to(device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        
        # Training loop
        print(f"Training MLP... (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        model.train()
        for epoch in range(100):  # Max 100 epochs
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 1 == 0:
                print(f'Epoch {epoch}, Loss: {total_loss/len(train_loader):.4f}')
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            y_pred = model(X_test)
            y_score = torch.softmax(y_pred, dim=1)
            y_pred = torch.argmax(y_pred, dim=1)
        
        # Convert predictions back to CPU for metric calculation
        y_pred_cpu = y_pred.cpu().numpy()
        y_test_cpu = y_test.cpu().numpy()
        y_score_cpu = y_score.cpu().numpy()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
        
        if n_classes == 2:
            auroc = roc_auc_score(y_test_cpu, y_score_cpu[:, 1])
        else:
            auroc_per_class = []
            for i in range(n_classes):
                y_binary = (y_test_cpu == i).astype(int)
                auroc_per_class.append(roc_auc_score(y_binary, y_score_cpu[:, i]))
            auroc = np.mean(auroc_per_class)
            
        fold_accuracies.append(accuracy)
        
        # Store fold results
        fold_results.append({
            'fold': fold + 1,
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'n_train_samples': len(y_train),
            'n_test_samples': len(y_test),
            'classification_report': classification_report(y_test_cpu, y_pred_cpu, output_dict=True)
        })
        
        if n_classes > 2:
            fold_results[-1]['auroc_per_class'] = {f'class_{i}': float(auc) for i, auc in enumerate(auroc_per_class)}
        
        # Print fold results
        print(f"Fold {fold + 1} Accuracy: {accuracy:.4f} (RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB)")
        print("\nClassification Report:")
        print(classification_report(y_test_cpu, y_pred_cpu))

        # Clean up memory
        del X_train, y_train, X_test, y_test, y_pred
        del model, accuracy, auroc
        if n_classes > 2:
            del auroc_per_class, y_score
        torch.cuda.empty_cache()
        gc.collect()
    
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
    
    results_file = os.path.join(output_dir, f'mlp{suffix}_{timestamp}_subject{subject_id}_trial{trial_id}_{eval_name}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MLP classification on brain data")
    parser.add_argument("--subject", type=int, required=True, help="Subject ID")
    parser.add_argument("--trial", type=int, required=True, help="Trial ID")
    parser.add_argument("--eval_name", type=str, required=True, help="eval_name name (e.g., 'rms')")
    parser.add_argument("--folds", type=int, default=5, help="Number of cross-validation folds")
    parser.add_argument("--spectrogram", type=int, default=0, help="Whether to compute spectrogram")
    
    args = parser.parse_args()
    
    run_mlp_classification(
        subject_id=args.subject,
        trial_id=args.trial,
        eval_name=args.eval_name,
        k_folds=args.folds,
        spectrogram=(args.spectrogram==1)
    )
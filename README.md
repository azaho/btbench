# BrainTreeBenchmark (BT-bench)

BT-bench is a suite of 19 standardized decoding tasks for evaluating foundation models on intracranial brain responses to naturalistic stimuli. The benchmark is based on the BrainTreebank dataset, which contains stereoelectroencephalography (SEEG) recordings from 10 patients watching Hollywood movies.

## Overview

BT-bench enables systematic evaluation of computational models on multimodal neural decoding tasks across:
- Visual features (brightness, motion flow, faces)
- Auditory features (volume, pitch) 
- Language features (speech detection, word properties)
- Multimodal features (speaker identification)

The benchmark includes defined train/test splits for assessing generalization:
| Train/Test Split | Description |
|-----------------|-------------|
| SS-SM | Same Subject - Same Movie |
| SS-DM | Same Subject - Different Movie | 
| DS-SM | Different Subject - Same Movie |
| DS-DM | Different Subject - Different Movie |

## Getting Started

Optionally, create a virtual environment:
```
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

1. Install required packages:
```
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn psutil
```

2. Download and extract the braintreebank dataset (this step can be skipped if the dataset is already downloaded and extracted; it should be all extracted into the braintreebank/ directory):
```
python braintreebank_download_extract.py
```
alternatively, you can specify the path to the braintreebank dataset in the `btbench_config.py` file:
```
ROOT_DIR = "braintreebank" # Root directory for the braintreebank data
```

3. Process the subject trial dataframes:
```
python btbench_process_subject_trial_df.py
```
This command will create the files in a directory called `subject_metadata`.

4. Then, you use the file `example_linear.ipynb` to see how to create a dataset and evaluate a linear model.
To specify which data you want (sentence onset, pitch, etc.), adjust the ‘eval_name’ parameter when defining a BrainTreebankSubjectTrialBenchmarkDataset
object in code block 2. 

### Description of the files in the repository
- `figures/` - directory for files creating the figures and storing the figures
- `subject_metadata/` - directory for the subject metadata (created by the script `btbench_process_subject_trial_df.py`)
- `speech_selectivity_data/` - directory for the speech selectivity data (created by the script `btbench_process_speech_selectivity.py`)
- `btbench_config.py` - configuration file for the benchmark
- `btbench_process_subject_trial_df.py` - script for processing the subject trial metadata
- `btbench_process_speech_selectivity.py` - script for processing the electrode speech selectivity data
- `btbench_datasets.py` - script that defines the BrainTreebankSubjectTrialBenchmarkDataset class, which contains the features/labels for a single subject and trial
- `btbench_train_test_splits.py` - script that defines the train/test splits for the benchmark, for each of the 4 train/test split types (SS-SM, SS-DM, DS-SM, DS-DM)
- `eval_single_run.py` - script for training and evaluating a simple linear model (optionally, after taking a spectrogram of the data) on a given subject and trial, and task
- `braintreebank/` - directory for the braintreebank dataset
- `braintreebank_download_extract.py` - script for downloading and extracting the braintreebank dataset
- `braintreebank_subject.py` - script that defines the Subject class, which contains the data for a single subject
- `example_linear.ipynb` - example of how to create a dataset and evaluate a linear model
- `eval_single_run_cnn_gpu.py` - script for training and evaluating a simple CNN model (optionally, after taking a spectrogram of the data) on a given subject and trial, and task
- `eval_linear_per_bin.py` - script for training and evaluating a simple linear model on separate small timebins around the onset of the word

## Citation

If you use BT-bench in your work, please cite the following paper:
TBD
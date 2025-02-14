# BrainTreeBenchmark (BT-bench)

BT-bench is a suite of 19 standardized decoding tasks for evaluating foundation models on intracranial brain responses to naturalistic stimuli. The benchmark is based on the BrainTreebank dataset, which contains stereoelectroencephalography (SEEG) recordings from 10 patients watching Hollywood movies.

## Overview

BT-bench enables systematic evaluation of computational models on multimodal neural decoding tasks across:
- Visual features (brightness, motion flow, faces)
- Auditory features (volume, pitch) 
- Language features (speech detection, word properties)
- Multimodal features (speaker identification)

The benchmark includes defined train/test splits for assessing generalization:
- Within-subject, within-session
- Within-subject, across-session  
- Across-subject, within-session
- Across-subject, across-session

## Getting Started

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

Then, you use the file `example_linear.ipynb` to see how to create a dataset and train a linear model.
To specify which data you want (sentence onset, pitch, etc.), adjust the ‘eval_name’ parameter when defining a BrainTreebankSubjectTrialBenchmarkDataset
object in code block 2. 

## Citation

If you use BT-bench in your work, please cite the following paper:
TBD
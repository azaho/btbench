# btbench
BrainTreeBenchmark - based on the braintreebank dataset.

To run the script, you need to install the following packages:
```
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn transformers python-dotenv peft datasets
```

Then, download and extract the braintreebank dataset:
```
python braintreebank_download.py
python braintreebank_extract.py
```

Then, process the subject trial dataframes:
```
python btbench_process_subject_trial_df.py
```
This command will create the files in a directory called `btbench_subject_trial_df`.

Then, you use the file `example_linear.ipynb` to see how to create a dataset and train a linear model.

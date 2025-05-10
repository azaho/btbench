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
pip install beautifulsoup4 requests torch torchvision h5py pandas scipy numpy matplotlib seaborn wandb scikit-learn psutil librosa
```

2. Specify the path to the braintreebank dataset (or the path to download it to) in the `btbench_config.py` file: 
```
ROOT_DIR = "braintreebank" # Root directory for the braintreebank data
```
Then, download and extract the braintreebank dataset (this step can be skipped if the dataset is already downloaded and extracted; it should be all extracted into the ROOT_DIR directory):
```
python braintreebank_download_extract.py
```

3. Process the subject trial dataframes:
```
python btbench_process_subject_trials.py
```
This command will create the files in a directory called `btbench_subject_metadata`.

4. Then, you use the file `quickstart.ipynb` to see how to create a dataset and evaluate a linear model.

5. To evaluate the linear regression model on all electrodes and time bins separately, run:
```
python single_electrode.py --subject SUBJECT_ID --trial TRIAL_ID --verbose
```
This command will create a JSON file in the `eval_results` directory with the results, according to the schema in `leaderboard_schema.json`. You can change the `save_dir` argument to save the results to a different directory: `--save_dir SAVE_DIR`.

## PopulationTransformer

Pre-reqs:
- Grab BrainBERT weights from [here](https://github.com/czlwang/BrainBERT) or from `victoria:/storage/czw/self_supervised_seeg/pretrained_weights/stft_large_pretrained.pth` 

### Write the BTBench tasks
- First, let's write all the BrainBERT embeddings for the BTBench tasks to disk
```
python3 -m data.write_multi_subject_multi_channel_btbench \
+data_prep=pretrain_multi_subj_multi_chan_template +data=subject_data_template \
++data_prep.task_name=volume +preprocessor=multi_elec_spec_pretrained \
++data_prep.electrodes=/storage/czw/PopTCameraReadyPrep/electrode_selections/clean_laplacian.json \
++data_prep.brain_runs=/storage/czw/btbench/trial_selections/debug_trials.json \
++data_prep.output_directory=/storage/czw/btbench/saved_examples/btbench_popt_embeds_lite \
++preprocessor.upstream_ckpt=/storage/czw/self_supervised_seeg/pretrained_weights/stft_large_pretrained.pth \
++data.raw_brain_data_dir=/storage/czw/braintreebank_data/ \
++data.movie_transcripts_dir=/storage/czw/braintreebank_data/transcripts
```
Important arguments:
- `data_prep.electrodes` and `data_prep.brain_runs` are in this branch. You should swap out `czw` for your username.
- `data.raw_brain_data_dir` --- the path to the braintreebank

### Fine-tune the PopulationTransformer
Pre-reqs:
- Grab randomized_replacement_no_gaussian_blur.pth from `victoria:/storage/czw/victoria/MultiBrainBERT/outputs`
```
WEIGHTS=randomized_replacement_no_gaussian_blur; python3 run_cross_val.py \
+exp=multi_elec_feature_extract ++exp.runner.save_checkpoints=False ++model.frozen_upstream=False \
+task=btbench_popt +criterion=pt_feature_extract_coords_criterion +data=btbench_decode \
+preprocessor=empty_preprocessor +model=pt_downstream_multiclass \
++model.upstream_path=/storage/czw/PopTCameraReadyPrep/outputs/${WEIGHTS}.pth \
++model.upstream_cfg.use_token_cls_head=True ++model.upstream_cfg.name=pt_model_custom \
++data.btbench_cache_path=/storage/czw/btbench/saved_examples/btbench_popt_embeds_lite \
++data.k_fold=5 ++exp.runner.num_workers=0 ++exp.runner.total_steps=1000 \
+data_prep=pretrain_multi_subj_multi_chan_template \
++data_prep.electrodes=/storage/czw/btbench/electrode_selections/clean_laplacian.json \
++data.raw_brain_data_dir=/storage/czw/braintreebank_data/ 
++exp.runner.results_dir_root=/storage/czw/btbench/outputs/btbench_popt_lite \
++data.eval_name="frame_brightness" ++data.subject=1 ++data.brain_run=1 ++data.split_type="SS_DM" 
```
Important arguments:
m.upstream_path` --- the path to the pretrained PopulationTransformer. Make sure to change the path to somewhere in your directory, i.e., not in `czw`
- `data.btbench_cache_path` --- the path to the cached BrainBERT embeddings for the BTBench tasks. This should match the output path from the first command.
- `data_prep.electrodes` --- this should be included in the repo. Just change the `czw` to your `<username>`
- `data_prep.brain_data_dir` --- this is the path to the `braintreebank_data`. Just change the `czw` to your `<username>`
- `data_prep.results_dir_root` --- output directory
- `model.frozen_upstream` --- whether to freeze the upstream model or not. If you change this you should also adjust `results_dir_root`.

But usually, I'm running this script, which goes through all subject-trial pairs:
```
run_scripts/run_popt_ss_sm.sh 0 11 SS_SM
```
- first argument is lower bound on subject-trial pair (12 total)
- second argument is upper bound
- third argument is split type (SS_DM or SS_SM)

### Instructions for Andrii
1. Copy over `baffin:/storage/czw/btbench/saved_examples/btbench_popt_embeds_lite/` to openmind. This means that you can skip the "Write the BTBench tasks" section above. You can use scp or rsync to do this.
2. To run "Write the BTBench tasks" first copy the pretrained PopT weights from `victoria:/storage/czw/victoria/MultiBrainBERT/outputs/randomized_replacement_no_gaussian_blur.pth`
3. Pay attention to the `results_dir_root` argument
4. For `braintreebank_data`, I put all my `h5` files under `all_subject_data` (see my baffin setup). You may have to do that too.

## Citation

If you use BT-bench in your work, please cite the following paper:
TBD

#!/bin/bash

declare -a subjects=(1 1 1 2 2 2 2 2 2 2 3 3 3 4 4 4 5 6 6 6 7 7 8 9 10 10)
declare -a trials=(0 1 2 0 1 2 3 4 5 6 0 1 2 0 1 2 0 0 1 4 0 1 0 0 0 1)
declare -a eval_names=(
    "frame_brightness"
    "global_flow"
    "local_flow"
    "global_flow_angle"
    "local_flow_angle" 
    "face_num"
    "volume"
    "pitch"
    "delta_volume"
    "delta_pitch"
    "speech"
    "onset"
    "gpt2_surprisal"
    "word_length"
    "word_gap"
    "word_index"
    "word_head_pos"
    "word_part_speech"
    "speaker"
)
LOWER_BOUND=$1
UPPER_BOUND=$2

#for EVAL_IDX in {0..18}
for EVAL_IDX in {0..0}
do
for ((PAIR_IDX=LOWER_BOUND; PAIR_IDX<=UPPER_BOUND; PAIR_IDX++))
do
# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}

echo "Running eval $PAIR_IDX for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
WEIGHTS=randomized_replacement_no_gaussian_blur; python3 run_cross_val.py +exp=multi_elec_feature_extract ++exp.runner.save_checkpoints=False ++model.frozen_upstream=False +task=btbench_popt +criterion=pt_multiclass_criterion +data=btbench_decode +preprocessor=empty_preprocessor +model=pt_downstream_multiclass ++model.upstream_path=/storage/czw/PopTCameraReadyPrep/outputs/${WEIGHTS}.pth ++model.upstream_cfg.use_token_cls_head=True ++model.upstream_cfg.name=pt_model_custom ++data.btbench_cache_path=/storage/czw/btbench/saved_examples/btbench_popt_embeds ++data.k_fold=5 ++exp.runner.num_workers=32 ++exp.runner.total_steps=1000 +data_prep=pretrain_multi_subj_multi_chan_template ++data_prep.electrodes=/storage/czw/btbench/electrode_selections/clean_laplacian.json ++data_prep.brain_runs=/storage/czw/btbench/trial_selections/test_trials.json ++data.raw_brain_data_dir=/storage/czw/braintreebank_data/ ++exp.runner.results_dir_root=/storage/czw/btbench/outputs/btbench_popt ++data.eval_name=${EVAL_NAME} ++data.subject=${SUBJECT} ++data.brain_run=${TRIAL}
done
done


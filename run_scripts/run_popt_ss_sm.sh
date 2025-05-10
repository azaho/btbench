#!/bin/bash

declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

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
SPLIT_TYPE=$3

for ((PAIR_IDX=LOWER_BOUND; PAIR_IDX<=UPPER_BOUND; PAIR_IDX++))
do
for EVAL_IDX in {0..18}
do
# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}

echo "Running eval $PAIR_IDX for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
WEIGHTS=randomized_replacement_no_gaussian_blur; python3 run_cross_val.py +exp=multi_elec_feature_extract ++exp.runner.save_checkpoints=False ++model.frozen_upstream=False +task=btbench_popt +criterion=pt_feature_extract_coords_criterion +data=btbench_decode +preprocessor=empty_preprocessor +model=pt_downstream_multiclass ++model.upstream_path=/storage/czw/PopTCameraReadyPrep/outputs/${WEIGHTS}.pth ++model.upstream_cfg.use_token_cls_head=True ++model.upstream_cfg.name=pt_model_custom ++data.btbench_cache_path=/storage/czw/btbench/saved_examples/btbench_popt_embeds_lite ++data.k_fold=5 ++exp.runner.num_workers=0 ++exp.runner.total_steps=1000 +data_prep=pretrain_multi_subj_multi_chan_template ++data_prep.electrodes=/storage/czw/btbench/electrode_selections/clean_laplacian.json ++data.raw_brain_data_dir=/storage/czw/braintreebank_data/ ++exp.runner.results_dir_root=/storage/czw/btbench/outputs/btbench_popt_lite ++data.eval_name=${EVAL_NAME} ++data.subject=${SUBJECT} ++data.brain_run=${TRIAL} ++data.split_type=${SPLIT_TYPE}
done
done



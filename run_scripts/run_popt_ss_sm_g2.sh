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

gpu_id=2

REPO_DIR="/home/geeling/Projects/ieeg_project/btbench"
BRAINTREEBANK_DIR="/home/geeling/Projects/ieeg_project/data/braintreebank_data"
for ((PAIR_IDX=LOWER_BOUND; PAIR_IDX<=UPPER_BOUND; PAIR_IDX++))
do
for EVAL_IDX in {0..18}
do
# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}

echo "Running eval $PAIR_IDX for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
WEIGHTS=randomized_replacement_no_gaussian_blur; 

python3 run_cross_val.py \
+exp=multi_elec_brant \
++exp.runner.device="cuda:${gpu_id}" \
++exp.runner.multi_gpu=False \
++exp.runner.num_workers=8 \
++exp.runner.train_batch_size=16 \
++exp.runner.valid_batch_size=16 \
++exp.runner.total_steps=1000 \
++exp.runner.log_step=100 \
++exp.runner.checkpoint_step=100 \
++exp.runner.scheduler.gamma=0.99 \
++exp.runner.save_checkpoints=False \
++exp.runner.results_dir_root="${REPO_DIR}/outputs/all_btbench_brant_lite" \
+data=btbench_decode_brant \
++data.btbench_cache_path="${REPO_DIR}/saved_examples/all_btbench_brant_wavs" \
++data.raw_brain_data_dir="${BRAINTREEBANK_DIR}" \
++data.eval_name=${EVAL_NAME} \
++data.subject=${SUBJECT} \
++data.brain_run=${TRIAL} \
++data.split_type=${SPLIT_TYPE} \
++data.k_fold=5 \
+model=brant_model \
++model.num_electrodes="placeholder" \
++model.aggregation_mode="linear_concat" \
++model.device="cuda:${gpu_id}" \
++model.gpu_id=${gpu_id} \
+task=btbench_brant \
++task.task_name=${task_name} \
+criterion=brant_feature_extract_criterion \
++criterion.loss_fn="bce" \
++criterion.use_power=True \
+preprocessor=brant_preprocessor \
+data_prep=pretrain_multi_subj_multi_chan_template \
++data_prep.electrodes="${REPO_DIR}/electrode_selections/clean_laplacian.json" \
++data_prep.brain_runs="${REPO_DIR}/trial_selections/lite_trials.json" \
++hydra.run.dir="${REPO_DIR}/outputs/all_btbench_brant_lite/logs/\${now:%H-%M-%S}_brant_${SUBJECT}_${TRIAL}_${EVAL_NAME}" 



done
done



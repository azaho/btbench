#!/bin/bash
#SBATCH --job-name=e_popt_frozen          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --exclude=dgx001,dgx002,node057
#SBATCH --array=1-456  # 285 if doing mini btbench
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p use-everything

source ../.venv/bin/activate

nvidia-smi

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

declare -a split_types=(
    "SS_SM"
    "SS_DM"
)

# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} ))

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
SPLIT_TYPE=${split_types[$SPLITS_TYPE_IDX]}

echo "Running eval $PAIR_IDX for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
WEIGHTS=randomized_replacement_no_gaussian_blur; python run_cross_val.py \
    +exp=multi_elec_feature_extract ++exp.runner.save_checkpoints=False ++model.frozen_upstream=True +task=btbench_popt \
    +criterion=pt_feature_extract_coords_criterion +data=btbench_decode +preprocessor=empty_preprocessor +model=pt_downstream_multiclass \
    ++model.upstream_path=/om2/user/zaho/btbench_popt/PopTCameraReadyPrep/outputs/${WEIGHTS}.pth \
    ++model.upstream_cfg.use_token_cls_head=True ++model.upstream_cfg.name=pt_model_custom ++data.btbench_cache_path=/om2/user/zaho/btbench_popt/btbench/saved_examples/btbench_popt_embeds_lite \
    ++data.k_fold=5 ++exp.runner.num_workers=0 ++exp.runner.total_steps=1000 +data_prep=pretrain_multi_subj_multi_chan_template \
    ++data_prep.electrodes=/om2/user/zaho/btbench_popt/btbench/electrode_selections/clean_laplacian.json ++data.raw_brain_data_dir=/om2/user/zaho/braintreebank_laplacian_rereferenced_line_noise_removed/ \
    ++exp.runner.results_dir_root=/om2/user/zaho/btbench_popt/btbench/outputs/btbench_popt_lite ++data.eval_name=${EVAL_NAME} ++data.subject=${SUBJECT} ++data.brain_run=${TRIAL} ++data.split_type=${SPLIT_TYPE}
done
done



#!/bin/bash
#SBATCH --job-name=e_se_lite          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=1    # Request 8 CPU cores per GPU
#SBATCH --mem=4G
#SBATCH --exclude=dgx001,dgx002
#SBATCH -t 6:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-570  # 285 if doing mini btbench
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p normal

export PYTHONUNBUFFERED=1
source .venv/bin/activate
# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 2 3 4 7 10)
declare -a trials=(2 4 1 1 0 0)

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
declare -a splits_type=(
    "SS_SM"
    #"SS_DM"
)
declare -a n_samples_per_bin=(
    1
    2
    4
    8
    16
)
# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#splits_type[@]} ))   
N_SAMPLES_PER_BIN_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#splits_type[@]} % ${#n_samples_per_bin[@]} ))

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
N_SAMPLES_PER_BIN=${n_samples_per_bin[$N_SAMPLES_PER_BIN_IDX]}

SAVE_DIR="eval_results_lite_${SPLITS_TYPE}_test"

echo "Using python: $(which python)"
echo "Using python version: $(python --version)"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, splits_type $SPLITS_TYPE"
echo "Command: python -u eval_single_electrode_test.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --splits_type $SPLITS_TYPE --verbose --save_dir $SAVE_DIR --lite --n_samples_per_bin $N_SAMPLES_PER_BIN --skip_electrode_chance 0.9"

# Add the -u flag to Python to force unbuffered output
.venv/bin/python -u eval_single_electrode_test.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --splits_type $SPLITS_TYPE --verbose --save_dir $SAVE_DIR --lite --n_samples_per_bin $N_SAMPLES_PER_BIN --skip_electrode_chance 0.9
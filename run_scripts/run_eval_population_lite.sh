#!/bin/bash
#SBATCH --job-name=eval_population          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --mem=64G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-1824  # 285 if doing mini btbench
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate
# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
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
declare -a preprocess=(
    "fft_abs"
    "fft_absangle"
    "fft_realimag"
    "none" 
)
declare -a splits_type=(
    "SS_SM"
    "SS_DM"
)


# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
PREPROCESS_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} % ${#splits_type[@]} ))   

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
PREPROCESS=${preprocess[$PREPROCESS_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}

SAVE_DIR="eval_results_lite_${SPLITS_TYPE}"

echo "Using python: $(which python)"
echo "Using python version: $(python --version)"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS, splits_type $SPLITS_TYPE"
echo "Command: python -u eval_population.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --preprocess $PREPROCESS --splits_type $SPLITS_TYPE --verbose --save_dir $SAVE_DIR --lite --only_1second"

# Add the -u flag to Python to force unbuffered output
.venv/bin/python -u eval_population.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --preprocess $PREPROCESS --splits_type $SPLITS_TYPE --verbose --save_dir $SAVE_DIR --lite --only_1second
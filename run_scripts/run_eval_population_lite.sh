#!/bin/bash
#SBATCH --job-name=e_p_lite          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --mem=128G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-1977  # 285 if doing mini btbench
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate

declare -a subjects=(1 1 1 2 2 2 2 2 2 2 3 3 3 4 4 4 5 6 6 6 7 7 8 9 10 10)
declare -a trials=(0 1 2 0 1 2 3 4 5 6 0 1 2 0 1 2 0 0 1 4 0 1 0 0 0 1)
# mini btbench
#declare -a subjects=(1 2 3 7 10)
#declare -a trials=(2 6 0 0 0)

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


# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
PREPROCESS_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} ))

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
PREPROCESS=${preprocess[$PREPROCESS_IDX]}

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS"
# Add the -u flag to Python to force unbuffered output
python -u eval_population.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --preprocess $PREPROCESS --verbose --lite
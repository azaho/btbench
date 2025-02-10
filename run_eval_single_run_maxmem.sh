#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=12    # Request 8 CPU cores per GPU
#SBATCH --mem=128G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-992      # 26 subject-trial pairs * 13 eval names = 338 total jobs
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Create arrays of subject IDs and trial IDs that correspond to array task ID
# for all subject_trials in the dataset
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

# Calculate indices for this task
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / 2 / 19 ))
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / 2 % 19 ))
SPECTROGRAM=$(( ($SLURM_ARRAY_TASK_ID-1) % 2 ))

# Get subject, trial and eval name for this task
SUBJECT_ID=${subjects[$PAIR_IDX]}
TRIAL_ID=${trials[$PAIR_IDX]} 
EVAL_NAME=${eval_names[$EVAL_IDX]}

echo "Running eval for subject $SUBJECT_ID, trial $TRIAL_ID, eval $EVAL_NAME"
# Add the -u flag to Python to force unbuffered output
python -u eval_single_run.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_name $EVAL_NAME --folds 5 --spectrogram $SPECTROGRAM
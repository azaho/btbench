#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=384G
#SBATCH -t 4:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=6,21,22,38,39,51,91,92,118,125,126,141,142,146,164,167,168,206,207,208,212,215,216,217,224,225,226,238,239,240,250,251,252,257,258,259,266,267,268,277,278,279,283,284,285,288,293,294,295,305,306,307,316,317,318,328,329,335,336,338,339,363,364,388,391,395,396,398,478,772,792,794,796,810,832,984      # Only run failed jobs
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p use-everything

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
declare -a spectrogram_string=(
    "--spectrogram 1"
    ""
)

# Calculate indices for this task
SPECTROGRAM_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % 2 ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / 2 / 19 ))
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / 2 % 19 ))

# Get subject, trial and eval name for this task
SUBJECT_ID=${subjects[$PAIR_IDX]}
TRIAL_ID=${trials[$PAIR_IDX]} 
EVAL_NAME=${eval_names[$EVAL_IDX]}
SPECTROGRAM_STRING=${spectrogram_string[$SPECTROGRAM_IDX]}

echo "Running eval for subject $SUBJECT_ID, trial $TRIAL_ID, eval $EVAL_NAME"
# Add the -u flag to Python to force unbuffered output
python -u eval_single_run_ss_sm.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_name $EVAL_NAME --folds 5 $SPECTROGRAM_STRING --normalize 1

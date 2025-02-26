#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=96G
#SBATCH -t 4:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-16      # 26 subject-trial pairs * 13 eval names = 338 total jobs
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p yanglab

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Create arrays of subject IDs and trial IDs that correspond to array task ID
# for all subject_trials in the dataset
declare -a subjects=(3)
declare -a trials=(2)
declare -a eval_names=(
    "pitch"
    "volume"
    "volume_v2_raw"
    "pitch_v2_raw" 
    "volume_v2_enhanced"
    "pitch_v2_enhanced"
    "volume__reproduced"
    "pitch__reproduced"
)
declare -a spectrogram_string=(
    "--spectrogram 1"
    ""
)

# Calculate indices for this task
SPECTROGRAM_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % 2 ))
PAIR_IDX=0
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / 2 ))

# Get subject, trial and eval name for this task
SUBJECT_ID=${subjects[$PAIR_IDX]}
TRIAL_ID=${trials[$PAIR_IDX]} 
EVAL_NAME=${eval_names[$EVAL_IDX]}
SPECTROGRAM_STRING=${spectrogram_string[$SPECTROGRAM_IDX]}

echo "Running eval for subject $SUBJECT_ID, trial $TRIAL_ID, eval $EVAL_NAME"
# Add the -u flag to Python to force unbuffered output
python -u eval_single_run_ss_sm.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_name $EVAL_NAME --folds 5 $SPECTROGRAM_STRING # add this for using spectrogram
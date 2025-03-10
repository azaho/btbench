#!/bin/bash
#SBATCH --job-name=eval_linear_per_bin          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=64G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-494      # 26 subject-trial pairs * 13 eval names = 338 total jobs
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate

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
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / 19 ))
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % 19 ))

# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
# Add the -u flag to Python to force unbuffered output
python -u eval_linear_per_bin.py --fold 1 --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --normalize 1
python -u eval_linear_per_bin.py --fold 2 --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --normalize 1
python -u eval_linear_per_bin.py --fold 3 --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --normalize 1
python -u eval_linear_per_bin.py --fold 4 --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --normalize 1
python -u eval_linear_per_bin.py --fold 5 --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --normalize 1
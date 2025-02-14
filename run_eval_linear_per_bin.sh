#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=64G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=0-24      # 26 subject-trial pairs * 13 eval names = 338 total jobs
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p yanglab
export PYTHONUNBUFFERED=1
source .venv/bin/activate

declare -a folds=(1 2 3 4 5)
declare -a eval_names=(
    "word_part_speech"
    "word_index"
    "word_length"
    "volume"
    "word_head_pos"
    "gpt2_surprisal"
    "onset"
)

# Calculate indices for this task
FOLD_IDX=$(( ($SLURM_ARRAY_TASK_ID) / 5 ))
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID) % 5 ))

# Get subject, trial and eval name for this task
FOLD=${folds[$FOLD_IDX]}
EVAL_NAME=${eval_names[$EVAL_IDX]}

echo "Running eval for fold $FOLD, eval $EVAL_NAME"
# Add the -u flag to Python to force unbuffered output
python -u eval_linear_per_bin.py --fold $FOLD --eval_name $EVAL_NAME
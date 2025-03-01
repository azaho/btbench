import os
import numpy as np

# Create arrays of subject IDs and trial IDs that correspond to array task ID
subjects = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6, 7, 7, 8, 9, 10, 10]
trials = [0, 1, 2, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 0, 1, 2, 0, 0, 1, 4, 0, 1, 0, 0, 0, 1]
eval_names = [
    "frame_brightness", "global_flow", "local_flow", "global_flow_angle",
    "local_flow_angle", "face_num", "volume", "pitch", "delta_volume",
    "delta_pitch", "speech", "onset", "gpt2_surprisal", "word_length",
    "word_gap", "word_index", "word_head_pos", "word_part_speech", "speaker"
]
spectrogram_string = ["_spectrogram_normalized", "_voltage_normalized"]

failed_jobs = []
output_dir = "eval_results_ss_sm/"

# Loop through all possible job IDs (1-988)
for job_id in range(1, 989):
    # Calculate indices for this task
    spectrogram_idx = ((job_id-1) % 2)
    pair_idx = ((job_id-1) // 2 // 19)
    eval_idx = ((job_id-1) // 2 % 19)
    
    # Get parameters for this job
    subject_id = subjects[pair_idx]
    trial_id = trials[pair_idx]
    eval_name = eval_names[eval_idx]
    suffix = spectrogram_string[spectrogram_idx]
    
    # Check if result file exists
    results_file = os.path.join(output_dir, f'linear{suffix}_subject{subject_id}_trial{trial_id}_{eval_name}.json')
    
    if not os.path.exists(results_file):
        failed_jobs.append(job_id)

print(f"Failed jobs: {failed_jobs}")
print(f"Number of failed jobs: {len(failed_jobs)}")

# Create a new SLURM script for failed jobs with more memory
slurm_script = """#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=384G
#SBATCH -t 4:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array={failed_jobs_str}      # Only run failed jobs
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
SUBJECT_ID=${{subjects[$PAIR_IDX]}}
TRIAL_ID=${{trials[$PAIR_IDX]}} 
EVAL_NAME=${{eval_names[$EVAL_IDX]}}
SPECTROGRAM_STRING=${{spectrogram_string[$SPECTROGRAM_IDX]}}

echo "Running eval for subject $SUBJECT_ID, trial $TRIAL_ID, eval $EVAL_NAME"
# Add the -u flag to Python to force unbuffered output
python -u eval_single_run_ss_sm.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_name $EVAL_NAME --folds 5 $SPECTROGRAM_STRING --normalize 1
""".format(failed_jobs_str=','.join(map(str, failed_jobs)))

# Write the SLURM script to a file
with open('run_eval_single_run_ss_sm_maxmem.sh', 'w') as f:
    f.write(slurm_script)
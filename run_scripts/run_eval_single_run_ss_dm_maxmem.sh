#!/bin/bash
#SBATCH --job-name=eval_single_run          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=6    # Request 8 CPU cores per GPU
#SBATCH --mem=384G
#SBATCH -t 4:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=126,128,136,142,146,148,164,174,202,210,212,214,215,218,220,224,225,227,228,230,240,244,248,250,261,262,264,268,271,277,278,280,288,289,292,293,296,298,300,315,316,318,320,321,324,325,326,338,340,344,345,346,348,351,354,355,357,358,363,364,369,370,374,376,378,391,394,400,401,402,408,412,414,415,420,422,424,426,430,436,440,444,451,452,472,484,496,498,499,506,508,509,522,530,546,547,550,551,554,558,560,564,566,570,574,576,578,583,586,587,595,604,606,626,630,635,644,740,755,771,778,794,803,816,820      # Only run failed jobs
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Create arrays of subject IDs and trial IDs that correspond to array task ID
# for all subject_trials in the dataset
declare -a subjects=(1 1 1 2 2 2 2 2 2 2 3 3 3 4 4 4 6 6 6 7 7 10 10)
declare -a trials=(0 1 2 0 1 2 3 4 5 6 0 1 2 0 1 2 0 1 4 0 1 0 1)
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
python -u eval_single_run_ss_dm.py --subject $SUBJECT_ID --trial $TRIAL_ID --eval_name $EVAL_NAME $SPECTROGRAM_STRING --normalize 1

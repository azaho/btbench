#!/bin/bash
#SBATCH --job-name=e_p_lite          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=2    # Request 8 CPU cores per GPU
#SBATCH --mem=96G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --exclude=dgx001,dgx002
#SBATCH --array=1-1200  # 285 if doing mini btbench
#SBATCH --output logs/%A_%a.out # STDOUT
#SBATCH --error logs/%A_%a.err # STDERR
#SBATCH -p use-everything

export PYTHONUNBUFFERED=1
source .venv/bin/activate

# Use the BTBENCH_LITE_SUBJECT_TRIALS from btbench_config.py
declare -a subjects=(1 1 2 2 3 3 4 4 7 7 10 10)
declare -a trials=(1 2 0 4 0 1 0 1 0 1 0 1)

declare -a eval_names=(
    #"frame_brightness"
    "global_flow"
    #"local_flow"
    #"global_flow_angle"
    #"local_flow_angle" 
    #"face_num"
    "volume"
    #"pitch"
    #"delta_volume"
    #"delta_pitch"
    "speech"
    #"onset"
    "gpt2_surprisal"
    #"word_length"
    #"word_gap"
    #"word_index"
    "word_head_pos"
    #"word_part_speech"
    #"speaker"
)
declare -a preprocess=(
    'none' # no preprocessing, just raw voltage
    #'fft_absangle', # magnitude and phase after FFT
    #'fft_realimag' # real and imaginary parts after FFT
    #'fft_abs' # just magnitude after FFT ("spectrogram")

    #'remove_line_noise' # remove line noise from the raw voltage
    #'downsample_200' # downsample to 200 Hz
    #'downsample_200-remove_line_noise' # downsample to 200 Hz and remove line noise
)

declare -a splits_type=(
    #"SS_SM"
    "SS_DM"
)

declare -a n_data=(
    8
    32
    128
    512
)
declare -a seed=(
    1001
    1002
    1003
    1004
    1005
)

# Calculate indices for this task
EVAL_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) % ${#eval_names[@]} ))
PAIR_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} % ${#subjects[@]} ))
PREPROCESS_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} % ${#preprocess[@]} ))
SPLITS_TYPE_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} % ${#splits_type[@]} ))
N_DATA_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} / ${#splits_type[@]} % ${#n_data[@]} ))
SEED_IDX=$(( ($SLURM_ARRAY_TASK_ID-1) / ${#eval_names[@]} / ${#subjects[@]} / ${#preprocess[@]} / ${#splits_type[@]} / ${#n_data[@]} % ${#seed[@]} ))



# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}
PREPROCESS=${preprocess[$PREPROCESS_IDX]}
SPLITS_TYPE=${splits_type[$SPLITS_TYPE_IDX]}
N_DATA=${n_data[$N_DATA_IDX]}
SEED=${seed[$SEED_IDX]}

save_dir="eval_results_lite_${SPLITS_TYPE}_subsets"

echo "Running eval for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL, preprocess $PREPROCESS --save_dir $save_dir"
# Add the -u flag to Python to force unbuffered output
python -u eval_population_subsets.py --eval_name $EVAL_NAME --subject $SUBJECT --trial $TRIAL --preprocess $PREPROCESS --verbose --save_dir $save_dir --lite --splits_type $SPLITS_TYPE --n_data $N_DATA --seed $SEED --only_1second
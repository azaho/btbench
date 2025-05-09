#!/bin/bash

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
LOWER_BOUND=$1
UPPER_BOUND=$2
MAX_PARALLEL=40

CPU_ID=0

# Array to track PIDs and their assigned cores
declare -A pid_to_core
declare -a available_cores

# Initialize available core list
for i in $(seq 0 $((MAX_PARALLEL - 1))); do
  available_cores+=($i)
done

# Function to launch a job
launch_job() {
  local job_id=$1
  local core=$2
  local EVAL_NAME=$3
  local SUBJECT=$4
  local TRIAL=$5

  echo "Running eval $PAIR_IDX for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL on CPU $CPU_ID"
  (taskset -c "$core" python eval_single_electrode.py --subject $SUBJECT --trial $TRIAL --verbose --eval_name $EVAL_NAME --preprocess remove_line_noise) &
  local pid=$!
  pid_to_core[$pid]=$core
  echo "Started job $job_id on CPU core $core (PID $pid)"
}

job_count=0
for EVAL_IDX in {0..18}
do
for ((PAIR_IDX=LOWER_BOUND; PAIR_IDX<=UPPER_BOUND; PAIR_IDX++))
do
# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}

  while [[ ${#available_cores[@]} -eq 0 ]]; do
    wait -n
    for pid in "${!pid_to_core[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        core=${pid_to_core[$pid]}
        available_cores+=($core)
        unset pid_to_core[$pid]
        echo "Job with PID $pid finished. Core $core is now available."
      fi
    done
    sleep 1
  done

  # Get an available core and launch the job
  core=${available_cores[0]}
  available_cores=("${available_cores[@]:1}")
  launch_job $job_count $core $EVAL_NAME $SUBJECT $TRIAL
  ((job_count++))

done
done
echo "All jobs completed."



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
NUM_JOBS=10

for EVAL_IDX in {0..18}
do
for ((PAIR_IDX=LOWER_BOUND; PAIR_IDX<=UPPER_BOUND; PAIR_IDX++))
do
# Get subject, trial and eval name for this task
EVAL_NAME=${eval_names[$EVAL_IDX]}
SUBJECT=${subjects[$PAIR_IDX]}
TRIAL=${trials[$PAIR_IDX]}

echo "Running eval $PAIR_IDX for eval $EVAL_NAME, subject $SUBJECT, trial $TRIAL"
(python single_electrode.py --subject $SUBJECT --trial $TRIAL --verbose --eval_name $EVAL_NAME) &

# Limit the number of parallel jobs
if (( $(jobs -r -p | wc -l) >= NUM_JOBS )); then
wait -n # Wait for any job to complete
fi

done
done
wait

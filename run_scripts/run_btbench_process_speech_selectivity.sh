#!/bin/bash
#SBATCH --job-name=btbench_speech_selectivity          # Name of the job
#SBATCH --ntasks=1             # 8 tasks total
#SBATCH --cpus-per-task=4    # Request 8 CPU cores per GPU
#SBATCH --mem=128G
#SBATCH -t 12:00:00         # total run time limit (HH:MM:SS) (increased to 24 hours)
#SBATCH --array=1-10      # 14 jobs (108/8 rounded up)
#SBATCH --output r/%A_%a.out # STDOUT
#SBATCH --error r/%A_%a.err # STDERR

source .venv/bin/activate
python btbench_process_speech_selectivity.py --subject $SLURM_ARRAY_TASK_ID
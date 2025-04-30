#!/bin/bash

# wrapper to run all models for all subjects and sessions which are defined in the sessions file

# Exit immediately if any command fails
set -e

# Read the JSON file using Python and extract the list
sessions=$(python - <<END
import os
import json

with open('sessions.json', 'r') as file:
    data = json.load(file)

version="allsessions_prediction"
method="merge"

included_sessions = []
for session in data[version]: 
    included_sessions.append(session)

if method=="merge":
    included_sessions = [i[:7] for i in included_sessions]
    included_sessions = sorted(set(included_sessions))

print('\n'.join(included_sessions))

END
)

# Parse the JSON array in Bash
readarray -t sessions_arr <<< "$sessions"

for session in "${sessions_arr[@]}"; do
    echo $session;
done

# Define the models and contexts to run
netz_arr=("EEGNetv4") # "Deep4Net" ,  "ShallowFBCSPNet"
#context_arr=("inter") #  "inter"
context_arr=("inter" "intra_human" "intra_monkey") #  "inter"

# run the multiverse and universes for all (subjects and) sessions in list
for session in "${sessions_arr[@]}"; do

    # Code to run if the file exists
    echo "Sending $session prediction to SLURM."
    for net in "${netz_arr[@]}"; do
        for context in "${context_arr[@]}"; do
            sbatch src/run_prediction.slurm $session braindecode_single_merge $net $context
        done
    done

done

# R1: adults extra
sbatch src/run_prediction.slurm sub-901 braindecode_single_merge EEGNetv4 inter
sbatch src/run_prediction.slurm sub-901 braindecode_single_merge EEGNetv4 intra_human
sbatch src/run_prediction.slurm sub-901 braindecode_single_merge EEGNetv4 intra_monkey
sbatch src/run_prediction.slurm sub-902 braindecode_single_merge EEGNetv4 inter
sbatch src/run_prediction.slurm sub-902 braindecode_single_merge EEGNetv4 intra_human
sbatch src/run_prediction.slurm sub-902 braindecode_single_merge EEGNetv4 intra_monkey

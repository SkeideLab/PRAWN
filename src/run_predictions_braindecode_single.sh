#!/bin/bash

# wrapper to run all models for all subjects and sessions which are defined in the sessions file

# Exit immediately if any command fails
set -e

# Read the JSON file using Python and extract the list
sessions=$(python - <<END
import os
import json
from collections import Counter

with open('sessions.json', 'r') as file:
    data = json.load(file)

version="allsessions_prediction"

included_sessions = []
for session in data[version]: 
    included_sessions.append(session)

# R1
# Extract subject IDs
subject_ids = [s.split('_')[0] for s in included_sessions]
# Count how many times each subject appears
counts = Counter(subject_ids)
# Filter the list to keep only entries where the subject appears exactly twice
filtered_sessions = [s for s in included_sessions if counts[s.split('_')[0]] == 2]


print('\n'.join(filtered_sessions))

END
)

# Parse the JSON array in Bash
readarray -t sessions_arr <<< "$sessions"

for session in "${sessions_arr[@]}"; do
    echo $session;
done

# Define the models and contexts to run
netz_arr=("EEGNetv4") # "Deep4Net", "ShallowFBCSPNet"
context_arr=("inter" "intra_human" "intra_monkey") #  "inter"

# DEBUG
#sessions_arr=("sub-002_ses-002" "sub-003_ses-002" "sub-003_ses-001")
#sbatch src/run_prediction.slurm sub-002_ses-001 braindecode_single EEGNetv4 inter

# run the multiverse and universes for all (subjects and) sessions in list
for session in "${sessions_arr[@]}"; do

    # Code to run if the file exists
    echo "Sending $session prediction to SLURM."
    for net in "${netz_arr[@]}"; do
        for context in "${context_arr[@]}"; do
            sbatch src/run_prediction.slurm $session braindecode_single $net $context
        done
    done

done


#!/bin/bash


# wrapper to run all lgbms for all subjects and sessions which are defined in the 
# "downloadable_sub...json" file

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

# run the multiverse and universes for all (subjects and) sessions in list
for session in "${sessions_arr[@]}"; do
    echo "Sending $session prediction to SLURM."
    sbatch src/run_prediction.slurm $session timeresolved_single_merge dummy dummy
done


# R1: adults extra
sbatch src/run_prediction.slurm sub-901 timeresolved_single_merge dummy dummy
sbatch src/run_prediction.slurm sub-902 timeresolved_single_merge dummy dummy
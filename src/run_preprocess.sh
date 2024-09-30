#!/bin/bash


# wrapper to run all multiverses for all subjects and sessions which are defined in the 
# "downloadable_sub...json" file

# Exit immediately if any command fails
set -e

# Read the JSON file using Python and extract the list
sessions=$(python - <<END
import json

with open('sessions.json', 'r') as file:
    data = json.load(file)
    sessions = data['allsessions']   
    print('\n'.join(sessions))
END
)


# Parse the JSON array in Bash
readarray -t sessions <<< "$sessions"

# Print the list
#echo "$sessions"

# run the multiverse for all (subjects and) sessions in list
for session in "${sessions[@]}"; do
    echo "Sending $session preprocessing to SLURM."
    sbatch src/run_preprocess.slurm $session 
done


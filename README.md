# PRAWNp

## Setup

The conda environment is saved in folder env.
The R/targets environment is found in targets.


## How to run

On a Cluster with SLURM job scheduling system

Preprocessing (one job per participant)

```bash
bash src/run_preprocess.sh
```

Time-resolved decoding (one job per participant), incl. interpretation

```bash
bash src/run_predictions_timeresolved_merge.sh
```

EEGNet decoding (one job per participant and contrast)

```bash
bash src/run_predictions_braindecode_merge.sh
```

Postprocessing of 
- Time-resolved decoding
- Time-resolved interpretation
- EEGNet decoding


```bash
python src/postprocess_accuracies.py
```



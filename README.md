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

## Notebooks with one-off analyses

*demographics.ipynb* will just output the demographic information for the manuscript
*dnn_sitmulus_representation.ipynb* is quick-and-dirty analysis of stimulus embeddings
*interpret_timeresolved.ipynb* does a group analysis for the model interpretation of the time resolved decoding 

## targets (R)

The targets 




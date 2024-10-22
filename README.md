# PRAWN

Paper: *Electrophysiological decoding captures the temporal trajectory of face categorization in infants*

**Roman Kessler** & **Michael A. Skeide**


When using, please cite:

- Kessler, R., & Skeide, M. A. (2024). Electrophysiological decoding captures the temporal trajectory of face categorization in infants. bioRxiv. https://doi.org/10.1101/2024.10.07.617144
  
- Kessler, R., & Skeide, M. A. (2024). EEG Dataset of 38 Infants (Aged 5-11 Months) Viewing Human and Monkey Faces (Two Identities Each) (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13881206


The raw data can be downloaded at https://doi.org/10.5281/zenodo.13881207.


## Setup

The conda environment is saved in folder [env](/env). All python/bash/slurm scripts are found in [src](/src).

The system architecture and hardware details of the HPC used for all *Python* and *Bash* scripts  with *SLURM* job scheduling system can be found in [MPCDF RAVEN user guide](https://docs.mpcdf.mpg.de/doc/computing/raven-details.html).

The *R* environment, which is used in a *targets* pipeline and all related processing scripts are found in [targets](/targets).

The system architecture and hardware details of the Mac used to process the *targets* pipeline in *R* can be found [here](https://support.apple.com/en-us/111893). A 16 GB RAM version was used.




## Preprocessing, and machine learning

For all scripts, adjust the paths depending on your environment.
Run the following on a HPC cluster with SLURM job scheduling system.

Preprocessing (starts one job per participant):

```bash
bash src/run_preprocess.sh
```

Time-resolved decoding (starts one job per participant), incl. interpretation:

```bash
bash src/run_predictions_timeresolved_merge.sh
```

EEGNet decoding (starts one job per participant and contrast) - up to 1 day per participant and node. Make sure to set appropriate time limit in *run_prediction.slurm*.

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

## Jupyter notebooks with adhoc analyses

- *demographics.ipynb* will output the demographic information for the manuscript (see Methods)
- *dnn_stimulus_representation.ipynb* is an analysis of stimulus embeddings (see Methods)
- *interpret_timeresolved.ipynb* performs a group analysis for the model interpretation of the time-resolved decoding (Fig. 2) 

## targets (R) - statistics and vizualization

The targets pipeline includes all remaining analyses and outputs all plots. Other numbers are written in the respective processing nodes. The project can be opened in RStudio. The *_targets* file needs to be sourced. Then, *tar_make()* produces all results or re-runs all outdated nodes of the network.


# License

Shield: [![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International License][cc-by-nc].

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

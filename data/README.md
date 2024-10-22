
# Data structure

- Raw contains the raw data from the EEG system. Needs to be downloaded from the Zenodo repository.
- Interim contains the minimally cleaned raw EEG data: It is easiest to work with these files, as they are the result of concatenation of EEG snippets. Every session per participant is presented here with 1 "raw.fif" file.
-   Triggers were recoded
-   Technical problems during EEG recording sometimes lead to several recording files per session. These were unified here.
- Processed contains the preprocessed epochs to be used for ML

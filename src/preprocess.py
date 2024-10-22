""" HEADER START """

import itertools
import mne
from scipy.signal import detrend
import sys, os
from tqdm import tqdm
from glob import glob
import re
import numpy as np
import itertools
import pandas as pd
if sys.platform.startswith('darwin'):
    mne.viz.set_browser_backend('qt', verbose=None) # 'qt' or 'matplotlib'

# go to base directory and import globals
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(base_dir)
sys.path.append(base_dir)
from src.utils import *
from src.config import *

# DEBUG
#sub_ses_str = "sub-001_ses-001"

# define subject and session by arguments to this script
if len(sys.argv) != 2:
    print("Usage: python script.py sub_ses_str")
    sys.exit(1)
else:
    sub_ses_str = sys.argv[1]

sub = sub_ses_str[4:7]
ses = sub_ses_str[-3:]
print(f'Processing Subject {sub} Session {ses}!')

# create directories --> was in config, but sub_ses_str is needed
create_sub_ses_dirs(sub_ses_str, dirs)

# if adult subject, then no rating is done
if int(sub) >= 900:
    subject_group = 'adults'
elif int(sub) >= 100:
    subject_group = 'rsvp'    
    if int(sub) >= 300:
        subject_group = 'rsvp_categories'
else:
    subject_group = 'infants'

""" HEADER END """


""" PART I: pre-preprocessing """

# download subject & session from datashare
download_datashare_dir(datashare_dir = 'PRAWN/raw/eeg/' + sub_ses_str, 
                   target_dir = dirs['raw_dir'] + sub_ses_str)  # eeg data (+ cut file if available)
if subject_group == 'infants':
    for i in range(10): # assume max 10 different raters were involved
        try:
            download_datashare_file(datashare_file = f"PRAWN/raw/videos/ratings/{sub_ses_str}_coded_{i}.xlsx", 
                    target_dir = dirs['ratings_dir'] + sub_ses_str)  # ratings
        except:
            pass
    download_datashare_file(datashare_file = 'PRAWN/raw/psychopy/' + sub_ses_str + '/info_conditions.txt',
                    target_dir = dirs['raw_dir'] + sub_ses_str)  # psychopy conditions file (needed for potential pattern matching if triggers != codings)

# save data characteristics
manager = CharacteristicsManager(f"{dirs['interim_dir']}{sub_ses_str}/characteristics.json", force_new=True)

# load raw data
raw = load_raw_data(sub_ses_str=sub_ses_str,
                    update_eeg_headers_callback=update_eeg_headers) 

# new: check on "rating" files
# - save to file for next step, and replace file in next steps
if subject_group == 'infants':
    rater_ids, soerensen_dice, n_ratings = calculate_ratings(sub_ses_str, plot=True)
    manager.update_subfield('ratings', 'rater_ids', rater_ids)
    manager.update_subfield('ratings', 'soerensen_dice', soerensen_dice)
    manager.update_subfield('ratings', 'n_ratings', n_ratings)

# compare ratings files, triggers, and theoretical available conditions, and harmonize them
if subject_group == 'infants':
    load_raw_trigger_rating_matching(sub_ses_str, plot=True)

# delete trials, in which participant has not focused the screen
if subject_group == 'infants':
    try:
        raw, n1, n2 = delete_non_fixation_trials(raw.copy(), 
                            ratings_file = dirs['ratings_dir'] + sub_ses_str + '/' + sub_ses_str + '_analyzed_merged.xlsx',
                            inclusion=rating_version, 
                            return_characteristics=True,     
        ) 
    except AnnotationsMismatchError as e:
        print(f'Error: {e}')
        sys.exit(1)
    manager.update_subfield('n_trials', 'before video coding', n1)
    manager.update_subfield('n_trials', 'after video coding', n2)


# create custom montage
raw.rename_channels({'VP': 'Fp2',  # this is a detour
                     'VM': 'Fp1'})
raw.set_montage(make_31_montage(raw, sub_ses_str, plot=False, save=False))

# change annotations and extract events
raw, events, event_id, event_counts = process_annotations_and_events(raw.copy(), sub_ses_str=sub_ses_str, version="humansmonkeys")

print(event_counts)
manager.update_characteristic('event_counts', event_counts)

# downsample to 250 Hz
raw, events = raw.resample(250, events = events)
raw.events = events
np.save(dirs['interim_dir'] + sub_ses_str + '/events.npy', events)
np.save(dirs['interim_dir'] + sub_ses_str + '/event_id.npy', event_id, allow_pickle=True)

# create artificial eye channel (VM - VP) for potential later eye blink correction
raw = calculate_artificial_channels(raw.copy(), pairs=[['Fp1', 'Fp2'],['F9', 'F10']], labels=['eyeV', 'eyeH'])

# the bad channel Fp1 might not be used anymore? so delete it!
# during robust regression (or PREP pipeline in general), bad channels are confusing everything
raw.drop_channels(['Fp1'])

# save raw data
raw.save(dirs['interim_dir'] + sub_ses_str + '/raw.fif', overwrite=True)

# decide, if this subject / session has enough trials to proceed
check_inclusion(event_counts, min_trials, sub_ses_str)

""" PART II: preprocessing """ 

# load pre-pre-processed raw, and event information
raw = mne.io.read_raw_fif(dirs['interim_dir'] + sub_ses_str + '/raw.fif', preload=True)
events = np.load(dirs['interim_dir'] + sub_ses_str + '/events.npy')
event_id = np.load(dirs['interim_dir'] + sub_ses_str + '/event_id.npy', allow_pickle=True).item()

cv = 5 # number of chunks for cv splits
manager.update_characteristic('cv-folds', cv)

mne.set_log_level('ERROR')  # only show warning messages

# Filter
_raw0 = raw.copy().filter(l_freq=0.5, h_freq=15, method='fir', fir_design='firwin', skip_by_annotation='EDGE boundary', n_jobs=-1)

# Eye artifact correction
_raw1, n1 = ica_eog_emg(_raw0.copy(), sub_ses_str, method='eog', save_ica=False, save_plot=False, save_str=None)
manager.update_characteristic('ICA EOG n components', n1)

# robust average reference
try: 
    _raw2, n1 = robust_average_PREP(_raw1.copy(), delete_bad_info=True)
    manager.update_characteristic('robust average', n1)
except: # if too many bad channels for RANSAC, then try without RANSAC; if too many bad channels for RobRef, then go with default reference
    _raw2 = _raw1.copy().set_eeg_reference('average', projection=False) 
    manager.update_characteristic('robust average', "failed, average reference instead")

# keep only eeg channels
_raw2.pick_types(eeg=True)

# epoching, baseline correction, detrending
epochs = mne.Epochs(_raw2.copy(), 
                    events, 
                    tmin=-0.4, tmax=1.0,
                    baseline=(-0.2, 0.),
                    detrend=1, # linear
                    proj=False,
                    reject_by_annotation=False, 
                    preload=True)


epochs.metadata = pd.DataFrame({"trial_id": list(range(len(epochs)))})

# autoreject
# set default values for AR: interpolate all bad channels, and drop none
n_interpolate=len(epochs.info['ch_names']) # or False for default hyperparameter finding
consensus=len(epochs.info['ch_names'])  # or False for default hyperparameter finding

# estimate autoreject model on all epochs (not only training epochs)
epochs_ar, n1 = autorej(epochs.copy(), 
                    log_path=None,  
                    plot_path=None, 
                    ar_model_path=None,  
                    mode='estimate', 
                    n_interpolate=n_interpolate,
                    consensus=consensus,
                    show_plot=False,
                    save_plot=False, 
                    save_model=False, 
                    save_log=False, 
                    save_drop_dict=False,
                    drop_dict_path=None)

interp_frac_channels, interp_frac_trials, total_interp_frac = summarize_artifact_interpolation(n1)

# train_test_split moved here here
epochs_chunks, rest_chunk = train_test_split_epochs_kfcv(epochs_ar.copy(), cv=cv)

# save epochs
[epochs_chunks[i].save(f"{dirs['processed_dir']}{sub_ses_str}/250Hz_chunk{i}-epo.fif", overwrite=True) for i in range(len(epochs_chunks))]
if rest_chunk:
    rest_chunk.save(f"{dirs['processed_dir']}{sub_ses_str}/250Hz_chunkrest-epo.fif", overwrite=True)



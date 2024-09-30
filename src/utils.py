import os.path
import os
import fnmatch  # for file deletion
import sys
import numpy as np
import pandas as pd
import mne
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from math import isclose
import random

import getpass
from pathlib import Path
import keyring
import owncloud  # pip install pyocclient
import json
from autoreject import AutoReject, get_rejection_threshold, read_auto_reject
from mne.preprocessing import EOGRegression
from mne.preprocessing import ICA, corrmap, create_ecg_epochs, create_eog_epochs
import pyprep # this must be installed via pip, as the conda install crashes the whole environment!
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap

import sys
from glob import glob 
import h5py
import re

from src.config import *

import time

class Timer:
    def __init__(self):
        self.start_time = None
        self.total_paused_time = 0
        self.pause_start_time = None

    def start_or_resume(self):
        if self.start_time is None:
            self.start_time = time.time()
        else:
            self.total_paused_time += time.time() - self.pause_start_time
            self.pause_start_time = None

    def stop(self):
        if self.start_time is None:
            raise ValueError("Timer is not running.")
        elapsed_time = time.time() - self.start_time - self.total_paused_time
        self.start_time = None
        return elapsed_time

    def pause(self):
        if self.start_time is None:
            raise ValueError("Timer is not running.")
        self.pause_start_time = time.time()

class TimerManager:
    def __init__(self):
        self.timers = {}

    def create_timer(self, name):
        if name in self.timers:
            raise ValueError(f"Timer with name '{name}' already exists.")
        self.timers[name] = Timer()

    def start_or_resume_timer(self, name):
        timer = self.timers.get(name)
        if timer is None:
            raise ValueError(f"Timer with name '{name}' does not exist.")
        timer.start_or_resume()

    def stop_timer(self, name):
        timer = self.timers.get(name)
        if timer is None:
            raise ValueError(f"Timer with name '{name}' does not exist.")
        elapsed_time = timer.stop()
        print(f"Timer '{name}' took {elapsed_time:.4f} seconds.")
        return elapsed_time

    def pause_timer(self, name):
        timer = self.timers.get(name)
        if timer is None:
            raise ValueError(f"Timer with name '{name}' does not exist.")
        timer.pause()

    def resume_timer(self, name):
        timer = self.timers.get(name)
        if timer is None:
            raise ValueError(f"Timer with name '{name}' does not exist.")
        timer.start_or_resume()

    def stop_all_timers(self):
        for name in self.timers:
            elapsed_time = self.stop_timer(name)  # Stop and print the time for each timer
            #print(f"Timer '{name}' took {elapsed_time:.4f} seconds.")
            # already printed above


# this to save some information about e.g. dropouts in some pipelines, n interpolated, and so on
class CharacteristicsManager:
    def __init__(self, file_path, force_new=False):
        self.file_path = file_path
        self.characteristics = {}
        if not force_new:
            self._load_characteristics()
        self.save_characteristics()  # Save the file upon initialization

    def _load_characteristics(self):
        if os.path.isfile(self.file_path):
            with open(self.file_path, 'r') as file:
                self.characteristics = json.load(file)

    def save_characteristics(self):
        with open(self.file_path, 'w') as file:
            json.dump(self.characteristics, file, indent=4)

    def update_characteristic(self, key, value):
        # if only 1 level is added
        self.characteristics[key] = value
        self.save_characteristics()

    def get_characteristic(self, key):
        return self.characteristics.get(key, None)

    def update_subfield(self, key, subfield, subfield_value):
        # if more than 1 level is added
        if key not in self.characteristics:
            self.characteristics[key] = {}
        self.characteristics[key][subfield] = subfield_value
        self.save_characteristics()

    def get_subfield(self, key, subfield):
        return self.characteristics.get(key, {}).get(subfield, None)

    def update_subsubfield(self, key, subfield, subsubfield, subsubfield_value):
        # if more than 2 level is added
        if key not in self.characteristics:
            self.characteristics[key] = {}
        if subfield not in self.characteristics[key]:
            self.characteristics[key][subfield] = {}
        self.characteristics[key][subfield][subsubfield] = subsubfield_value
        self.save_characteristics()

    def get_subsubfield(self, key, subfield, subsubfield):
        return self.characteristics.get(key, {}).get(subfield, {}).get(subsubfield, None)


def delete_files_in_folder(input_folder, filename_pattern):
    try:
        for root, _, files in os.walk(input_folder):
            for filename in files:
                if fnmatch.fnmatch(filename, filename_pattern):
                    file_path = os.path.join(root, filename)
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
        print("Deletion completed.")
    except Exception as e:
        print(f"An error occurred: {e}")

def download_datashare_dir(datashare_dir, target_dir):  # target_dir was bids_ds
    """Downloads all files within a directory from MPCDF DataShare.
    adapted from: https://github.com/SkeideLab/bids_template_code/blob/b7751c5ed7419d4401b07112d7dd553a9b99ffca/helpers.py#L48
    Please change your datashare credentials in the code below.
    
    Args:
        datashare_dir (str): Path of the raw data starting from the DataShare root, like this:
            https://datashare.mpcdf.mpg.de/apps/files/?dir=<datashare_dir>. It
            is e.g. PRAWN/raw/<modelity>/sub-<sub>_ses-<ses> for the PRAWN study.
            All files in this folder will be downloaded.
        target_dir (str): Local directory where the files will be downloaded to. 
            The directory will be created if it does not exist.
    """
    # establish connection
    datashare = datashare_establish_connection(datashare_user='kroma')

    # Loop over session folders on DataShare
    files = datashare.list(datashare_dir)
    for file in files:
        # Explicity exclude certain file names
        if file.name.startswith('_'):
            continue

        # Download if it doesn't exist
        local_file = Path(f"{target_dir}/{file.name}")
        if not local_file.exists():  # and not exclude_file.exists():
            
            # Download zip file
            print(f'Downloading {file.path} to {local_file}')
            create_if_not_exist(target_dir)
            datashare.get_file(file, local_file)        
        else:
            print(f"File {local_file} already exists. Skipping download.")



def download_datashare_file(datashare_file, target_dir):  # target_dir was bids_ds
    """Downloads all files within a directory from MPCDF DataShare.
    adapted from: https://github.com/SkeideLab/bids_template_code/blob/b7751c5ed7419d4401b07112d7dd553a9b99ffca/helpers.py#L48
    Please change your datashare credentials in the code below.
    
    Args:
        datashare_file (str): Path of the file starting from the DataShare root.
        target_dir (str): Local directory where the files will be downloaded to. The directory
            will be created if it does not exist. The filename will be retrieved from 
            the datashare_file path.

    """
    # establish connection
    datashare = datashare_establish_connection(datashare_user='kroma')

    # Loop over session folders on DataShare
    #files = datashare.list(datashare_file)
    files = datashare.list(str(Path(datashare_file).parent))
    for i in files:
        if i.name == Path(datashare_file).name:
            file = i
            break
    
    print(file)
    if not file:
        print('no file !!!')

    # Download if it doesn't exist
    local_file = Path(f"{target_dir}/{file.name}")
    if not local_file.exists():  # and not exclude_file.exists():
        
        # Download file
        print(f'Downloading {file} to {local_file}')
        create_if_not_exist(target_dir)
        datashare.get_file(file, local_file)
    else:
        print(f"File {local_file} already exists. Skipping download.")



def datashare_establish_connection(datashare_user):
    """Establishes connection to MPCDF DataShare.
    adapted from: https://github.com/SkeideLab/bids_template_code/blob/b7751c5ed7419d4401b07112d7dd553a9b99ffca/helpers.py#L48
    
    Args:
        datashare_user : str
            Username for MPCDF Datashare. If None, the username will be
            retrieved from the system (only works in MPCDF HPC).

    Returns:
        owncloud.Client
    """

    # Get DataShare login credentials
    if not datashare_user:
        datashare_user = getpass.getuser()  # get username from system (only works in MPCDF HPC)

    datashare_pass = keyring.get_password('datashare', datashare_user)
    if datashare_pass is None:
        datashare_pass = getpass.getpass()
        keyring.set_password('datashare', datashare_user, datashare_pass)        
    
    # Login to DataShare
    domain = 'https://datashare.mpcdf.mpg.de'
    datashare = owncloud.Client(domain)
    datashare.login(datashare_user, datashare_pass)

    return datashare


def update_eeg_headers(file):
    """
    Uses a .vhdr file and updates the references to .eeg and .vmrk files based on the
    current filename of the .vhdr file.
    This must be done if a raw file was renamed, as BrainVision writes the original filename in the headers.
    
    Args:
        file (str): Path to the .vhdr file
    """
    # Read the .vhdr file
    with open(file, 'r') as f:
        lines = f.readlines()

    # Update the lines
    updated_lines = []
    for line in lines:
        if line.startswith("DataFile="):
            # Replace characters after "DataFile=" with the filename and ".eeg"
            line = "DataFile=" + os.path.basename(file).replace(os.path.splitext(file)[1], ".eeg") + "\n"
        elif line.startswith("MarkerFile="):
            # Replace characters after "MarkerFile=" with the filename and ".vmrk"
            line = "MarkerFile=" + os.path.basename(file).replace(os.path.splitext(file)[1], ".vmrk") + "\n"
        updated_lines.append(line)

    # Write the updated content back to the file
    with open(file, 'w') as f:
        f.writelines(updated_lines)

    # Open and update the .vmrk file
    vmrk_file = os.path.splitext(file)[0] + ".vmrk"
    if os.path.exists(vmrk_file):
        with open(vmrk_file, 'r') as f:
            vmrk_lines = f.readlines()

        # Update the lines
        updated_vmrk_lines = []
        for line in vmrk_lines:
            if line.startswith("DataFile="):
                # Replace content after "DataFile=" with the filename of "file" but with extension ".eeg"
                line = "DataFile=" + os.path.basename(file).replace(os.path.splitext(file)[1], ".eeg") + "\n"
            updated_vmrk_lines.append(line)

        # Write the updated content back to the .vmrk file
        with open(vmrk_file, 'w') as f:
            f.writelines(updated_vmrk_lines)

def load_raw_data(sub_ses_str, update_eeg_headers_callback=None):
    """
    Load and first clean-up of raw EEG data.

    Args:
        path (str): Path to the directory containing raw EEG data files.
        update_eeg_headers_callback (callable): A callback function to update EEG headers.

    Returns:
        mne.io.Raw: Processed raw EEG data.
    """
    path = dirs['raw_dir'] + sub_ses_str + '/'
    raw_files = sorted(glob(path + "*.vhdr"))

    if len(raw_files) == 0:
        raise ValueError(f"No raw files found in {path}.")
    elif len(raw_files) == 1:
        print(f"Found {len(raw_files)} raw files in {path}.")
    else:
        print(f"Found {len(raw_files)} raw files in {path}. Concatenate them.")

    if update_eeg_headers_callback:
        #update_eeg_headers_callback(raw_files[0])
        # DEBUG
        update_eeg_headers(raw_files[0])
        
    raw = mne.io.read_raw_brainvision(raw_files[0], misc='auto', scale=1.0, preload=True, verbose=False)

    if len(raw_files) > 1:
        for i in range(1, len(raw_files)):
            if update_eeg_headers_callback:
                #update_eeg_headers_callback(raw_files[i])
                # DEBUG        
                update_eeg_headers(raw_files[i])

            raw_temp = mne.io.read_raw_brainvision(raw_files[i], misc='auto', scale=1.0, preload=True, verbose=False)
            raw.append(raw_temp) # this calls concatenate_raws

    # delete the EDGE annotations which are marked as "bad"
    raw.annotations.delete([i for (i, j) in enumerate(raw.annotations.description)
                            if (j not in ["Stimulus/S  1", "Stimulus/S  2",
                                          "Stimulus/S  3", "Stimulus/S  4",
                                          "Stimulus/S  5", "Stimulus/S  6",
                                          "Stimulus/S  7", "Stimulus/S  8"])
                            ])
    
    # info file for trigger exclusions
    try:
        info = json.load(open(f"{dirs['raw_dir']}{sub_ses_str}/info.json")) # TODO: files like this for each eeg file, if several present in folder
        print("info with raw data found: ")
        print(info)
    except:
        print("no info file found with raw data.")
    # if trigers shall be excluded (cause experiment was startet twice for instance)
    # should be after non-stimulus triggers are already deleted!!
    if 'info' in locals():
        raw.annotations.delete(np.arange(info['exclude_triggers_begin']))
        raw.annotations.delete(np.arange(len(raw.annotations) - info['exclude_triggers_end'],
                                            len(raw.annotations)))

    return raw

# sørensen-dice coefficient
def dice(a, b):
    return 2 * np.sum(a * b) / (np.sum(a) + np.sum(b))

# fixation ratings of two raters
def calculate_ratings(sub_ses_str, plot=False):
    """takes two rating files, and compares and unifies their ratings.
    Exactly 2 rating files need to be downloaded to the session folder.
    Will generate a new file with the unified ratings (conservative and liberal), 
    and return the rater ids and the soerensen dice coefficient.

    Args:
        sub_ses_str (str)
        plot (bool): if plot should be saved to dirs['plot_dir']

    Returns:
        list of ints: rater ids
        float: soerensen dice coefficient
    """
    rating_files = sorted(glob(dirs['ratings_dir'] + sub_ses_str + '/*_coded_*.xlsx'))

    # there must be exactly 2 rating files
    if len(rating_files) != 2:
        print(f'ERROR: {len(rating_files)} rating files found, but 2 expected!')
        sys.exit(1)

    # open xlsx
    dfr1 = pd.read_excel(rating_files[0], usecols=[0,1]) # dtype={'trial': int, 'fixation': int}
    dfr2 = pd.read_excel(rating_files[1], usecols=[0,1])
    dfr1.columns = ['trial', 'fixation']
    dfr2.columns = ['trial', 'fixation']

    # get the "index"/name/number of the rater
    rater_ids = [int(re.findall(r"coded_(\d+).", i)[0]) for i in rating_files]

    # then out of the combined file, merge the triggers of the EEG session onto it
    dfr = dfr1.copy()
    dfr.columns = ['trial', 'fixation_rater_A']
    dfr['fixation_rater_B'] = dfr2['fixation']
    dfr['fixation_liberal'] = dfr[['fixation_rater_A', 'fixation_rater_B']].max(axis=1)
    dfr['fixation_conservative'] = dfr[['fixation_rater_A', 'fixation_rater_B']].min(axis=1)

    # dice coefficient
    soerensen_dice = dice(dfr['fixation_rater_A'], dfr['fixation_rater_B'])
    
    # save the new file
    dfr.to_excel(dirs['ratings_dir'] + sub_ses_str + '/' + sub_ses_str + '_analyzed.xlsx', index=False)
    
    # plot and save plot
    if plot:
        # make a heatmap of the ratings 
        dfr.drop(columns=['trial'], inplace=True) # drop the condition column
        colors = ["red", "lightgreen"]
        custom_cmap = ListedColormap(colors)
        # Create an image from the DataFrame
        plt.figure(figsize=(4, 16))
        f = sns.heatmap(dfr.values, cmap=custom_cmap, cbar=False) #sns.color_palette("flare", as_cmap=True)
        f.set_xticklabels(dfr.columns, rotation=90)
        
        # custom legend for binary values
        legend_handles = [Patch(color=colors[1], label='fixation'),  # red
                  Patch(color=colors[0], label='non-fixation')]  # green
        plt.legend(handles=legend_handles, ncol=2, bbox_to_anchor=[0.5, 1.02], loc='lower center', fontsize=12, handlelength=.8)

        # Show and save the plot
        plt.title("participant fixation by rater & agreement of raters")
        plt.savefig(dirs['plot_dir'] + sub_ses_str + '/qa/fixation_raters.png', dpi=300)
        plt.show()
    
    return rater_ids, soerensen_dice, len(dfr)


# handle the info json file, that might indicate if triggers need to be dismissed



# pattern matching on the combined ratings file
def load_raw_trigger_rating_matching(sub_ses_str, plot=False):
    
    # load ratings file(s)
    ratings_file = dirs['ratings_dir'] + sub_ses_str + '/' + sub_ses_str + '_analyzed.xlsx'
    merged_ratings_file = dirs['ratings_dir'] + sub_ses_str + '/' + sub_ses_str + '_analyzed_merged.xlsx'
    # don't use, because the combined file will be used
    #rater_index = [int(re.findall(r"coded_(\d+).", i)[0]) for i in rating_files]
    #rater_index = [i[-5] for i in ratings_files]    
    
    ratings = pd.read_excel(ratings_file)
    # Test: important: double check if the trial number is a perfect sequence, or if the rating file had an error already
    is_perfect_sequence = (ratings['trial'].diff().fillna(1) == 1).all()
    if not is_perfect_sequence:
        print(f"ERROR: The trial numbers are not a perfect sequence in the original ratings file. Double check this before continuing.")
        sys.exit(1)
    
    # now only 1 combined file
    # ratings_single = []
    # for i, ratings_file in enumerate(ratings_files):
    #     ratings_single.append(pd.read_excel(ratings_file, usecols=[0,1])) #, engine="odf")
    #     ratings_single[-1].columns = ["trial", f"fixation_rater_{rater_index[i]}"]
    #     # Handle missing values in ratings (e.g. if some trials were not rated because experiment ended)
    #     mask = ratings_single[-1].isnull().any(axis=1)
    #     if mask.any():
    #         first_missing_index = mask.idxmax()
    #         ratings_single[-1] = ratings_single[-1].iloc[:first_missing_index]
    # ratings = ratings_single[0]
    # if len(ratings_single) > 1:
    #     for i in ratings_single[1:]:
    #         ratings.merge(i, on="trial", how="outer")
    
    # load psychopy condition info
    all_conditions_file = dirs['raw_dir'] + sub_ses_str + '/info_conditions.txt'
    conditions = []
    with open(all_conditions_file, 'r') as file:
        # Read the entire line, split it by commas, and convert the parts to integers
        line = file.readline()
        conditions = [int(num) for num in line.split(',')[:-1]]

    # merge all conditions (cropped to the length of ratings) onto ratings
    # --> the ratings file always starts with trial 1, the condition file as well. Only be careful that the triggers might not start with trial 1 later!
    # leave a buffer of 20 conditions at the end, so in case the EEG ran longer, but it was not coded on the video
    if sub_ses_str == "sub-001_ses-001":
        max_conditions = 640
    else:
        max_conditions = 480 # maximal amount of possible conditions
    
    if len(ratings) < (max_conditions - 20):
        n_buffer = 20
    else:
        n_buffer = max_conditions - len(conditions)

    # Create a new DataFrame with the extended length
    extended_ratings = pd.concat([ratings, pd.DataFrame(np.nan, 
                                                        index=np.arange(len(ratings), len(ratings) + n_buffer), 
                                                        columns=ratings.columns)], 
                                 ignore_index=True)
    
    # Fill the "trial" column with a continuous sequence
    extended_ratings['trial'] = np.arange(1, len(extended_ratings) + 1) # no error is allowed in the 

    conditions = conditions[:len(ratings)+n_buffer]
    
    try:
        print(conditions)
        print(extended_ratings)
        extended_ratings['condition'] = conditions
    except:
        print("ERROR: The length of the conditions does not match the length of the ratings file. Double check this before continuing.")
        sys.exit(1)

    # info file for trigger exclusions
    try:
        info = json.load(open(f"{dirs['raw_dir']}{sub_ses_str}/info.json")) # TODO: files like this for each eeg file, if several present in folder
        print("info with raw data found: ")
        print(info)
    except:
        print("no info file found with raw data.")
    
    # load all raw data files
    raw_files = sorted(glob(dirs['raw_dir'] + sub_ses_str + '/*.vhdr'))    
    n_annotations = []
    for j, eeg_file in enumerate(raw_files):
        update_eeg_headers(eeg_file)
        raw = mne.io.read_raw_brainvision(eeg_file, misc='auto', scale=1.0, preload=True, verbose=False)
                
        # get the trigger codes from the raw data
        non_trial_triggers = [i for i,j in enumerate(raw.annotations.description) if "Stimulus" not in j]
        raw.annotations.delete(non_trial_triggers)
        
        #raw.annotations = [j for i,j in zip(raw.annotations.description, raw.annotations) if "Stimulus" in i] # delete brainvision annotations
        #raw.annotations.description = [i for i in raw.annotations.description if "Stimulus" in i] # delete brainvision annotations
        
        # if trigers shall be excluded (cause experiment was startet twice for instance)
        if 'info' in locals():
            raw.annotations.delete(np.arange(info['exclude_triggers_begin']))
            raw.annotations.delete(np.arange(len(raw.annotations) - info['exclude_triggers_end'],
                                             len(raw.annotations)))

        condition_map = {"Stimulus/S  1": 1,
                        "Stimulus/S  2": 2,
                        "Stimulus/S  3": 3,
                        "Stimulus/S  4": 4,
                        "Stimulus/S  5": 5,
                        "Stimulus/S  6": 6,
                        "Stimulus/S  7": 7,
                        "Stimulus/S  8": 8}

        raw.annotations.description = np.array(
            [
                condition_map[i] if i in condition_map else i
                for i in raw.annotations.description
            ]
        )
        annotations = list(raw.annotations.description)
        
        
        
        n_annotations.append(len(annotations))
        #raw.annotations.duration = np.array([0.75 for _ in raw.annotations.description])

        # find the matching sequences
        extended_ratings[f'eeg_file_{j+1}'] = False
        ratings_values = extended_ratings['condition'].tolist()
        found_something = False
        for i in range(len(ratings_values) - len(annotations) + 1):
            if ratings_values[i:i + len(annotations)] == annotations:
                extended_ratings[f'eeg_file_{j+1}'][i:i + len(annotations)] = True
                found_something = True
        if not found_something:
            for dismiss_one in range(1,6): # if the eeg recording went up to 5 trials longer than the ratings, try to dismiss the last 1-5 trials
                annotations = annotations[:-dismiss_one]
                for i in range(len(ratings_values) - len(annotations) + 1):
                    if ratings_values[i:i + len(annotations)] == annotations:
                        extended_ratings[f'eeg_file_{j+1}'][i:i + len(annotations)] = True
                        found_something = True
                        print(f"Found a match by dismissing {dismiss_one} trials at the end in EEG file number {j+1}")
                        break
                break
    
    # check: if one of the EEG files has trials, which are not coded, then code them with 0 (subject has not fixated)
    eeg_columns = [i for i in extended_ratings.columns if "eeg_file" in i]
    # columns with only false should be excluded
    eeg_columns_with_trues = [i for i in eeg_columns if extended_ratings[i].any()] 
    # Find the index of the last True value across all eeg columns
    last_true_index = extended_ratings[eeg_columns_with_trues].apply(lambda col: col[col == True].index.tolist()[-1]).max()
    # delete the remaining buffer zone
    extended_ratings = extended_ratings[:last_true_index+1]
    # delete all the trials where none EEG file is True (trial not catched by eeg system) 
    indices_with_true = extended_ratings[extended_ratings[eeg_columns_with_trues].any(axis=1)].index
    extended_ratings = extended_ratings.loc[indices_with_true]
    # pseudo-code the NaN rows with 0s
    fixation_columns = [i for i in extended_ratings.columns if "fixation" in i]
    extended_ratings[fixation_columns] = extended_ratings[fixation_columns].fillna(0)
    
    # save the processed ratings file
    extended_ratings.to_excel(merged_ratings_file, index=False)
    
    if plot:
        # make a heatmap of the merges (not the ratings)    
        extended_ratings = extended_ratings.astype(int) # convert to int for plotting
        fixation_columns = [i for i in extended_ratings.columns if "fixation" in i or "condition" in i or "trial" in i]
        extended_ratings.drop(columns=fixation_columns, inplace=True) # drop the condition column
        
        # Create an image from the DataFrame
        plt.figure(figsize=(8, 16))
        f = sns.heatmap(extended_ratings.values, cmap=sns.color_palette("flare", as_cmap=True), cbar=False)
        f.set_xticklabels(extended_ratings.columns, rotation=90)

        # Show and save the plot
        plt.title("Participant fixation and EEG data merging")
        plt.savefig(dirs['plot_dir'] + sub_ses_str + '/qa/eeg_file_merging.png', dpi=300)
        plt.show()
    
 

class AnnotationsMismatchError(Exception):
    pass

def delete_non_fixation_trials(raw, ratings_file, inclusion='conservative', return_characteristics=False):
    """
    Delete trials in which participants have not focused on the screen.

    Args:
        raw (mne.io.Raw): Raw EEG data.
        ratings_file (str): Ratings file containing trial fixation information.
        inclusion (str): Liberal (all trials for which at least 1 rater confirmed fixation)
            or conservative (2 raters need to agree on fixation). Defaults to conservative.
        return_characteristics (bool): Whether to return the number of trials removed
    Returns:
        mne.io.Raw: Raw EEG data with non-fixation trials removed.
    """
    # load rating file
    ratings = pd.read_excel(ratings_file) #, usecols=[0,1], engine="odf")
    #ratings.columns = ["trial", "fixation"]

    # Handle missing values in ratings (?)
    # new: due to the merging of the files (load_raw_trigger_rating_matching), this should not be necessary anymore
    #mask = ratings.isnull().any(axis=1)
    #if mask.any():
    #    first_missing_index = mask.idxmax()
    #    ratings = ratings.iloc[:first_missing_index]

    # Check annotation lengths
    if len(raw.annotations) != len(ratings):
        raise AnnotationsMismatchError("Annotations differ between ratings and EEG data. Check before continuing.")
    
    # Delete annotations based on non-fixation
    #idx_to_remove = np.array(ratings[ratings[f'fixation_{inclusion}'] == 0].trial) - 1  # -1 because annotations start at 0
    # new: this should be the index of the file now, as the indices are now harmonized with the triggers
    idx_to_remove = np.array(ratings[ratings[f'fixation_{inclusion}'] == 0].index)
    raw.annotations.delete(idx_to_remove)
    
    if return_characteristics:
        return raw, len(ratings), len(raw.annotations)
    else:
        return raw


# montage adapted by GHOST study
def make_31_montage(raw, sub_ses_str, plot=True, save=True):
    """
    Defines the GHOST montage for PRAWN
    adapted from https://stackoverflow.com/questions/58783695/how-can-i-plot-a-montage-in-python-mne-using-a-specified-set-of-eeg-channels
    
    Args:
        raw (mne.io.Raw): Raw EEG data.
        plot (bool): Whether to plot the montage.
        save (bool): Whether to save the montage.
    
    Returns: montage
    """
    # Form the 10-20 montage
    mont1020 = mne.channels.make_standard_montage('standard_1020')
    # Choose what channels you want to keep
    # Make sure that these channels exist e.g. T1 does not exist in the standard 10-20 EEG system!
    kept_channels = raw.ch_names
    # additionally recode VP to Fp2
    add_channels = ['Fp2']

    ind = [i for (i, channel) in enumerate(mont1020.ch_names) if (channel in kept_channels) or (channel in add_channels)]
    mont = mont1020.copy()
    
    # Keep only the desired channels
    mont.ch_names = [mont1020.ch_names[x] for x in ind]
    kept_channel_info = [mont1020.dig[x + 3] for x in ind]
    
    # Keep the first three rows as they are the fiducial points information
    mont.dig = mont1020.dig[:3] + kept_channel_info
    if plot:
        mont.plot()
    if save:
        mont.plot()
        plt.savefig(dirs['plot_dir'] + "/" + sub_ses_str + "/montage/montage30.png", dpi=300)
    return mont



def process_annotations_and_events(raw, sub_ses_str, version="humansmonkeys"):
    """
    Process annotations, events, and save data.

    Args:
        raw (mne.io.Raw): Raw EEG data.
        sub_ses_str (str): Subject and session string.
        version (str): Version of the experiment. Defaults to "humansmonkeys". Else "categories"
    Returns:
        np.ndarray, dict: Processed events and event IDs.
    """
    
    # rename annotations
    if version == "humansmonkeys":
        condition_map = {"Stimulus/S  1": 'human1',
                        "Stimulus/S  2": 'human1',
                        "Stimulus/S  3": 'human2',
                        "Stimulus/S  4": 'human2',
                        "Stimulus/S  5": 'monkey1',
                        "Stimulus/S  6": 'monkey1',
                        "Stimulus/S  7": 'monkey2',
                        "Stimulus/S  8": 'monkey2'}
    elif version == "categories":
        condition_map = {"Stimulus/S  1": 'face',
                        "Stimulus/S  2": 'hand',
                        "Stimulus/S  3": 'ball',
                        "Stimulus/S  4": 'bike'}
    else:
        print("ERROR: version not recognized.")
        sys.exit(1)

    raw.annotations.description = np.array(
        [
            condition_map[i] if i in condition_map else i
            for i in raw.annotations.description
        ]
    )
    raw.annotations.duration = np.array([0.75 for _ in raw.annotations.description])

    # delete remaining stimulus events
    indices_to_remove = [i for i, j in enumerate(raw.annotations.description) if "Stimulus" in j]
    raw.annotations.delete(indices_to_remove)
    
    # get events and event_id
    events, event_id = mne.events_from_annotations(raw)

    fig = mne.viz.plot_events(events, event_id=event_id, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, show=False)
    fig.savefig(dirs['plot_dir'] + sub_ses_str + "/qa/events.png")

    np.save(dirs['interim_dir'] + sub_ses_str + "/events.npy", events)
    np.save(dirs['interim_dir'] + sub_ses_str + "/event_id.npy", event_id)

    # print the number of events per condition
    event_counts = {}
    for key in event_id.keys():
        event_counts[key] = len(events[events[:, 2] == event_id[key]])
        print(key, len(events[events[:, 2] == event_id[key]]))
    # save event_counts
    with open(dirs['interim_dir'] + sub_ses_str + "/event_counts.json", "w") as f:
        json.dump(event_counts, f)

    return raw, events, event_id, event_counts


def get_train_test_sizes(event_counts, min_trials, perc_train=0.8):
    """
    Get train and test sizes based on event counts.

    Args:
        event_counts (dict): Dictionary of event counts.
        train_size (float): Train size.

    Returns:
        int: Train size
    """
    if min(event_counts.values()) >= min_trials:
        train_size = int(np.round(perc_train * min(event_counts.values())))

    return train_size



def calculate_artificial_channels(raw, pairs=[['Fp1', 'Fp2'], ['F9', 'F10']], labels=['eyeV', 'eyeH']):
    """
    Calculate artificial eye movement electrodes, by subtracting two electrodes
    timeseries from each other.

    Substraction of fitting electrodes leads to boosting of the 
    (anticorrelated) eye movement signals,
    while canceling out brain-related signals or other signals.
    The electrodes need to be on opposite sites of the eye(s).

    For instance: VM and VP (above and below the right eye). Vertical movement.
    T9 and T10 (left and right from the eyes). Horizontal movement.

    Args:
        raw (mne.raw) raw data object
        pairs (nested list): list of list of electrode names which are subtracted from each other
        labels (list): list of labels (names) of the new artificial electrodes
    
    Returns:
        raw (mne.raw): processed raw data object
    """
    
    # Create a copy of the original raw data
    raw_new = raw.copy()

    for i in range(len(pairs)):
        
        # Specify the names of the existing channels you want to subtract
        channel1 = pairs[i][0] #'Fp1'
        channel2 = pairs[i][1] #'Fp2'  # was VP and VM, but got recorded as Fp1 and Fp2
        
        # Subtract the values of channel2 from channel1 and create a new channel
        new_channel_data = raw[channel1][0] - raw[channel2][0]
        new_channel_name = labels[i] #'new_channel'

        # Reshape the new channel data to have shape (1, n_samples)
        new_channel_data = np.reshape(new_channel_data, (1, -1))

        # Create a new info object for the new channel
        new_info = mne.create_info([new_channel_name], raw_new.info['sfreq'], ch_types='eog') # WAS EEG

        # Create a new RawArray object for the new channel
        new_channel_raw = mne.io.RawArray(new_channel_data, new_info)

        # Add the new channel to the raw data
        raw_new.add_channels([new_channel_raw], force_update_info=True)

    return raw_new


def check_inclusion(event_counts, min_trials, sub_ses_str):
    """checks if the subject can be included or not.
    Will write a INCLUDE or EXCLUDE file into the interim
    data folder of the subject and session.

    Args:
        event_counts (dict): counts of each condition
        min_trials (int): minimum number of trials which are necessary for further processing
        sub_ses_str (str): subject session str
    """
    if min(event_counts.values()) < min_trials:
        print(f"Subject has not enough trials, as defined by min_trials. Min_trials is {min_trials} but subject has {event_counts.values()}")
        # write an empty text file 
        with open(f"{dirs['interim_dir']}{sub_ses_str}/EXCLUDE", "w") as f:
            pass
        print(f"A file EXCLUDE has been added to interim data folder of subject and session.")
        # if an inclusion file already existed, delete it
        if os.path.exists(f"{dirs['interim_dir']}{sub_ses_str}/INCLUDE"):
            os.remove(f"{dirs['interim_dir']}{sub_ses_str}/INCLUDE")
            print(f"The former INCLUDE file has been deleted.")
        # exit the processing
        sys.exit(0) # exiting without error code --> SLURM thinks it's a success
    else:
        print(f"Subject has enough trials. Min_trials is {min_trials} and subject has {event_counts.values()}. Subject can be included. Multiverse processing is starting.")
        # write an empty text file 
        with open(f"{dirs['interim_dir']}{sub_ses_str}/INCLUDE", "w") as f:
            pass
        print(f"A file INCLUDE has been added to interim data folder of subject and session.")
        # if an exclusion file already existed, delete it
        if os.path.exists(f"{dirs['interim_dir']}{sub_ses_str}/EXCLUDE"):
            os.remove(f"{dirs['interim_dir']}{sub_ses_str}/EXCLUDE")
            print(f"The former EXCLUDE file has been deleted.")
    


def ica_eog_emg(raw, sub_ses_str, method='eog', save_ica=False, save_plot=False, save_str=''):
    """ 
    ICA to find EOG or EMG artifacts and remove them
    For EOG, EOG channels need to be defined in raw.
    Args:   
        raw (mne.raw): raw data object
        method (str): automated detection of either eye (eog) or muscle (emg) artifacts
        save_ica (bool): save ica object
        save_plot (bool): save ica plots
        save_str (str): string to add to save name
    Returns:
        raw_new (mne.raw): raw data with artifacts regressed out
        n_corr_components (int): number of found components that correlate with eog/emg
    """
    raw_new = raw.copy()

    # HPF (necessary for ICA), l_freq is HPF cutoff, and h_freq is LPF cutoff
    filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None, n_jobs=-1)

    # ica
    ica = ICA(n_components=20, max_iter="auto", method='picard', random_state=97)
    ica.fit(filt_raw) # bads seem to be ignored by default
    if save_ica:
        ica.save(f"{dirs['model_dir']}{sub_ses_str}/ica/{method}_{save_str}-ica.fif", overwrite=True, verbose='WARNING')
    # viz
    #ica.plot_sources(raw_new, show_scrollbars=False) # original data is used here

    # automatic detection of EOG/EMG components
    ica.exclude = []

    if method == 'eog':
        # find which ICs match the EOG pattern
        indices, scores = ica.find_bads_eog(raw_new)
    elif method == 'emg':
        # find which ICs match the muscle pattern
        indices, scores = ica.find_bads_muscle(raw_new)
    
    print(f'Found {len(indices)} independent components correlating with {method.upper()}.')
    ica.exclude.extend(indices) 

    # barplot of ICA component "EOG/EMG match" scores
    if save_plot:
        # plot scores
        f = ica.plot_scores(scores,
                        title=f'IC correlation with {method.upper()}',
                        show=False)
        f.savefig(f"{dirs['plot_dir']}{sub_ses_str}/ica/{method}_scores_{save_str}.png", dpi=100)
        
        # plot diagnostics
        if indices: # only if some components were found to correlate with EOG/EMG
            g = ica.plot_properties(raw, 
                                    picks=indices, 
                                    show=False)
            for gi, p in zip(g, indices):
                gi.savefig(f"{dirs['plot_dir']}{sub_ses_str}/ica/{method}_diagnostics_ic{p}_{save_str}.png", dpi=100)
        plt.close('all')

    # plot ICs applied to raw data, with EOG matches highlighted
    #ica.plot_sources(raw, show_scrollbars=False)

    # because it is an "additive" process, the ica component removel on filtered data 
    # can be used on the unfiltered raw data (source: tutorial)

    # ica.apply() changes the Raw object in-place, so let's make a copy first:
    ica.apply(raw_new)

    return raw_new, len(indices)


# def eog_regression(raw, events, event_id, plot=True):
#     """_summary_

#     inspired by: https://mne.tools/dev/auto_tutorials/preprocessing/35_artifact_correction_regression.html
#     especially the part: "create EOG evoked before regression"
#     based on R. J. Croft and R. J. Barry. Removal of ocular artifact from the EEG: a review. Clinical Neurophysiology, 30(1):5–19, 2000. doi:10.1016/S0987-7053(00)00055-1.
#     Args:
#         raw (_type_): _description_
#         plot (bool, optional): _description_. Defaults to True.

#     Returns:
#         _type_: _description_
        
#     DISCONTINUED: it would be a mess to account for all possible different references
#     OR IF USED: use only before referencing step in MULTIVERSE
#     """

#     # i need to set the artificial reference channel, if no other reference is specifie
#     raw2 = raw.copy()
#     raw2.add_reference_channels(['artificial']) # has 0s everywhere
#     epochs_orig = mne.Epochs(raw2, events, event_id=event_id, preload=True)
    
#     plot_kwargs = dict(picks="all", ylim=dict(eeg=(-10, 10), eog=(-5, 15)))

    
#     eog_epochs = mne.preprocessing.create_eog_epochs(raw2)
#     # We need to explicitly specify that we want to average the EOG channel too.
#     eog_evoked = eog_epochs.average("all")
#     eog_evoked.plot("all")
#     fig.set_size_inches(6, 6)




#     # perform regression on the evoked blink response
#     model_evoked = EOGRegression(picks="eeg", picks_artifact="eog").fit(eog_evoked)
#     fig = model_evoked.plot(vlim=(-1, 1))
#     fig.set_size_inches(3, 2)

#     # apply the regression coefficients to the original epochs
#     epochs_clean_evoked = model_evoked.apply(epochs_orig).apply_baseline()
#     fig = epochs_clean_evoked.average("all").plot(**plot_kwargs)
#     fig.set_size_inches(6, 6)

#     # for good measure, also show the effect on the blink evoked
#     eog_evoked_clean = model_evoked.apply(eog_evoked)
#     eog_evoked_clean.apply_baseline()
#     eog_evoked_clean.plot("all")
#     fig.set_size_inches(6, 6)
    
    
#     # TODO this delivers extremly high values, does not seem right
#     # TODO maybe include more steps from tutorial but upper examples
#     return raw_cleaned

def robust_average_PREP(raw, delete_bad_info=True):
    """Do robust averaging AND find bad channels using PREP.

    Args:
        raw (mne.raw): raw data (TODO: is epoch also working?)
        delete_bad_indo (bool): delete bads from the info object, as being done for MULTIVERSE analysis
    Returns:
        raw (mne.raw): processed raw data object
    """
    prep_params = {'ref_chs': 'eeg', #A list of channel names to be used for rereferencing. Can be a str ‘eeg’ to use all EEG channels.
                'reref_chs': 'eeg', #A list of channel names to be used for line-noise removed, and referenced. Can be a str ‘eeg’ to use all EEG channels.
                'line_freqs': [], # empty to skip line noise removal, list of floats indicating frequencies to be removed. For example, for 60Hz you may specify np.arange(60, sfreq / 2, 60). Specify an empty list to skip the line noise removal step.
                'max_iterations': 4} #The maximum number of iterations of noisy channel removal to perform during robust referencing. Defaults to 4.
    try:
        # RANSAC can fail, when there are too few good channels left. Therefore try first with RANSAC, then without if it failed.
        out = pyprep.PrepPipeline(raw, 
                                prep_params=prep_params, 
                                montage=raw.get_montage(),
                                ransac=True, # whether to ALSO use RANSAC for noisy channel detection
                                channel_wise=True, # whether to use channel-wise RANSAC (more RAM demanding)
                                max_chunk_size=None, # NONE = maximum RAM available
                                random_state=42,
                                filter_kwargs=None, # defaults for mne.filter.notch_filter
                                matlab_strict=False, # whether to use strict matlab implementation
                                )
        out.fit()
    except OSError:
        out = pyprep.PrepPipeline(raw, 
                                prep_params=prep_params, 
                                montage=raw.get_montage(),
                                ransac=False, # whether to ALSO use RANSAC for noisy channel detection
                                channel_wise=True, # whether to use channel-wise RANSAC (more RAM demanding)
                                max_chunk_size=None, # NONE = maximum RAM available
                                random_state=42,
                                filter_kwargs=None, # defaults for mne.filter.notch_filter
                                matlab_strict=False, # whether to use strict matlab implementation
                                )
        out.fit()        
        
    raw_prepped = out.raw_eeg.copy() # out.raw would be better, but PREP is messing up the eye data, so drop it here
    
    # remove information about bad channels (for MULTIVERSE)
    if delete_bad_info:
        while len(raw_prepped.info['bads']) > 0: 
            raw_prepped.info['bads'].pop()

    # add eye electrodes, which are discarded during PREP
    raw_eog = raw.copy().pick_types(eog=True)

    # add the eye channels to raw
    new_channel_data = raw_eog.get_data()
    new_channel_name = raw_eog.ch_names 

    # Create a new info object for the new channel
    new_info = mne.create_info(new_channel_name, raw_prepped.info['sfreq'], ch_types='eog') # WAS EEG

    # Create a new RawArray object for the new channel
    new_channel_raw = mne.io.RawArray(new_channel_data, new_info)

    # Add the new channel to the raw data
    raw_prepped.add_channels([new_channel_raw], force_update_info=True)

    # some characteristics for saving
    chars = {'noisy_channels_before_interpolation': out.noisy_channels_before_interpolation,
             'noisy_channels_after_interpolation': out.noisy_channels_after_interpolation}
    
    return raw_prepped, chars




def find_bad_channels_PREP(raw, as_dict=False):
    """Find bad channels using PREP pipeline.

    Args:
        raw (mne.raw): raw data (TODO: is epoch also working?)
        as_dict (bool): return bad channels as dict or list
    Returns:
        bads (list or dict): list or dict of bad channels
    """
    noisy = pyprep.NoisyChannels(raw, 
                                do_detrend=True, # lpf for detection applied, do it always
                                random_state=42, # make it reproducible
                                matlab_strict=False) # we use the python implementation
    noisy.find_all_bads(ransac=True, channel_wise=True, max_chunk_size=None)
    
    return noisy.get_bads(as_dict=as_dict)
        

def smoothing(data, fwhm=10):
    """Applies Gaussian smoothing on the data

    Args:
        data (mne.raw or mne.Epochs): raw data object
        fwhm (int, optional): FWHM of the Gaussian kernel. Defaults to 10.
    Returns:
        data (mne.raw or mne.Epochs): filtered raw data object
    """
    sigma = fwhm / 2.3548
    data.apply_function(gaussian_filter1d, sigma=sigma)
    return data.copy()

def univariate_noise_normalization(epochs, baseline=(-0.2,0.)):
    """Univariate Noise Normalization according to Xiu et al / Ashton et al (Baby EEG MVPA papers).
    Xiu: "univariate noise normalization (i.e. each trial and channel separately was z-transformed based on baseline activity from –200 to 0ms"

    Args:
        epochs (mne.epochs): epochs to be transformed
        baseline (tuple, optional): Baseline period for calculating mean and std. Defaults to (-200,0).

    Returns:
        epochs_normalized
    """
    baseline_data = epochs.get_data(tmin=baseline[0], tmax=baseline[1])
    epoch_data = epochs.get_data(tmin=epochs.tmin, tmax=None)
    # here use epochs.tmin, for cases where not all data is used for baselining, but still needs to be transformed

    # Calculate the mean for each channel and trial (axis=2)
    mean_values = np.mean(baseline_data, axis=2)

    # Calculate the standard deviation for each channel and trial (axis=2)
    std_values = np.std(baseline_data, axis=2)

    # z normalize epochs
    norm_epoch_data = (epoch_data - mean_values[:,:,np.newaxis]) / std_values[:,:,np.newaxis]

    return mne.EpochsArray(
        norm_epoch_data,
        info=epochs.info,
        events=epochs.events,
        tmin=epochs.tmin,
        event_id=epochs.event_id,
        metadata=epochs.metadata,
    )

def train_test_split_epochs(epochs, test_indices, max_ind_train, validation_indices=[]):
    """ 
    split epochs into train and test set 
    
    Args:
        epochs (mne.epochs): epochs object
        test_indices (list): indices of test data which are always present
        max_ind_train (int): minimum index of datapoints which belongs always to test data
        validation_indices (list): indices of validation data. 
            If not present, no validation set is created. Defaults to empty list.
    
    Returns:
        epochs_train (mne.epochs): epochs object of train data
        epochs_test (mne.epochs): epochs object of test data
        epochs_validation (mne.epochs): epochs object of validation data (only returned if validation indices are given)
    """
    # List to store the new 'Epochs' objects for each condition
    #new_epochs_list = []
    train = []
    test = []
    validation = []

    # Iterate through each condition
    for condition_name in epochs.event_id:

        for i in range(len(epochs[condition_name])):

            if i in test_indices or i >= max_ind_train:
                test.append(epochs[condition_name][i])
            elif i in validation_indices:
                validation.append(epochs[condition_name][i])
            else:
                train.append(epochs[condition_name][i])
        
    # Concatenate the new 'Epochs' objects from each condition
    epochs_train = mne.concatenate_epochs(train)
    epochs_test = mne.concatenate_epochs(test)
    if validation_indices:
        epochs_validation = mne.concatenate_epochs(validation)
        
        return epochs_train, epochs_test, epochs_validation
    else:     
        return epochs_train, epochs_test


# new version: where data is split for 5-f-CV instead of one single holdout set or one holdout set and one validation set
def train_test_split_epochs_kfcv(epochs, cv=5):
    """ 
    Splits an epochs object into cv folds of approximately same sizes.
    Also returns a rest epoch, containing the remaining epochs
    that would have lead to imbalance.    

    Args:
        epochs (_type_): epochs object
        cv (int, optional): How many Cross-Validation folds. 
        In that many junks the data will be separated trying to
        find similarly sized sets.  Defaults to 5.
    Returns:
        epochs_chunks (list of epochs): chunks of epochs
        rest_chunks (epochs object): epochs object of the remaining data    
    
    """
    # DEBUG
    #epochs = epochs_backup.copy()
    #epochs = epochs_ar.copy()

    # get event counts from epochs object
    event_counts = {}
    for key in epochs.event_id.keys():
        event_counts[key] = len(epochs.events[epochs.events[:, 2] == epochs.event_id[key]])

    # get minimum number of trials per condition
    n_smallest_class = min(event_counts.values())
    
    # split the data into cv folds of approximately same sizes
    chunk_sizes = split_approximate(n_smallest_class, cv, descending=True)

    #epochs = epochs_ar.copy()

    # make it deterministic

    # Shuffle the epochs to randomize the order
    np.random.seed(22) # must be right before the random number generator!
    # must be np.random.seed, because np.random.permutation is used
    epochs = epochs[np.random.permutation(len(epochs))]
    # DEBUG
    #print(epochs.events[:10, 2])
    # generate a list of random samples per condition (for all chunks together first)
    epoch_indices = []
    rest_indices = []
    for condition in event_counts.keys():
        # indices to be used for chunks
        random.seed(21) # RANDOM, because the function is random and not np.random
        epoch_indices.append(random.sample(
                range(event_counts[condition]), 
                n_smallest_class)) 
        # indices to be put to rest chunk
        rest_indices.append([i for i in range(event_counts[condition]) if i not in epoch_indices[-1]])
    
    # DEBUG: print indices
    print(f"epoch indices: {epoch_indices}")
    print(f"rest indices: {rest_indices}")
        
    # convert condition indices to indices for each condition and chunk
    chunk_indices = []
    for s, size in enumerate(chunk_sizes):
        this_chunk_indices = []
        for c, condition in enumerate(event_counts.keys()):
            start_idx = sum(chunk_sizes[:s])
            end_idx = start_idx + size
            this_chunk_indices.append(epoch_indices[c][start_idx:end_idx])
        chunk_indices.append(this_chunk_indices)

    # DEBUG: print indices
    #print(f"chunk indices: {chunk_indices}")                    
    
    # put epochs into the chunks
    epochs_chunks = []
    for chunk in chunk_indices:
        epochs_chunk = []
        for this_condition, this_chunk_indices in zip(epochs.event_id.keys(), chunk):
            epochs_chunk.append(epochs[this_condition][this_chunk_indices])
        epochs_chunk = mne.concatenate_epochs(epochs_chunk)
        epochs_chunks.append(epochs_chunk)

    # DEBUG
    #for i in epochs_chunks:
    #    print(f"epochs chunk {i} with len {len(i)}")
                    
    rest_chunk = []
    for this_condition, rest_indices in zip(epochs.event_id.keys(), rest_indices):
        # skip if empty condition, might lead to concatenation error if first condition is empty
        # TODO: double check if the condition numbers correspond to the chunks and rest chunks later
        if len(epochs[this_condition][rest_indices]) > 0:
            rest_chunk.append(epochs[this_condition][rest_indices])
    
    # DEBUG
    #for i in rest_chunk:
    #    print(f"rest chunk {i} with len {len(i)}")
    
    # if all conditions are same size, rest chunk is empty
    if rest_chunk:
        rest_chunk = mne.concatenate_epochs(rest_chunk)    
    
    return epochs_chunks, rest_chunk
    
    
def split_approximate(x, n, descending=True):
    """Splits a number x into n parts of approximately the same size.

    Args:
        x (int): number to be split
        n (int): how many splits
        descending (bool, optional): Sorting descending. Defaults to True.

    Raises:
        ValueError: If x < n

    Returns:
        _type_: _description_
    """
    
    sizes = []
    # If we cannot split the 
    # number into exactly 'N' parts
    if(x < n): 
        raise ValueError("ValueError: x should be bigger than n!")
 
    # If x % n == 0 then the minimum 
    # difference is 0 and all 
    # numbers are x / n
    elif (x % n == 0):
        for i in range(n):
            sizes.append(x//n)
    else:
        # upto n-(x % n) the values 
        # will be x / n 
        # after that the values 
        # will be x / n + 1
        zp = n - (x % n)
        pp = x//n
        for i in range(n):
            if(i>= zp):
                sizes.append(pp + 1)
            else:
                sizes.append(pp)
    
    # sort sizes from high to low (or vice versa)
    sizes.sort(reverse=descending)
    return sizes
    
    
def split_sublists(sublists, sizes):
    result = []
    for sublist, size in zip(sublists, sizes):
        # Ensure the sizes in 'sizes' list sum up to the length of the sublist
        if sum(sizes) != len(sublist):
            raise ValueError("The sizes in 'sizes' list do not match the length of the sublist.")

        split_parts = []
        start = 0
        for part_size in size:
            end = start + part_size
            split_parts.append(sublist[start:end])
            start = end
        result.append(split_parts)
    return result
    

def autorej(epochs, log_path=None, plot_path=None, ar_model_path=None, mode='estimate', 
            n_interpolate=False, consensus=False, show_plot=False, 
            save_plot=False, save_model=False, save_log=False, save_drop_dict=False,
            drop_dict_path=None):
    """
    Run Autoreject on trials.
    
    Args:
        epochs (mne.epochs): epochs object
        log_path (str): filename where drop log will be saved as csv
        plot_path (str): filename where drop log will be saved as png
        ar_model_path (str): filename where ar model will be saved or loaded from
        mode (str): either estimate ar model or load ar model from disk .hdf5 (defaults to estimate)
        n_interpolate (int): number of channels to interpolate (defaults False, then use default of package)
        consensus (bool): if consensus is used (defaults False, then use default of package)
        show_plot (bool): if plot shall be shown (defaults False)
    
    Returns:
        epochs_ar (mne.epochs): artifact rejected epochs
        drop_log (pandas.DataFrame): drop log  
    
    """
    # input checks
    if save_plot is True:
        if not isinstance(plot_path, str):
            raise ValueError("save_plot is true but plot_path not given.")
    if save_model is True:
        if not isinstance(ar_model_path, str):
            raise ValueError("save_model is true but ar_model_path not given.")
    if save_log is True:
        if not isinstance(log_path, str):
            raise ValueError("save_log is true but log_path not given.")   
    if save_drop_dict is True:
        if not isinstance(drop_dict_path, str):
            raise ValueError("save_drop_dict is true but drop_dict_path not given.")
    if mode=='load':
        if not isinstance(ar_model_path, str):
            raise ValueError("mode=='load' but ar_model_path not given.")    
    
    epochs.del_proj()  # remove proj, don't proj while interpolating (https://autoreject.github.io/stable/auto_examples/plot_auto_repair.html)

    # possibility 1: global rejection threshold
    #global_reject = get_rejection_threshold(epochs, decim=2) # global rejection threshold, one simple possibility
    #print('The global rejection dictionary is %s' % reject)

    # must be lists for a hyperparameter optimization
    n_interpolate = [n_interpolate] if n_interpolate else [4, 8, 12, 16]
    consensus = [consensus] if consensus else np.linspace(0, 1.0, 11)
    
    # automated estimation of rejection threshold based on channel and trial per participant
    if mode=='estimate':
        ar = AutoReject(n_interpolate=n_interpolate, 
                        consensus=consensus,
                        random_state=11,
                        n_jobs=-1, 
                        verbose=False)
        ar.fit(epochs)  # fit only a few epochs if you want to save time
        if save_model:
            ar.save(ar_model_path, overwrite=True)  # save the object to disk
    elif mode=='load':
        ar = read_auto_reject(ar_model_path)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    # plot the rejection log and save plot
    rej_plot = reject_log.plot('horizontal', show=show_plot)
    if save_plot:
        rej_plot.savefig(plot_path, dpi=300)
    
    # save reject log
    if save_log:
        reject_log.save(log_path, overwrite=True) # must end with .npz

    # calculate the number of dropped trials per condition
    dropped_epochs = epochs[reject_log.bad_epochs]
    drop_dict = count_epochs_per_condition(dropped_epochs)
    if save_drop_dict:
        df_drop = pd.DataFrame(drop_dict)
        df_drop.to_csv(drop_dict_path, index=False)

    return epochs_ar, reject_log


# function to count bads / interpolated from AR reject log matrix
def summarize_artifact_interpolation(reject_log):
    """if epochs are ONLY interpolated (not rejected),
    this function returns some summaries about which trials and channels
    or how many percent of them are interpolated.

    Args:
        reject_log (reject_log object): ar reject_log object

    Returns:
        dict: interp_frac_channels (key value pair of channel and percentage rejected)
        dict: interp_frac_trials (key value pair of trial and percentage rejected)
        float: total_interp_frac
    """
    ch_names = reject_log.ch_names
    armat = reject_log.labels
    armat_binary = np.where(armat == 2.0, 1.0, armat) 
    # 2 means interpolated, 1 would be bad(?), but we then interpolate anyway therfore not in the data, 0 means ok
    mean_per_channel = np.mean(armat_binary, axis=0)
    mean_per_trial = np.mean(armat_binary, axis=1)
    interp_frac_channels = {channel: value for channel, value in zip(ch_names, mean_per_channel)}
    interp_frac_trials = {channel: value for channel, value in enumerate(mean_per_trial)}
    total_interp_frac = np.mean(mean_per_trial)
    return interp_frac_channels, interp_frac_trials, total_interp_frac


def count_epochs_per_condition(epochs):
    """
    uses epochs file, and counts the number of trials per 
    condition (extracted from the events Nx3 array).
    
    Args:
        epochs (mne.epochs object)
    
    Returns:
        event dict ('event_id': event_counts)
    """
    conditions = list(epochs.event_id)
    return {condition: len(epochs[condition]) for condition in conditions}


def oversample_epochs(epochs, drop_dict, min_samples=4):
    """
    Uses drop_dict dict (number of missing trials N per condition)
    to oversample remaining trials to fill the N missing trials.
    
    Args:
        epochs (mne.epochs object)
        drop_dict (dict): number of missing trials per condition
        add_offset (bool): if event times should be changed or remain the same
    Returns:
        oversampled_epochs (mne.epochs object)
    
    """
    #new_epochs = []
    counter = 0
    conditions = drop_dict.keys()

    # DEBUG
    #print(epochs)
    #print(drop_dict)

    for condition in conditions:
        
        # find random n epochs for condition
        possible_indices = list(range(len(epochs[condition])))

        # if there are no epochs for a condition: warn and don't oversample this epoch
        if not possible_indices:
            print(f"Warning: no trials left to perform oversampling in condition {condition}. Return only oversampled epochs from other conditions.")
            continue
        elif len(possible_indices) < min_samples:
            print(f"Warning: <= {min_samples} trials left to perform oversampling in condition {condition}. Return only oversampled epochs from other conditions.")
            continue


        # DEBUG
        #print(condition)
        #print(possible_indices)

        # if there are more possible indices than trials to be drawn, draw without replacement
        if len(possible_indices) > drop_dict[condition]:
            np.random.seed(22) # must be right before the random number generator!
            random_idxs = np.random.choice(
                a=possible_indices,
                size=drop_dict[condition],
                replace=False) # a trial can only be drawn once

            if counter==0:
                new_epochs = epochs[condition][random_idxs]
                counter += 1
            else:
                new_epochs = mne.concatenate_epochs([new_epochs, epochs[condition][random_idxs]],
                                                        on_mismatch='warn',
                                                        add_offset=True)

        else:
            #print(f'Condition {condition}: Oversampling with replacement done, because to few trials leftover.')

            dividend = drop_dict[condition]
            divisor = len(possible_indices)

            quotient = dividend // divisor
            remainder = dividend % divisor
            #print(f"{divisor} fits into {dividend} a total of {quotient} times with a remainder of {remainder}.")
            n_draws = quotient + 1

            for i in range(n_draws):
                size = remainder if i == n_draws-1 else divisor
                np.random.seed(21) # must be right before the random number generator!
                random_idxs = np.random.choice(
                    a=possible_indices,
                    size=size,
                    replace=False)

                if counter==0:
                    new_epochs = epochs[condition][random_idxs]
                    counter += 1
                else:
                    new_epochs = mne.concatenate_epochs([new_epochs, epochs[condition][random_idxs]],
                        on_mismatch='warn',
                        add_offset=True)

    # add oversampled epochs to old epochs
    epochs = mne.concatenate_epochs([epochs, new_epochs],
                on_mismatch='warn',
                add_offset=True) # if False, this would hinder concatenation of epochs because they are duplicates

    return epochs

# alternative concat, successive
#final_epochs = mne.concatenate_epochs([epochs, new_epochs[3]], on_mismatch='warn', add_offset=True) # if False, this would hinder concatenation of epochs because they are duplicates

def build_query(param_dict):
    """Takes a dict (key value pairs) of parameters and builds a query string 
    to query the epochs object using the metadata.
    It particulary makes the query correct if there are comparisons with NoneType objects
    and comparisons with strings.

    Args:
        param_dict (dict): key value pairs of parameters
    """
    query_str = ''
    for i, key in enumerate(param_dict.keys()):
        # compare NoneType objects
        if param_dict[key] is None:
            query_str += f'{key}.isna()'
        elif type(param_dict[key]) == str:
            query_str += f'{key} == "{param_dict[key]}"'
        elif type(param_dict[key]) == bool:
            query_str += f'{key} == {param_dict[key]}'
        elif type(param_dict[key]) in [int, float]:
            query_str += f'{key} == {param_dict[key]}'
        else:
            continue
        if i < len(param_dict.keys()) - 1:
            query_str += " and "

    return query_str


def save_identifier_metadata(metadata, output_file):
    """ saves the identifier and metadata of a multiverse path
    
    Args:
        epochs (mne.Epochs): epochs object
        output_file (str): filename where to save the metadata
    """
    meta = metadata.head(1)

    # replace none with 
    
    # if first path: create new file
    if meta["path_id"].iloc[0] != 1:
        # load already existing file
        meta_old = pd.read_csv(output_file)
        # append new row
        meta = pd.concat([meta_old, meta], ignore_index=True)
    
    # save
    meta.to_csv(output_file, index=False)
    

# DEBUG
#input_dir_pattern=f"{dirs['interim_dir']}{sub_ses_str}/1000Hz*train-epo.fif", 
#output_file=f"{dirs['processed_dir']}{sub_ses_str}/1000Hz_train-epo.fif", 
#verbose=False

def concatenate_all_epochs(input_dir_pattern, output_file, save=True, verbose=False):
    """ 
    loads and concatenates all epochs files in a directory which mach the pattern.
    Baseline is deleted, else concatenation function would fail
    or do an actual baseline correction according to the values
    (which is not wanted).
    Some metadata is also adjusted.
    
    Args:
        input_dir_pattern (str): directory and regex pattern where to look for epochs files
        output_file (str): full filename where to save concatenated epochs
        save (bool): if True, saves concatenated epochs
        verbose (bool): if True, prints out some information
    Returns:
        concatenated_epochs (mne.epochs object)
    
    """
    
    leaves = glob(input_dir_pattern)

    # load all leaves
    epochs_leaves = [mne.read_epochs(i, verbose=verbose) for i in leaves]
    #epochs_leaves_backup = [i.copy() for i in epochs_leaves] # need to copy each element, not whole list, because then only references to the same objects are copied

    # pseudo-remove baseline to be able to concatenate epochs
    for i in epochs_leaves:
        i.baseline = None # use None, because if you chose a number, baseline correction will be applied under the hood during concatenation

    # concatenate
    concatenated_leaves = mne.concatenate_epochs(epochs_leaves, verbose=verbose)

    # extract the lower edge of the baseline, as easier to index --> not done anymore, because only 1 value of "base" is given
    #concatenated_leaves.metadata["base_l"] = concatenated_leaves.metadata["base"].apply(lambda x: x[0])
    # also extract the upper edge of the baseline
    #concatenated_leaves.metadata["base_h"] = concatenated_leaves.metadata["base"].apply(lambda x: x[1])

    # save new huge epochs object
    if save:
        concatenated_leaves.save(output_file, overwrite=True)

    return concatenated_leaves

def check_chunk_leakage(file_pattern="63Hz_chunk*-epo.fif"):
    """
    Check for data leakage based on epoch indices. So check, if in the multiverse
    augmented dataset per chunk, there are really the same epochs augmented, and not same 
    epochs augmented in different ways in different chunks, which would totally
    inflate model performance.

    Args:
        file_pattern (str): file pattern to get all the chunks
    """
    files = sorted(glob(f"{dirs['processed_dir']}/{sub_ses_str}/multiverse/{file_pattern}"))
    
    chunks = [mne.read_epochs(i, verbose=50) for i in files]
    
    # check the number of unique trial ids per chunk
    uniq = []
    for chunk in chunks:
        uniq.append(chunk.metadata["trial_id"].unique().shape[0])
    print(f"Number of unique trial ids per chunk: {uniq}")
    
    # check if any trial_ids overlap between chunks
    for i, chunk in enumerate(chunks):
        for j, chunk2 in enumerate(chunks):
            if i==j:
                continue
            a = set(chunk.metadata["trial_id"])
            b = set(chunk2.metadata["trial_id"])
            c = a.intersection(b)
            if len(c) > 0:
                print(f"Overlap between chunks {i} and {j}: {c}")
                raise ValueError('ERROR: Data leakage detected!')
    
    # check the combined trial ids across chunks
    combined = mne.concatenate_epochs(chunks)
    total_ids = combined.metadata["trial_id"].unique().shape[0]
    
    # check if the sum of unique ids per chunk is equal to the total number of unique ids
    sum_of_chunk_ids = np.sum(np.array(uniq))
    
    if sum_of_chunk_ids == total_ids:
        print("No leakage")
    elif sum_of_chunk_ids > total_ids:
        print("More unique ids in chunks than in total --> leakage")
        raise ValueError('ERROR: Data leakage detected!')
    elif sum_of_chunk_ids < total_ids:
        print("Less unique ids in chunks than in total --> something went wrong")
        raise ValueError('ERROR: Data leakage detected!')
        

def merge_age_column(input_df, file_path, key_column='sub_ses_str',
                     merge_column='age', input_key_column='session'):
    """
    Merge the 'age' column from a stored table onto the input DataFrame.

    Parameters:
    - input_df (pd.DataFrame): Input DataFrame to which the 'age' column will be added.
    - file_path (str): Path to the file containing the table with the 'age' column.
    - key_column (str, optional): Key column in the stored table used for merging. Default is 'sub_ses_str'.
    - merge_column (str, optional): Column to be merged onto the input DataFrame. Default is 'age'.
    - input_key_column (str, optional): Key column in the input DataFrame used for merging. Default is 'session'.

    Returns:
    - pd.DataFrame: Input DataFrame with the 'age' column merged.
    """
    # Read the table with the 'age' column
    age_table = pd.read_excel(file_path)

    # Merge based on the specified key columns
    merged_df = pd.merge(input_df, age_table[[key_column, merge_column]], left_on=input_key_column, right_on=key_column, how='left')

    # Drop the duplicated key column (if added from the age_table)
    merged_df = merged_df.drop(columns=[key_column])

    return merged_df

# Example usage:
# input_df = ...  # Your input DataFrame
# file_path = 'path/to/age_table.csv'  # Replace with the actual path to your file
# merged_result = merge_age_column(input_df, file_path)


# get all session identifiers for a particular folder from sessions.json file
def get_session_identifiers(type='included'):
    # get all included sessions

    with open(f"{dirs['base_dir']}/sessions.json", 'r') as file:
        data = json.load(file)
    output = data[type]
    return output

def eeg_std_on_raw(raw, plot=True):
    """ calculates standard deviation
    for each electrode on raw data 
    
    Args:
        raw (mne.io.Raw): raw data
        plot (bool): if True, plots the standard deviation
    """
    data = raw.get_data().transpose()
    stds = np.std(data, axis=0)
    df_std = pd.DataFrame({"electrode": raw.ch_names[:-1], # exclude stim
                       "std": stds[:-1]}) # exclude stim

    if plot:
        df_std.plot.bar(x="electrode", y="std", figsize=(20,10))
        #sns.swarmplot(data=df_std, x="electrode", y="std", hue="condition")
        plt.show()
    return df_std




def mne_save(obj, name, plot_dir):
    """ saves an object to a .fif file
    :param obj: an object to be saved
    :param name: name of the file to be saved
    """
    if not name:
        name = "temp.fif"
    mne.write_evokeds(name, obj)




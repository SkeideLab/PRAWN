
# imports
import itertools
import mne
import sys, os
from tqdm import tqdm
from glob import glob
import numpy as np
import pandas as pd
if sys.platform.startswith('darwin'):
    mne.viz.set_browser_backend('qt', verbose=None) # 'qt' or 'matplotlib'
import seaborn as sns
sns.set()
import re
import time
import random
import json
import pickle

# define base directory
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)
os.chdir(base_dir)

from src.utils import *
from src.config import *
from src.ml import load_and_prepare_chunks, load_and_prepare_chunks_merge
from braindecode.preprocessing import exponential_moving_standardize
from braindecode import EEGClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, cross_val_predict
from joblib import Parallel, delayed, dump
from multiprocessing import Manager

# set some parameters
model = 'EEGNetv4'

# DEBUG TODO comment!
#sub_ses_str = "sub-001"
#species="inter"
#merge= True

# define subject and session by arguments to this script
if len(sys.argv) not in [3,4]:
    print("Usage: python script.py sub_ses_str [inter|intra_human|intra_monkey] [merge]")
    sys.exit(1)
else:
    sub_ses_str = sys.argv[1]
    species = sys.argv[2]
    
    if len(sys.argv) == 4:
        merge = sys.argv[3]
        if merge == "merge":
            merge = True
            print(f"merge the sessions of participant {sub_ses_str}")
        else:
            merge = False
            print(f"DO NOT merge the sessions of participant {sub_ses_str}")
    else:
        merge = False
        print(f"DO NOT merge the sessions of participant {sub_ses_str}")
        
sub = sub_ses_str[4:7]
if merge == False:
    ses = sub_ses_str[-3:]

# if adult subject, then less steps because they have much more data
if int(sub) >= 900:
    subject_group = 'adults'
else:
    subject_group = 'infants'


create_if_not_exist(f"{dirs['model_dir']}{sub_ses_str}/braindecode")


train_i = [0,1,2,3,4]
test_i = 5 # dummy, or used as separate test set again (rest chunk)

""" subset prep """    

#if merge == True:
    # l


# separate for 2 sessions
try:
    event_counts1 = json.load(open(f"{dirs['interim_dir']}{sub_ses_str}_ses-001/event_counts.json", 'r'))
    max_epochs1 = np.min(list(event_counts1.values()))
except:
    max_epochs1 = None

try:
    event_counts2 = json.load(open(f"{dirs['interim_dir']}{sub_ses_str}_ses-002/event_counts.json", 'r'))
    max_epochs2 = np.min(list(event_counts2.values()))
except:
    max_epochs2 = None

sessions = []
if max_epochs1 and max_epochs2:
    max_epochs = max_epochs1 + max_epochs2
    print(f"max_epochs: {max_epochs1} + {max_epochs2} = {max_epochs}")
    sessions.extend(["ses-001","ses-002"])
elif max_epochs1:
    max_epochs = max_epochs1
    print(f"max_epochs (only session 1): {max_epochs1}")
    sessions.append("ses-001")
elif max_epochs2:
    max_epochs = max_epochs2
    print(f"max_epochs (only session 2): {max_epochs2}")
    sessions.append("ses-002")
        

subset = max_epochs

delete_files_in_folder(f"{dirs['model_dir']}{sub_ses_str}/braindecode", f'*_{species}_*') 


# vary for subset analysis
n_trials_per_condition = subset #40
n_trials_per_junk = int(n_trials_per_condition / 5)

""" load data """

# load data for forking path
if len(sessions)==1:
    chunk_files_universe = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[0]}/250Hz_chunk*-epo.fif"))
    X_train, y_train, _, _, _, _, y_train_orig = load_and_prepare_chunks(
                train_files = [file for j, file in enumerate(chunk_files_universe) if j in train_i],  
                test_file = None, #chunk_files_universe[test_i],
                classification=species,
                flatten=False,
                asEpoch=False, # True
                #sfreq=sfreq,
                trials_per_cond_per_chunk=n_trials_per_junk, # 8*5chunks = 40 
                )
elif len(sessions)==2:
    chunk_files_universe1 = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[0]}/250Hz_chunk*-epo.fif"))
    chunk_files_universe2 = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[1]}/250Hz_chunk*-epo.fif"))

    X_train, y_train, _, _, _, _, y_train_orig = load_and_prepare_chunks_merge(
                train_files1 = [file for j, file in enumerate(chunk_files_universe1) if j in train_i],  
                train_files2 = [file for j, file in enumerate(chunk_files_universe2) if j in train_i],  
                test_file = None, 
                classification=species, 
                flatten=False,
                asEpoch=False, 
                #sfreq=sfreq,
                trials_per_cond_per_chunk=n_trials_per_junk, 
                )

# The exponential moving standardization function work on single trials, therefore:
for i in range(X_train.shape[0]):
    X_train[i,:,:] = exponential_moving_standardize(X_train[i,:,:], factor_new=0.001, init_block_size=None, eps=1e-4)


""" real model training and evaluation """
# train model
net = EEGClassifier(
    model, 
    module__final_conv_length='auto',
    train_split=None, #ValidSplit(0.2),
    max_epochs=200, 
    batch_size=16, 
    module__sfreq=250,
)

train_val_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

start_time = time.time()  # Start time

cvs = cross_validate(   net, 
                    X_train, 
                    y_train, 
                    scoring="accuracy", 
                    cv=train_val_split, 
                    n_jobs=-1, # NEW, prevent overload
                    return_estimator=True , 
                    return_train_score=True,
                    )

end_time = time.time()  # End time
elapsed_time = end_time - start_time
print(f"The CV took {elapsed_time} seconds to complete.")

validation_score = np.mean(cvs['test_score'])
print(f"The average validation score is  {validation_score}.")
print(f"Single validation scores: {cvs['test_score']}")

''' test the individual subclasses of the model '''

# Initialize an array to accumulate predictions
n_samples = len(y_train)
predictions = np.zeros((n_samples,))  # To store final predictions for each sample
prediction_counts = np.zeros((n_samples,))  # To track how many times each sample was predicted

# Iterate over each fold and make predictions with the corresponding estimator
for fold_idx, (train_idx, test_idx) in enumerate(train_val_split.split(X_train, y_train)):
    estimator = cvs['estimator'][fold_idx]  # Retrieve the estimator for this fold
    
    # Predict on the test set for this fold
    y_pred = estimator.predict(X_train[test_idx])
    
    # Store the predictions
    predictions[test_idx] += y_pred
    prediction_counts[test_idx] += 1

# Average the predictions across the splits
average_predictions = predictions / prediction_counts

# Convert averaged predictions to final predicted class (if it's a classifier with discrete labels)
# This works if your model's predictions are discrete (e.g., class labels).
final_predictions = np.round(average_predictions).astype(int)

# Calculate per-sample accuracy
individual_accuracies = (final_predictions == y_train).astype(float)

# Identify unique classes in y_train
unique_classes = np.unique(y_train_orig)

# Initialize a dictionary to store the accuracy for each class
class_accuracies = {}

# Iterate over each unique class
for class_label in unique_classes:
    # Get the indices of the samples belonging to this class
    class_indices = np.where(y_train_orig == class_label)[0]
    
    # Calculate accuracy for this class: 
    # compare the true labels with the predicted labels for this class
    class_accuracy = np.mean(final_predictions[class_indices] == y_train[class_indices]) # here y_train, not orig
    
    # Store the accuracy for this class
    class_accuracies[class_label] = class_accuracy

# save individual accuracies to file
with open(f"{dirs['model_dir']}{sub_ses_str}/braindecode/{subset}_{species}_subclass_accuracies.pck", 'wb') as f:
    pickle.dump(class_accuracies, f)

""" fake models training and evaluation """

manager = Manager() # shared list
shared_result_list = manager.list()
    
def process_braindecode_fake(shared_list):        

    # shuffle the labels
    y_train_shuffled = y_train.copy()
    y_train_shuffled = np.random.permutation(y_train_shuffled)
    
    # train model
    net = EEGClassifier(
        model, 
        module__final_conv_length='auto',
        train_split=None,
        max_epochs=200, 
        batch_size=16, 
        module__sfreq=250,
    )

    # TODO if 10f-CV takes to long for RAVEN, then do 5f
    train_val_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    cvs = cross_validate(net, 
                        X_train, 
                        y_train_shuffled, 
                        scoring="accuracy", 
                        cv=train_val_split, 
                        n_jobs=1, # new: prevent overload
                        return_estimator=False , #True,
                        return_train_score=True,
                        )
    
    # Append the single result to the shared list
    shared_list.append(np.mean(cvs['test_score']))



# Use Parallel to execute the processes
Parallel(n_jobs=-1)(delayed(process_braindecode_fake)(shared_result_list) for _ in range(1000)) 


# Access the results in the shared list
permutation_results = list(shared_result_list)

# write to file
with open(f"{dirs['model_dir']}{sub_ses_str}/braindecode/{subset}_{species}_permutations.pck", 'wb') as f:
    pickle.dump(permutation_results, f)
with open(f"{dirs['model_dir']}{sub_ses_str}/braindecode/{subset}_{species}_result.pck", 'wb') as f:
    pickle.dump(validation_score, f)

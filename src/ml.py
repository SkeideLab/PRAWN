import numpy as np 
import matplotlib.pyplot as plt
# import lightgbm as lgb
# from sklearn.metrics import accuracy_score
# from src.ml.bayesian_hyperparameter import *
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.svm import SVC
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
# from sklearn.pipeline import make_pipeline
# from sklearn.model_selection import cross_val_predict, StratifiedKFold, GridSearchCV, KFold, cross_val_score
# import pickle
import mne
from src.config import *

#from sklearn.decomposition import PCA
#from src.ml.EEGModels import EEGNet, EEGNetRK, DeepConvNet, ShallowConvNet
#from tensorflow.keras import utils as np_utils
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import backend as K
#import tensorflow as tf



# single path models

def check_data_size(y,
                    min_samples = 80,
                    min_samples_per_class = 20,
                    ):
    """ is there enough data for training and testing? """
    # count # in each class zeros in y
    y = np.array(y) # might be an np.array, convert to one
    n_zeros = np.sum(y==0)
    n_ones = np.sum(y==1)
    n_total = y.shape[0]
    if n_total < min_samples:
        print(f"not enough samples: {n_total} < {min_samples}")
        return False
    else:
        if n_zeros < min_samples_per_class or n_ones < min_samples_per_class:
            print(f"not enough samples per class: {n_zeros} zeros, {n_ones} ones")
            return False
        else:
            return True    
    
            
    

def extract_metadata_slice(epochs):    
    meta_slice = epochs.metadata.copy().drop("trial_id", axis=1).iloc[0]
    meta_str = "_".join(str(i) for i in meta_slice)
    return meta_slice, meta_str
    

def replace_labels_with_ints(y, classification='inter'):
    """ takes a vector and replaces the mne labels with integers according 
    to the classification that needs to be done
    :param y: vector with mne given labels
    :param classification: 'inter' or 'intra_human' or 'intra_monkey'
    :return: vector with integer labels
    """
    
    if classification == "inter":
        y_new = np.array([1 if i in [10003, 10004] else 0 if i in [10001, 10002] else -1 for i in list(y)])
    # TODO: check if the following returns correct labels
    elif classification == "intra_human":
        y_new =  np.array([1 if i in [10002] else 0 if i in [10001] else -1 for i in list(y)])
    elif classification == "intra_monkey":
        y_new =  np.array([1 if i in [10004] else 0 if i in [10003] else -1 for i in list(y)])
    elif classification == "multiclass": # all 4 classes shall be returned
        y_new =  y.copy()
    elif classification == "original": # all classes, starting with 0,1,2,...
        y_new = np.where(y == 10001, 0, 
                np.where(y == 10002, 1, 
                np.where(y == 10003, 2, 
                np.where(y == 10004, 3, -1))))    
    # DEBUG
    #print(y_new)
            
    if -1 in y_new:
        print("ERROR: y contained classes which are not meant to be!")
        raise ValueError("Invalid y (probably not yet filtered for the correct classes of interest.") 
    return y_new




def get_weights(y):
    """ takes a labels vector (binary, 0,1) and returns the weights 
    (relative probability) for each class """
    weights = np.bincount(y) / len(y)
    weights_dict = {0: weights[0], 1: weights[1]}
    return weights_dict

class NotEnoughTrialsException(Exception):
    pass

class HyperparamOptimizationException(Exception):
    pass

def load_and_prepare_train_test(train_file, test_file, validation_file=None,
                                classification='inter',
                                flatten=True,
                                min_train_samples=80, min_test_samples=20, min_validation_samples=20,
                                min_train_samples_per_class=20, min_test_samples_per_class=5, min_validation_samples_per_class=5):
    """ loads and processes epochs and returns training and testing data """
    
    epochs_train = mne.read_epochs(train_file, verbose=0)
    epochs_test = mne.read_epochs(test_file, verbose=0)
    
    good_picks = mne.pick_types(epochs_train.info, 
                            eeg=True, 
                            eog=False, 
                            exclude='bads') # TODO: are bad channels correctly labeled in each subject?
    # the "bads" information is assumed to be the same in train and test data
    
    # time range for prediction
    times = (0, None)

    X_train = epochs_train.get_data(picks=good_picks,tmin=times[0],tmax=times[1])
    X_test = epochs_test.get_data(picks=good_picks,tmin=times[0],tmax=times[1])
    
    # reshape to 2D; n_trials x n_channels*n_timepoints
    if flatten:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    
    y_train = epochs_train.events[:, -1]
    y_test = epochs_test.events[:, -1]
    y_train = replace_labels_with_ints(y_train, classification=classification)
    y_test = replace_labels_with_ints(y_test, classification=classification)
    
    if validation_file:
        epochs_validation = mne.read_epochs(validation_file, verbose=0)
        X_validation = epochs_validation.get_data(picks=good_picks,tmin=times[0],tmax=times[1])
        if flatten:
            X_validation = X_validation.reshape(X_validation.shape[0], -1)
        y_validation = epochs_validation.events[:, -1]
        y_validation = replace_labels_with_ints(y_validation, classification=classification)
        
    # test if data diemensions are big enough
    enough_train = check_data_size(y_train, min_samples=min_train_samples, min_samples_per_class=min_train_samples_per_class)
    enough_test = check_data_size(y_test, min_samples=min_test_samples, min_samples_per_class=min_test_samples_per_class)
    if validation_file:
        enough_validation = check_data_size(y_validation, min_samples=min_validation_samples, min_samples_per_class=min_validation_samples_per_class)
    
    if not enough_train or not enough_test:
        raise NotEnoughTrialsException("Not enough trials, not returning data.")
    elif validation_file and not enough_validation:
        raise NotEnoughTrialsException("Not enough trials, not returning data.")
        #return None


    
    train_weights = get_weights(y_train)
    meta_slice, meta_str = extract_metadata_slice(epochs_train)
    
    if not validation_file:
        return X_train, y_train, X_test, y_test, train_weights, meta_slice, meta_str
    else:
        return X_train, y_train, X_test, y_test, X_validation, y_validation, train_weights, meta_slice, meta_str


def get_train_test_chunk_id(cv=5):
    train_chunks = {i: [j for j in range(cv) if i != j] for i in range(cv)}
    test_chunks = {i: i for i in range(cv)}
    return train_chunks, test_chunks


def load_and_prepare_chunks(train_files, 
                            test_file, # can also be multiple or single
                            classification='inter',
                            flatten=True,
                            asEpoch=False, # if it shall be returned as epoch obj. Then flatten must be false.
                            sfreq=None,
                            trials_per_cond_per_chunk='all', # 'all': all trials, int: only the first X trials per condition and chunk
                            times = (0, None)):  # if not otherwise specified, start at t=0 to the end of the epoch
    """ loads and processes epochs and returns training and testing data """
    
    chunks = []
    for i, file in enumerate(train_files):
        if sfreq is None:
            chunk = mne.read_epochs(file, verbose=50)
        else:
            chunk = mne.read_epochs(file, verbose=50).resample(sfreq, npad="auto")
            
        chunk.metadata["inner_chunk_id"] = i # the chunk metadata is used for the inner cv loop (bayes opt), but the chunk i is not the actual chunk index of the raw file!

        if trials_per_cond_per_chunk == 'all':
            chunks.append(chunk)
        else:
            unique_conditions = np.unique(chunk.events[:, -1])  # Modify this line as per your data's event code structure

            filtered_chunk = None
            for condition in unique_conditions:
                # Select the first 8 trials for this condition
                condition_chunk = chunk[chunk.events[:, -1] == condition] 
                included_trial_ids = list(condition_chunk.metadata.trial_id.unique()[:trials_per_cond_per_chunk])
                condition_chunk_reduced = condition_chunk["trial_id in {}".format(included_trial_ids)]
                # TODO: this should be more than 8 for multiverse! --> test if correctly done in multiverse

                # Concatenate the condition-specific chunks
                if filtered_chunk is None:
                    filtered_chunk = condition_chunk_reduced
                else:
                    filtered_chunk = mne.concatenate_epochs([filtered_chunk, condition_chunk_reduced])

            #filtered_chunk.metadata["inner_chunk_id"] = i # should be redundant
            chunks.append(filtered_chunk)

    epochs_train = mne.concatenate_epochs(chunks)
    
    
    # if test file is a list, concatenate them
    if isinstance(test_file, list):
        for i, file in enumerate(test_file):
            if sfreq is None:
                chunk = mne.read_epochs(file, verbose=50)
            else:
                chunk = mne.read_epochs(file, verbose=50).resample(sfreq, npad="auto")
                
            chunk.metadata["inner_chunk_id"] = i # the chunk metadata is used for the inner cv loop (bayes opt), but the chunk i is not the actual chunk index of the raw file!
            chunks.append(chunk)
        epochs_test = mne.concatenate_epochs(chunks)
    elif test_file is not None:
        if sfreq is None:
            epochs_test = mne.read_epochs(test_file, verbose=50)
        else:
            epochs_test = mne.read_epochs(test_file, verbose=50).resample(sfreq, npad="auto")
    elif test_file is None:
        epochs_test = None
        
    good_picks = mne.pick_types(epochs_train.info, 
                            eeg=True, 
                            eog=False, 
                            exclude='bads')

    # if intra classification: discard half the data
    if "intra" in classification:
        if "human" in classification:
            epochs_train = epochs_train[["10001","10002"]]
            try:
                epochs_test = epochs_test[["10001","10002"]]
            except KeyError: # if the keys are not there
                epochs_test = None
            except TypeError: # if epochs_test already None
                epochs_test = None                
        elif "monkey" in classification:
            epochs_train = epochs_train[["10003","10004"]]
            try:
                epochs_test = epochs_test[["10003","10004"]]
            except KeyError:
                epochs_test = None
            except TypeError:
                epochs_test = None
                

    # extract the inner_chunk_ids before extracting the data from epochs object
    inner_chunk_ids = list(epochs_train.metadata["inner_chunk_id"])
    
    if not asEpoch:
        X_train = epochs_train.get_data(picks=good_picks,tmin=times[0],tmax=times[1])# DEBUG  [:, :, 2:-2]
        if epochs_test is not None:
            X_test = epochs_test.get_data(picks=good_picks,tmin=times[0],tmax=times[1])# DEBUG [:, :, 2:-2]
        else:
            X_test = None
        # reshape to 2D; n_trials x n_channels*n_timepoints
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if epochs_test is not None:
                X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train = epochs_train.copy()
        if epochs_test is not None:
            X_test = epochs_test.copy()
        else:
            X_test = None
        
    y_train = epochs_train.events[:, -1]
    if epochs_test is not None:
        y_test = epochs_test.events[:, -1]
    else:
        y_test = None
    
    y_train_orig = replace_labels_with_ints(y_train.copy(), classification="original") # just return all subclasses for separate evaluation
    y_train = replace_labels_with_ints(y_train, classification=classification)
    if epochs_test is not None:
        y_test = replace_labels_with_ints(y_test, classification=classification)
    else:
        y_test = None
        
    return X_train, y_train, X_test, y_test, inner_chunk_ids, epochs_train, y_train_orig #, train_weights, meta_slice, meta_str


def load_and_prepare_chunks_merge(train_files1,
                                 train_files2, 
                            test_file, # can also be multiple or single
                            classification='inter',
                            flatten=True,
                            asEpoch=False, # if it shall be returned as epoch obj. Then flatten must be false.
                            sfreq=None,
                            trials_per_cond_per_chunk='all', # 'all': all trials, int: only the first X trials per condition and chunk
                            times = (0, None)):  # if not otherwise specified, start at t=0 to the end of the epoch
    """ loads and processes epochs and returns training and testing data """
    
    chunks = []
    i=-1
    for file1, file2 in zip(train_files1, train_files2):
        i+=1
        if sfreq is None:
            chunk1 = mne.read_epochs(file1, verbose=50)
            chunk2 = mne.read_epochs(file2, verbose=50)
        else:
            chunk1 = mne.read_epochs(file1, verbose=50).resample(sfreq, npad="auto")
            chunk2 = mne.read_epochs(file2, verbose=50).resample(sfreq, npad="auto")
        chunk = mne.concatenate_epochs([chunk1, chunk2])
        
        chunk.metadata["inner_chunk_id"] = i # the chunk metadata is used for the inner cv loop (bayes opt), but the chunk i is not the actual chunk index of the raw file!

        if trials_per_cond_per_chunk == 'all':
            chunks.append(chunk)
        else:
            unique_conditions = np.unique(chunk.events[:, -1])  # Modify this line as per your data's event code structure

            filtered_chunk = None
            for condition in unique_conditions:
                # Select the first 8 trials for this condition
                condition_chunk = chunk[chunk.events[:, -1] == condition] 
                included_trial_ids = list(condition_chunk.metadata.trial_id.unique()[:trials_per_cond_per_chunk])
                condition_chunk_reduced = condition_chunk["trial_id in {}".format(included_trial_ids)]
                # TODO: this should be more than 8 for multiverse! --> test if correctly done in multiverse

                # Concatenate the condition-specific chunks
                if filtered_chunk is None:
                    filtered_chunk = condition_chunk_reduced
                else:
                    filtered_chunk = mne.concatenate_epochs([filtered_chunk, condition_chunk_reduced])

            #filtered_chunk.metadata["inner_chunk_id"] = i # should be redundant
            chunks.append(filtered_chunk)

    epochs_train = mne.concatenate_epochs(chunks)
    
    
    # if test file is a list, concatenate them
    if isinstance(test_file, list):
        for i, file in enumerate(test_file):
            if sfreq is None:
                chunk = mne.read_epochs(file, verbose=50)
            else:
                chunk = mne.read_epochs(file, verbose=50).resample(sfreq, npad="auto")
                
            chunk.metadata["inner_chunk_id"] = i # the chunk metadata is used for the inner cv loop (bayes opt), but the chunk i is not the actual chunk index of the raw file!
            chunks.append(chunk)
        epochs_test = mne.concatenate_epochs(chunks)
    elif test_file is not None:
        if sfreq is None:
            epochs_test = mne.read_epochs(test_file, verbose=50)
        else:
            epochs_test = mne.read_epochs(test_file, verbose=50).resample(sfreq, npad="auto")
    elif test_file is None:
        epochs_test = None
        
    good_picks = mne.pick_types(epochs_train.info, 
                            eeg=True, 
                            eog=False, 
                            exclude='bads')

    # if intra classification: discard half the data
    if "intra" in classification:
        if "human" in classification:
            epochs_train = epochs_train[["10001","10002"]]
            try:
                epochs_test = epochs_test[["10001","10002"]]
            except KeyError: # if the keys are not there
                epochs_test = None
            except TypeError: # if epochs_test already None
                epochs_test = None                
        elif "monkey" in classification:
            epochs_train = epochs_train[["10003","10004"]]
            try:
                epochs_test = epochs_test[["10003","10004"]]
            except KeyError:
                epochs_test = None
            except TypeError:
                epochs_test = None
                

    # extract the inner_chunk_ids before extracting the data from epochs object
    inner_chunk_ids = list(epochs_train.metadata["inner_chunk_id"])
    
    if not asEpoch:
        X_train = epochs_train.get_data(picks=good_picks,tmin=times[0],tmax=times[1])# DEBUG  [:, :, 2:-2]
        if epochs_test is not None:
            X_test = epochs_test.get_data(picks=good_picks,tmin=times[0],tmax=times[1])# DEBUG [:, :, 2:-2]
        else:
            X_test = None
        # reshape to 2D; n_trials x n_channels*n_timepoints
        if flatten:
            X_train = X_train.reshape(X_train.shape[0], -1)
            if epochs_test is not None:
                X_test = X_test.reshape(X_test.shape[0], -1)
    else:
        X_train = epochs_train.copy()
        if epochs_test is not None:
            X_test = epochs_test.copy()
        else:
            X_test = None
        
    y_train = epochs_train.events[:, -1]
    if epochs_test is not None:
        y_test = epochs_test.events[:, -1]
    else:
        y_test = None
    
    y_train_orig = replace_labels_with_ints(y_train.copy(), classification="original") # just return all subclasses for separate evaluation
    y_train = replace_labels_with_ints(y_train, classification=classification)
    if epochs_test is not None:
        y_test = replace_labels_with_ints(y_test, classification=classification)
    else:
        y_test = None
        
    return X_train, y_train, X_test, y_test, inner_chunk_ids, epochs_train, y_train_orig #, train_weights, meta_slice, meta_str


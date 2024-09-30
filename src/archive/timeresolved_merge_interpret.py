
# imports
import mne
import sys, os
from glob import glob
import numpy as np
import pandas as pd
if sys.platform.startswith('darwin'):
    mne.viz.set_browser_backend('qt', verbose=None) # 'qt' or 'matplotlib'
import seaborn as sns
sns.set()
import json
import pickle

# define base directory
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(base_dir)
os.chdir(base_dir)

from src.utils import *
from src.config import *
from src.ml import load_and_prepare_chunks, load_and_prepare_chunks_merge
from joblib import Parallel, delayed, dump
from multiprocessing import Manager

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from mne.decoding import (
    SlidingEstimator,
    cross_val_multiscore,
    CSP,
    LinearModel,
    get_coef,
)

with open('sessions.json', 'r') as file:
    data = json.load(file)
    sessions = data["allsessions_prediction"]

sessions = sorted(list(set([i[:7] for i in sessions])))
sessions = [i for i in sessions if "90" not in i] # 2 adult pilots


# TODO: just paralellize over subjects


def slider(X,y):
    #clf = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight='balanced'))
    clf = make_pipeline(StandardScaler(), 
                        LogisticRegression(solver="liblinear", class_weight='balanced'))
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="balanced_accuracy", verbose=True) # "roc_auc" balanced_accuracy
    scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=1)
    # Mean scores across cross-validation splits
    return np.mean(scores, axis=0)

def slider_interpret(X,y):
    #clf = make_pipeline(StandardScaler(), SVC(kernel="linear", class_weight='balanced'))
    clf = make_pipeline(StandardScaler(), 
                        LinearModel(LogisticRegression(solver="liblinear", class_weight='balanced')))
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="balanced_accuracy", verbose=True) # "roc_auc" balanced_accuracy
    
    #scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=1)
    
    time_decod.fit(X, y)
    
    # get coefs and save in epochs object
    coef = get_coef(time_decod, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    evoked_time_gen.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_patterns-ave.fif", overwrite=True)
    
    # plot        
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
    #f, ax = plt.subplots(figsize=(12, 4))  # Use plt.subplots() to ensure it links to the axes
    fig = evoked_time_gen.plot_joint(
        times=np.arange(-0.4, 0.800, 0.200), title=f"patterns {sub_ses_str} {species}", **joint_kwargs
    )
    fig.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_patterns.png")
    plt.close(fig)
    
    # TODO: if intra is analyzed, also save the single identities per species as ERP
    if species == "inter":

        evoked1 = epochs["10001", "10002"].average()
        evoked1.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_humans-ave.fif", overwrite=True)

        fig1 = evoked1.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg humans", **joint_kwargs
        )
        fig1.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_evoked_humans.png")
        
        evoked2 = epochs["10003", "10004"].average()
        evoked2.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_monkeys-ave.fif", overwrite=True)
        fig2 = evoked2.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg monkeys", **joint_kwargs
        )
        fig2.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_evoked_monkeys.png")
    
    elif species == "intra_human":
        evoked1 = epochs["10001"].average()
        evoked1.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_human1-ave.fif", overwrite=True)

        fig1 = evoked1.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg human 1", **joint_kwargs
        )
        fig1.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_evoked_human1.png")
        
        evoked2 = epochs["10002"].average()
        evoked2.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_human2-ave.fif", overwrite=True)
        fig2 = evoked2.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg human 2", **joint_kwargs
        )
        fig2.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_evoked_human2.png")
    
        
    elif species == "intra_monkey":
        evoked1 = epochs["10003"].average()
        evoked1.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_monkey1-ave.fif", overwrite=True)

        fig1 = evoked1.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg monkey 1", **joint_kwargs
        )
        fig1.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_evoked_monkey1.png")
        
        evoked2 = epochs["10004"].average()
        evoked2.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_monkey2-ave.fif", overwrite=True)
        fig2 = evoked2.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg monkey 2", **joint_kwargs
        )
        fig2.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret/{species}_evoked_monkey2.png")
    
                
        
    



    
train_i = [0,1,2,3,4]
test_i = 5 # dummy, or used as separate test set again (rest chunk)

for sub_ses_str in sessions:

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
            
    print(sub_ses_str)

    create_if_not_exist(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret")
    delete_files_in_folder(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_single_interpret", '*.pck') 

    # debug
    #species = "inter"
    for species in ['inter']: # , 'intra_human', 'intra_monkey' TODO
        
        """ subset prep """    
        min_epochs = 35

        """ load data """
            
        if len(sessions)==1:
            chunk_files_universe = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[0]}/250Hz_chunk*-epo.fif"))
            X, y, _, _, _, epochs = load_and_prepare_chunks(
                        train_files = [file for j, file in enumerate(chunk_files_universe) if j in train_i],  
                        test_file = None, #chunk_files_universe[test_i],
                        classification=species, #species, # there was an error, that's why the the results were so good before
                        flatten=False,
                        asEpoch=False, # True
                        #sfreq=sfreq,
                        trials_per_cond_per_chunk=int(max_epochs/5), # 8*5chunks = 40 # TODO: think about subsets or all?
                        times=(None, None),
                        )
        elif len(sessions)==2:
            chunk_files_universe1 = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[0]}/250Hz_chunk*-epo.fif"))
            chunk_files_universe2 = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[1]}/250Hz_chunk*-epo.fif"))
            #chunk_files_universe = chunk_files_universe1 + chunk_files_universe2

            X, y, _, _, _, epochs = load_and_prepare_chunks_merge(
                        train_files1 = [file for j, file in enumerate(chunk_files_universe1) if j in train_i],  
                        train_files2 = [file for j, file in enumerate(chunk_files_universe2) if j in train_i],  
                        test_file = None, #chunk_files_universe[test_i],
                        classification=species, #species, # there was an error, that's why the the results were so good before
                        flatten=False,
                        asEpoch=False, # True
                        #sfreq=sfreq,
                        trials_per_cond_per_chunk=int(max_epochs/5), # 8*5chunks = 40 # TODO: think about subsets or all?
                        times=(None, None),
                        )


        print(f"DEBUG: Context: {species}, Subset: {max_epochs}, X.shape: {X.shape}, y.shape: {y.shape}")
        
        """ one time resolved analysis"""


        
        slider_interpret(X.copy(),y.copy())



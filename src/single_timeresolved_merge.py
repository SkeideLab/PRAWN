
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
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    LinearModel,
    get_coef,
)

# DEBUG
#sub_ses_str = "sub-001"
#species="inter"

# define subject and session by arguments to this script
if len(sys.argv) not in [1,2]:
    print("Usage: python script.py sub_ses_str")
    sys.exit(1)
else:
    sub_ses_str = sys.argv[1]

sub = sub_ses_str[4:7]
#ses = sub_ses_str[-3:]
train_i = [0,1,2,3,4]
test_i = 5 # dummy, or used as separate test set again (rest chunk)



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

create_if_not_exist(f"{dirs['model_dir']}{sub_ses_str}/timeresolved")
create_if_not_exist(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret")
delete_files_in_folder(f"{dirs['model_dir']}{sub_ses_str}/timeresolved", '*.pck') 
delete_files_in_folder(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret", '*.fif') 
delete_files_in_folder(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret", '*.png') 

""" functions """

def slider(X,y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(solver="liblinear", class_weight='balanced'))
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="balanced_accuracy", verbose=True) # "roc_auc" balanced_accuracy
    scores = cross_val_multiscore(time_decod, X, y, cv=10, n_jobs=1)
    # Mean scores across cross-validation splits
    return np.mean(scores, axis=0)

def slider_permut(permut_scores): 
    """ permute labels, and then run slider """
    y_permut = np.random.permutation(y.copy())
    results = slider(X.copy(),y_permut)
        
    permut_scores.append(results)
    
        
def slider_interpret(X,y):
    clf = make_pipeline(StandardScaler(), 
                        LinearModel(LogisticRegression(solver="liblinear", class_weight='balanced')))
    time_decod = SlidingEstimator(clf, n_jobs=1, scoring="balanced_accuracy", verbose=True) # "roc_auc" balanced_accuracy
        
    time_decod.fit(X, y)
    
    # get coefs and save in epochs object
    coef = get_coef(time_decod, "patterns_", inverse_transform=True)
    evoked_time_gen = mne.EvokedArray(coef, epochs.info, tmin=epochs.times[0])
    evoked_time_gen.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_patterns-ave.fif", overwrite=True)
    
    # plot        
    joint_kwargs = dict(ts_args=dict(time_unit="s"), topomap_args=dict(time_unit="s"))
    #f, ax = plt.subplots(figsize=(12, 4))  # Use plt.subplots() to ensure it links to the axes
    fig = evoked_time_gen.plot_joint(
        times=np.arange(-0.4, 0.800, 0.200), title=f"patterns {sub_ses_str} {species}", **joint_kwargs
    )
    fig.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_patterns.png")
    plt.close(fig)
    
    # TODO: if intra is analyzed, also save the single identities per species as ERP
    if species == "inter":

        evoked1 = epochs["10001", "10002"].average()
        evoked1.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_humans-ave.fif", overwrite=True)

        fig1 = evoked1.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg humans", **joint_kwargs
        )
        fig1.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_evoked_humans.png")
        
        evoked2 = epochs["10003", "10004"].average()
        evoked2.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_monkeys-ave.fif", overwrite=True)
        fig2 = evoked2.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg monkeys", **joint_kwargs
        )
        fig2.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_evoked_monkeys.png")
    
    elif species == "intra_human":
        evoked1 = epochs["10001"].average()
        evoked1.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_human1-ave.fif", overwrite=True)

        fig1 = evoked1.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg human 1", **joint_kwargs
        )
        fig1.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_evoked_human1.png")
        
        evoked2 = epochs["10002"].average()
        evoked2.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_human2-ave.fif", overwrite=True)
        fig2 = evoked2.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg human 2", **joint_kwargs
        )
        fig2.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_evoked_human2.png")
    
        
    elif species == "intra_monkey":
        evoked1 = epochs["10003"].average()
        evoked1.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_monkey1-ave.fif", overwrite=True)

        fig1 = evoked1.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg monkey 1", **joint_kwargs
        )
        fig1.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_evoked_monkey1.png")
        
        evoked2 = epochs["10004"].average()
        evoked2.save(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_monkey2-ave.fif", overwrite=True)
        fig2 = evoked2.plot_joint(
            times=np.arange(-0.4, 0.800, 0.200), title=f"evoked {sub_ses_str} avg monkey 2", **joint_kwargs
        )
        fig2.savefig(f"{dirs['model_dir']}{sub_ses_str}/timeresolved_interpret/{species}_evoked_monkey2.png")
    
                

""" estimation """

for species in ['inter', 'intra_human', 'intra_monkey']:
    

    min_epochs = 35

    """ load data """

    if len(sessions)==1:
        chunk_files_universe = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[0]}/250Hz_chunk*-epo.fif"))
        X, y, _, _, _, epochs, _ = load_and_prepare_chunks(
                    train_files = [file for j, file in enumerate(chunk_files_universe) if j in train_i],  
                    test_file = None, #chunk_files_universe[test_i],
                    classification=species, #species, # there was an error, that's why the the results were so good before
                    flatten=False,
                    asEpoch=False, # True
                    #sfreq=sfreq,
                    trials_per_cond_per_chunk=int(max_epochs/5),
                    times=(None, None),
                    )
    elif len(sessions)==2:
        chunk_files_universe1 = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[0]}/250Hz_chunk*-epo.fif"))
        chunk_files_universe2 = sorted(glob(f"{dirs['processed_dir']}{sub_ses_str}_{sessions[1]}/250Hz_chunk*-epo.fif"))
        #chunk_files_universe = chunk_files_universe1 + chunk_files_universe2

        X, y, _, _, _, epochs, _ = load_and_prepare_chunks_merge(
                    train_files1 = [file for j, file in enumerate(chunk_files_universe1) if j in train_i],  
                    train_files2 = [file for j, file in enumerate(chunk_files_universe2) if j in train_i],  
                    test_file = None, #chunk_files_universe[test_i],
                    classification=species, #species, # there was an error, that's why the the results were so good before
                    flatten=False,
                    asEpoch=False, # True
                    #sfreq=sfreq,
                    trials_per_cond_per_chunk=int(max_epochs/5),
                    times=(None, None),
                    )


    print(f"DEBUG: Context: {species}, Subset: {max_epochs}, X.shape: {X.shape}, y.shape: {y.shape}")
    
    """ one time resolved analysis"""

    results = slider(X.copy(),y.copy())
    
    """ interpretation """
    
    if species == "inter":
        
        slider_interpret(X.copy(),y.copy())
    
    

    """ run permutations """
    manager = Manager() # shared list
    permut_scores = manager.list()
    Parallel(n_jobs=-1)(delayed(slider_permut)(permut_scores) for _ in range(1000)) # 200 * 5 cvs = 1000 fake models


    # Access the results in the shared list
    permutation_results = list(permut_scores)

    # concatenate the sublists vertically
    permutation_results = np.vstack(permutation_results)


    # write to file
    with open(f"{dirs['model_dir']}{sub_ses_str}/timeresolved/{max_epochs}_{species}_permutations.pck", 'wb') as f:
        pickle.dump(permutation_results, f)
    with open(f"{dirs['model_dir']}{sub_ses_str}/timeresolved/{max_epochs}_{species}_result.pck", 'wb') as f:
        pickle.dump(results, f)





import numpy as np
import pandas as pd
import pickle
import os
import sys
import glob
import json

base_dir = '/u/kroma/PRAWN/'

sys.path.append(base_dir)
os.chdir(base_dir)
from src.utils import *
from src.config import *
from src.ml import *
from significancefunctions import load_data

with open('sessions.json', 'r') as file:
    data = json.load(file)
    sessions = data["allsessions_prediction"]

# R1:
#sessions = sorted(list(set([i[:7] for i in sessions])))
sessions = [i for i in sessions if "90" not in i] # 2 adult pilots

times = np.linspace(-.4, 1., 351)


print(f"Total number of {len(sessions)} included. Before dropout")



##### TIME-RESOLVED ###########

df = []
failed_sessions = []
for species in ['inter', 'intra_human', 'intra_monkey']:
        
    for session in sessions:    
        
        try:
            # define subsets
            subsets = [] #stable_subset_numbers[species]
            files = sorted(glob(f"{dirs['model_dir']}{session}/timeresolved/*_{species}_result.pck"))
            subsets.append(np.max([int(os.path.basename(i).split('_')[0]) for i in files]))
            
            for subset in subsets:        
            
                # collate the results of each fp across all subsets
                result_file = f"{dirs['model_dir']}{session}/timeresolved/{subset}_{species}_result.pck"
                
                with open(result_file, "rb") as f:
                    results = pickle.load(f)
                    
                
                dfmean = pd.DataFrame({'accuracy': results, 
                                    'times': times,
                                    })
                dfmean['session'] = session
                dfmean['species'] = species
                dfmean['subset'] = str(subset) if subset == 35 else 'max'
                df.append(dfmean)
                
        except:
            failed_sessions.append(session)
            print(f"Failed {session}")

df = pd.concat(df)
#print(df.shape)
#df.head()


sessions = [i for i in sessions if i not in failed_sessions]
print(sessions)

# The failed sessions here are the participants with only 1 session,
# which were analyzed in sub-ABC folder instead of session-specific folders.
# This analyses only uses the one with 2 sessions!


# export for R stats
wide_together = []
for species in ["inter", "intra_human", "intra_monkey"]:

    data = df[(df.species==species) & (df.subset=="max")]
    
    wide_data = data.pivot(index='times', columns='session', values='accuracy')

    # Reset the index to make 'times' a regular column
    wide_data = wide_data.reset_index()
    times = wide_data['times']

    wide_data.drop(labels=['times'], axis=1, inplace = True)

    X = wide_data.values.T

    wide_data["species"] = species
    wide_together.append(wide_data)    
wide_together = pd.concat(wide_together)
wide_together.to_csv(f"{dirs['model_dir']}timeresolved_single.csv", index=False)

# R1 the sessions column/index column should be the timepoint column, but naming should be irrelevant as it is renamed later

###### EEGNET ###########
df = []
for session in sessions:
    for species in ["inter", "intra_human", "intra_monkey"]:
        subsets = [] #[stable_subset_numbers[species]]
        
        # NEW: find the highest subset for session
        files = sorted(glob(f"{dirs['model_dir']}{session}/braindecode/*_{species}_result.pck"))
        subsets.append(np.max([int(os.path.basename(i).split('_')[0]) for i in files]))
        
        for subset_version, subset in zip(['max'],subsets): #'35',
            #try:
            accuracy, accuracies_perm = load_data(session, f"{subset}_{species}", model=f'braindecode')

            # how many percent of the permutation accuracies are above the real accuracy?
            p = np.sum(accuracies_perm > accuracy) / len(accuracies_perm)

            # lower and upper limits of the permutation distributions
            # find the 0.05 and 0.95 quantiles of accuracies_perm
            lower = np.quantile(accuracies_perm, 0.05)
            upper = np.quantile(accuracies_perm, 0.95)
            
            df.append(pd.DataFrame({
                'session': [session],
                'context': [species],
                'subset': [subset_version],
                'accuracy': [accuracy],
                'p_uncorrected': [p],
                'll': [lower],
                'ul': [upper]
                }))
    #        except:
    #            pass

df = pd.concat(df)

# FDR correction will be done in R, then more flexible

df.to_csv(f"{dirs['model_dir']}eegnet_single.csv", index=False)


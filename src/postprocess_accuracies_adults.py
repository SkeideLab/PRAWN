
import numpy as np
import pandas as pd
import pickle
import os
import sys
import glob
import json
from scipy.stats import norm

base_dir = '/u/kroma/PRAWN/'

sys.path.append(base_dir)
os.chdir(base_dir)
from src.utils import *
from src.config import *
from src.ml import *
from significancefunctions import *

sessions = ["sub-901", "sub-902"]

times = np.linspace(-.4, 1., 351)

# precluster threshold (one-sided)
precluster_threshold = 0.05

# Find the Z-value corresponding to the given p-value (one-sided)
z_value = norm.ppf(1 - precluster_threshold)


print(f"Total number of {len(sessions)} included. Before dropout")


##### TIME-RESOLVED ###########

# debug
#session = sessions[0]
#species = 'inter'
#subset = subsets[0]

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
                dfmean['surrogate'] = False
                
                # R1: surrogate
                # load data
                accuracies, accuracies_perm = load_data(session, 
                                                        f"{subset}_{species}", 
                                                        model='timeresolved')

                # z transform data
                accuracies_z, accuracies_perm_z = z_transform_accuracies(accuracies.copy(), accuracies_perm.copy())

                # get permutation clusters
                max_clusters, max_clusters_95 = get_permutation_clusters(accuracies_perm_z, z_value)

                # get the clusters from the accuracies
                cluster_sum, clusters = get_clusters(accuracies_z, z_value)

                # get the average permutation timeseries
                average_permutation = np.mean(accuracies_perm, axis=0)

                # find the indices of clusters, which are above max_clusters_95
                significant_indices = [i for i, value in enumerate(cluster_sum) if value > max_clusters_95]

                significant_timepoints = [clusters[i] for i in range(len(clusters)) if i in significant_indices]
                significant_timepoints = [item for sublist in significant_timepoints for item in sublist]
                
                # dfmean['significant'] = True only for indices of significant time points, else False
                dfmean['significant'] = False
                dfmean.loc[significant_timepoints, 'significant'] = True
                
                df.append(dfmean)
                
        except:
            failed_sessions.append(session)
            print(f"Failed {session}")

df = pd.concat(df)
#print(df.shape)
#df.head()


sessions = [i for i in sessions if i not in failed_sessions]
print(sessions)

df.to_csv(f"{dirs['model_dir']}timeresolved_adults.csv", index=False)


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

df = pd.concat(df)

# FDR correction will be done in R, then more flexible

df.to_csv(f"{dirs['model_dir']}eegnet_adults.csv", index=False)

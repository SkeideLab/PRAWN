
import pickle
import numpy as np


def load_data(session, species, model='timeresolved_single'):
    # random floats, centered around 0.5, with a standard deviation of 0.1
    #accuracies = np.random.normal(0.6, 0.1, 251)
    with open(f'/ptmp/kroma/PRAWN/models/{session}/{model}/{species}_result.pck', 'rb') as f:
        accuracies = pickle.load(f, encoding='utf-8')

    # random floats, centered around 0.5, with a standard deviation of 0.1, but in form 1000, 251
    #accuracies_perm = np.random.normal(0.6, 0.1, (1000, 251))
    with open(f'/ptmp/kroma/PRAWN/models/{session}/{model}/{species}_permutations.pck', 'rb') as f:
        accuracies_perm = pickle.load(f, encoding='utf-8')

    return accuracies, accuracies_perm

def z_transform_accuracies(accuracies, accuracies_perm):
    # z transform the accuracies_perm for each column
    perm_mean = np.mean(accuracies_perm, axis=0)
    perm_std = np.std(accuracies_perm, axis=0)

    # z transform (column wise) the permutation accuracies
    accuracies_perm = (accuracies_perm - perm_mean) / perm_std

    # z transform (column wise) the real accuracies
    accuracies = (accuracies - perm_mean) / perm_std

    return accuracies, accuracies_perm

# function to get clusters, and their sum, from a time series
def get_clusters(ts, z_value):
    # z normalize ts
    #ts = (ts - np.mean(ts)) / np.std(ts)
    
    # find highest values, above z threshold
    #ts_above = ts[ts > z_value]
    
    # find indices of 5% highest values
    ts_above_idx = np.where(ts > z_value)
    
    # find clusters
    clusters = []
    cluster = []
    for idx in ts_above_idx[0]:
        if len(cluster) == 0:
            cluster.append(idx)
        elif idx == cluster[-1] + 1:
            cluster.append(idx)
        else:
            clusters.append(cluster)
            cluster = [idx]
    
    # for each cluster, summarize the z-values
    cluster_sum = []
    for cluster in clusters:
        cluster_sum.append(np.sum(ts[cluster]))

    return cluster_sum, clusters    


# loop through permutation_iterations
def get_permutation_clusters(accuracies_perm, z_value):
    max_clusters = []
    for i in range(1000):
        
        ts = accuracies_perm[i,:].copy()
        
        # get cluster sum
        cluster_sum, _ = get_clusters(ts, z_value)
        
        # write the biggest cluster sum to a list TODO, could also do with second biggest, and so on, to make it more liberal?
        if len(cluster_sum) > 0:
            max_cluster = np.max(cluster_sum)
        else:
            max_cluster = 0
        max_clusters.append(max_cluster)
        
    # find the 95% percentile of the max clusters
    max_clusters = np.array(max_clusters)
    max_clusters_95 = np.percentile(max_clusters, 95)

    return max_clusters, max_clusters_95
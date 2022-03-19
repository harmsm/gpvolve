
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import hdbscan

def dbscan_cluster(T, min_cluster_size=50, savefig=False):
    """
    This is a function to cluster genotypes using density-based
    spatial clustering of applications with noise (DBSCAN), which
    clusters based on density of points in an unsupervised method.
    The number of clusters does not need to be explicitly stated by
    the users. The only parameter that needs to be optimized is
    min_cluster_size, which is set to 50 here. But I recommend 1% of
    the len(data) Resulting plots are a bar chart showing the number
    of genotypes in each cluster and a heatmap of the median fitness
    value for each cluster.

    Note: clusters that are labeled '0' are genotypes that the
          DBSCAN could not cluster.

    Returns a dictionary of median fitness value represented in the
    heatmap.

    Parameters
    ----------
    T : probability transition matrix
    min_cluster_size : default = 50, needs to be optimized by user.
                       Typically needs to be 1% of len(data).
    savefig: Save generated bar chart showing the number of cells in
             each cluster and a heat map of the median fluorescence
             intensity in each channel for each cluster.
             Figure is saved using 'matplotlib' module.
    Returns
    -------
    final_dictionary : a tuple of two dictionaries.
                        The first dictionary is the
                        median fluorescence represented
                        in the heatmap while the second
                        dictionary holds all the
                        fluorescence vectors for each
                        cluster.
    See Also
    --------
    gaus_recluster, dip_test
    Examples
    --------
    measured = measure(sample, create_fcs=False)
    """
    # Create the clustering object
    cluster_obj = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)

    # Perform the clustering
    cluster_obj.fit(T)

    # Find the number of cells in each cluster
    cluster_counts = {}

    # clusters are
    for i in cluster_obj.labels_:
        if i not in cluster_counts.keys():
            cluster_counts[str(i + 1)] = list(cluster_obj.labels_).count(i)

    X = []

    # Make a 2d array of the vectors
    for index, row in T.iterrows():
        X.append([x for x in row])

    # Make a dictionary for our clusters to hold their associated vectors
    cluster_dict = {}
    for cluster_num in cluster_obj.labels_:
        if cluster_num not in cluster_dict.keys():
            cluster_dict[cluster_num] = []

    # Add the vector in each cluster
    for index, vector in enumerate(X):
        cluster_dict[cluster_obj.labels_[index]].append(vector)

    final_dictionary = {}

    # Make a new dictionary which will have the median value for each channel in the vector for a heatmap downstream
    for key, value in cluster_dict.items():
        median_values = []
        for i in range(len(value[0])):
            median_values.append(np.median([row[i] for row in value]))
            final_dictionary["Cluster " + str(key + 1)] = median_values

    df = pd.DataFrame(final_dictionary, index=list(T.columns))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(df.transpose(), cmap='copper')

    cluster_names = []
    count = []

    for key, value in cluster_counts.items():
        cluster_names.append(key)
        count.append(value)

    y_pos = np.arange(len(cluster_names))

    ax[0].bar(y_pos, count, color='black')
    ax[0].set_xticks(y_pos)
    ax[0].set_xticklabels(cluster_names)
    ax[0].set_xlabel('Cluster')
    ax[0].set_ylabel('Counts')
    ax[0].set_title('Cells per cluster')

    ax[1].set_title('Fluorescence profile of clusters')
    ax[1].set_xlabel('Fluorescence channel')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if savefig:
        plt.savefig("preliminary_clustering")

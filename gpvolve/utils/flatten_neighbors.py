__description__ = \
"""

"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

import gpmap

import numpy as np
import pandas as pd

def flatten_neighbors(gpm):
    """
    Flatten the neighbors in a GenotypePhenotypeMap into a 1D array. This is
    a useful form for neighbors in fast cython/c simluation methods.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap
        A GenotypePhenotypeMap with neighbors calculated.

    Returns
    -------
    neigbor_slicer : np.ndarray
        2D array of ints that is num_genotypes by 2. Indicates where to look for
        each genotype's neighbors in the 1D neighbors array. If
        neighbor_slicer[4,0] -> 6 and neighbor_slicer[4,1] -> 12, the neighbors
        for gentoype 4 are held in neighbors[6:12].
    neighbors : np.ndarray
        1D array of ints holding neighbors for each genotype flattened into a
        single vector of the form |genotype_0_neighbors|genotype_1_neighbors|...
    """
    # -------------------------------------------------------------------------
    # Parse arguments and validate sanity
    # -------------------------------------------------------------------------

    # Check gpm instance
    if not isinstance(gpm,gpmap.GenotypePhenotypeMap):
        err = "gpm must be a gpmap.GenotypePhenotypeMap instance\n"
        raise TypeError(err)

    # Look for gpm.data dataframe
    try:
        if not isinstance(gpm.data,pd.DataFrame):
            raise AttributeError
    except (AttributeError,TypeError):
        err = "gpm must have .data attribute that is a pandas DataFrame\n"
        raise ValueError(err)

    # Look for gpm.neighbors dataframe
    try:
        if not isinstance(gpm.neighbors,pd.DataFrame):
            raise AttributeError

        gpm.neighbors.loc[:,"source"]
        gpm.neighbors.loc[:,"target"]

    except (KeyError,AttributeError):
        err = "gpm must have .neighbors attribute that is a pandas\n"
        err += "DataFrame with source and target columns. Have you run\n"
        err += "gpm.get_neighbors()?\n"
        raise ValueError(err)

    # Structures for converting dataframe loc indexes to iloc indexes and vice
    # versa. gpm.neighbors stores edges with loc (to allow users to add and
    # remove rows), but contiguous iloc numbers will be much faster in numpy
    # and C. loc_to_iloc is a numpy array that effectively acts like a dict for
    # potentially non-contiguous loc indexes
    loc_to_iloc = -np.ones(np.max(gpm.data.index) + 1,dtype=int)
    loc_to_iloc[gpm.data.index] = np.arange(len(gpm.data.index))

    # Get number of genotypes
    num_genotypes = len(loc_to_iloc)

    non_self_neighbors_mask = gpm.neighbors.source != gpm.neighbors.target
    non_self_neighbors_mask = np.logical_and(keep_mask,
                                             non_self_neighbors_mask)
    num_total_neighbors = np.sum(non_self_neighbors_mask)

    # Sort edges by source, all in iloc indexes
    edges = np.zeros((num_total_neighbors,2),dtype=int)
    edges[:,0] = loc_to_iloc[gpm.neighbors.loc[non_self_neighbors_mask,"source"]]
    edges[:,1] = loc_to_iloc[gpm.neighbors.loc[non_self_neighbors_mask,"target"]]
    sorted_by_sources = np.argsort(edges[:,0])

    # List of all neighbor targets in a single, huge 1D array. This will act
    # as a jagged array, with neighbor_starts indicating where each source
    # starts in the array
    neighbors = edges[sorted_by_sources,1]

    # Where should we look for neighbors of genotype in neighbors array?
    genotypes_with_neighbors, start_indexes = np.unique(edges[sorted_by_sources,0],return_index=True)
    neighbor_slicer = -1*np.ones((num_genotypes,2),dtype=int)

    # Where to start looking for genotype's neighbors in neighbors array
    neighbor_slicer[genotypes_with_neighbors,0] = start_indexes

    # Where to stop looking for genotype's neighbors in neighbors array
    neighbor_slicer[genotypes_with_neighbors[:-1],1] = start_indexes[1:]
    neighbor_slicer[genotypes_with_neighbors[-1],1] = num_total_neighbors

    return neighbor_slicer, neighbors

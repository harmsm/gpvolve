__description__ = \
"""

"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

import gpmap
from gpvolve import check

import numpy as np
import pandas as pd

def flatten_neighbors(gpm):
    """
    Flatten the neighbors in a GenotypePhenotypeMap into a 1D array. This is
    a useful form for neighbors in fast cython/c simluation methods. All rows in
    gpm.neighbors with gpm.neighbors.include == True are returned.

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
        for gentoype 4 are held in neighbors[6:12]. neighbor_slicer is
        guaranteed to have an entry for every genotype, even those without any
        neighbors. If a genotype "i" has no neighbors, neighbor_slicer[i,[0,1]]
        == [-1,-1].
    neighbors : np.ndarray
        1D array of ints holding neighbors for each genotype flattened into a
        single vector of the form |genotype_0_neighbors|genotype_1_neighbors|...
    """
    # -------------------------------------------------------------------------
    # Parse arguments and validate sanity
    # -------------------------------------------------------------------------

    check.gpm_sanity(gpm)

    # Structures for converting dataframe loc indexes to iloc indexes and vice
    # versa. gpm.neighbors stores edges with loc (to allow users to add and
    # remove rows), but contiguous iloc numbers will be much faster in numpy
    # and C. loc_to_iloc is a numpy array that effectively acts like a dict for
    # potentially non-contiguous loc indexes
    loc_to_iloc = -np.ones(np.max(gpm.data.index) + 1,dtype=int)
    loc_to_iloc[gpm.data.index] = np.arange(len(gpm.data.index))

    # Get number of genotypes
    num_genotypes = len(loc_to_iloc)

    try:
        keep_mask = gpm.neighbors.loc[:,"include"]
    except KeyError:
        keep_mask = np.ones(len(gpm.neighbors),dtype=bool)

    # Num neighbors
    num_total_neighbors = np.sum(keep_mask)

    # Sort edges by source, all in iloc indexes
    edges = np.zeros((num_total_neighbors,2),dtype=int)
    edges[:,0] = loc_to_iloc[gpm.neighbors.loc[keep_mask,"source"]]
    edges[:,1] = loc_to_iloc[gpm.neighbors.loc[keep_mask,"target"]]
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
    if num_total_neighbors != 0:
        neighbor_slicer[genotypes_with_neighbors[-1],1] = num_total_neighbors

    return neighbor_slicer, neighbors

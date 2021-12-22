from gpvolve.cluster.criteria import *
import numpy as np


def optimize(T, criterion="Spectral gap", z=None):
    """
    Finding optimal number of clusters to use based on
    given criterion.

    Since the number of clusters nc is unknown in advance,
    the cluster algorithm is ran several times with different
    input parameters for nc, and then "best" solution is
    chosen based on chosen criteria (options are spectral
    gap, in

    Parameters
    ----------
    T : probability transition matrix calculated based on how many
        times a starting node i ended in j.
    criterion : criterion used to determine if clustering is optimal
                By default criterion is spectral gap.
                Options: spectral gap, optimality, and minchi
    z : maximum number of clusters to try. By default it's calculated
        as half of the size of transition matrix (i.e. half of the
        genotypes).

    Returns
    -------
    nc : optimal number of clusters according to given criteria
    """
    if z:
        pass
    else:
        z = int(np.shape(T)[0]/2)

    # Check criterion is a string of text
    assert isinstance(criterion,str)

    # Determine optimal number of clusters according to given criteria
    # Check if criterion chosen is spectral gap
    if criterion == "Spectral gap" or criterion == "spectral gap" or criterion == "Spectral Gap":
        nc = spectral_gap(T,z)
    # Check if criterion chosen is optimality
    elif criterion == "Optimality" or criterion == "optimality":
        nc = optimality(T,z)
    # Default to spectral gap is all else fails, but alert user
    else:
        print("Chosen criterion was not found, using spectral gap instead.")
        nc = spectral_gap(T,z)

    return nc

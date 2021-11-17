from gpvolve.cluster.utils import crispness
from gpvolve.utils import eigenvalues
from gpvolve.cluster import membership, cluster_trajectories
import numpy as np


def spectral_gap(T, z=None):
    """
    Find the spectral gap

    Parameters
    ----------
    T : probability transition matrix calculated based on how many
        times a starting node i ended in j.
    z : maximum number of clusters to try. By default it's calculated
        as half of the size of transition matrix (i.e. half of the
        genotypes).

    Returns
    -------
    nc : optimal number of clusters according to given criteria
    """
    # Check if max number of clusters to try is given
    # Calculate as half of size of T matrix if not given
    if z:
        pass
    else:
        z = int(np.shape(T)[0] / 2)

    # Get eigenvalues for given transition matrix
    eigv = eigenvalues(T)

    # Spectral gap for each n clusters in range(z)
    gaps = []

    # Spectral gap between lambda_nc and lambda_nc+1
    for i in range(z):
        lambda_nc_gap = abs(eigv[i] - eigv[i + 1])
        gaps.append(lambda_nc_gap)

    # Find the exact index of the max spectral gap found
    # That's the optimal number of clusters, nc.
    nc = int(np.where(gaps == np.max(gaps))[0])

    return nc


def optimality(T, z=None):
    """
    Find the optimality of solution by calculating the crispness
    of clustering for n number of clusters in range(z).

    Parameters
    ----------
    T : probability transition matrix calculated based on how many
        times a starting node i ended in j.
    z : maximum number of clusters to try. By default it's calculated
        as half of the size of transition matrix (i.e. half of the
        genotypes).

    Returns
    -------
    nc : optimal number of clusters according to given criteria
    """
    # Check if max number of clusters to try is given
    # Calculate as half of size of T matrix if not given
    if z:
        pass
    else:
        z = int(np.shape(T)[0] / 2)

    # Crispness of clustering for each n in range(1,z)
    # because it needs at least 2 clusters for calculation
    crisps = []
    for i in range(1, z):
        memberships = membership(T, i)
        clusters = cluster_trajectories(T, i)
        crisps.append(crispness(memberships, clusters))

    # Invert crispness because otherwise answer is always zero
    inv = [1 / i for i in crisps]

    # Find the exact index of the largest crispness value
    # That's the optimal number of clusters, nc.
    # Add 1 because range above start at 1.
    nc = int(np.where(inv == np.max(inv))[0]) + 1

    return nc



def minChi(T, z=None):
    """
    Finding minChi criterion value for each n in range(2,z) to
    obtain optimal number of clusters [1].
    Since minChi indicator is always 0 for nc=2, there is three
    possible cases here [2]:
        1.  2 < nc << len(genotypes)
        2. If minChi value is only zero for nc=2 and much larger
            for n>2, then use another method to determine optimal
            number of clusters.

    Parameters
    ----------
    T : probability transition matrix calculated based on how many
        times a starting node i ended in j.
    z : maximum number of clusters to try. By default it's calculated
        as half of the size of transition matrix (i.e. half of the
        genotypes).

    Returns
    -------
    nc : optimal number of clusters according to given criteria

    References
    ----------
    .. [1] Roeblitz, S and M Weber. 2013. Fuzzy spectral clustering by
        PCCA+: application to Markov state models and data
        classification. Advances in Data Analysis and Classification 7
        (2): 147-179
    .. [2] Weber M (2006) Meshless methods in conformation dynamics.
        Doctoral thesis, Department of Mathematics and Computer Science,
        Freie Universität Berlin. Verlag Dr. Hut, München
    """
    # Check if max number of clusters to try is given
    # Calculate as half of size of T matrix if not given
    if z:
        pass
    else:
        z = int(np.shape(T)[0] / 2)

    # NOTE: Interpretation here might be wrong
    minchis = []
    for i in range(2, z):
        memberships = membership(T, i)
        minchis.append(np.min(memberships))

    # Find the exact index of the minimum minChi value.
    # That's the optimal number of clusters, nc.
    # Add 2 because range above start at 2.
    nc = int(np.where(minchis == np.min(minchis))[0]) + 2

    return nc
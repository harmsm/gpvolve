from gpvolve.cluster.utils import cluster_assignments
from gpvolve.cluster.utils import cluster_sets
import msmtools.analysis as mana


def cluster(T, m):
    """
    Runs PCCA++ [1] to compute a metastable decomposition of MSM states.
    (i.e. find clusters using transition matrix and PCCA)

    Parameters
    ----------
    T: a probability transition matrix.
    m : Desired number of metastable sets (int).

    Notes
    -----
    The metastable decomposition is done using the PCCA method of the pyemma.msm.MSM class.
    For more details and references: https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/msm/models/msm.py
    """
    # Compute membership vectors.
    memberships = mana.pcca_memberships(T, m)
    assignments = cluster_assignments(memberships)
    clusters = cluster_sets(assignments)

    return clusters


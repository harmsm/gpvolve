from gpvolve.cluster.utils import cluster_assignments
from gpvolve.cluster.utils import cluster_sets
import msmtools.analysis as mana


def cluster_trajectories(gpm, T, m):
    # Find clusters using transition matrix and PCCA
    """Runs PCCA++ [1] to compute a metastable decomposition of MSM states.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object.
    T: a probability transition matrix.
    m : Desired number of metastable sets (int).

    Notes
    -----
    The metastable decomposition is done using the pcca method of the pyemma.msm.MSM class.
    For more details and references: https://github.com/markovmodel/PyEMMA/blob/devel/pyemma/msm/models/msm.py
    """
    # Compute membership vectors.
    memberships = mana.pcca_memberships(T, m)
    assignments = cluster_assignments(memberships)
    clusters = cluster_sets(assignments)

    return clusters


from gpvolve.check import gpm_sanity
from gpvolve.utils import flatten_neighbors
import numpy as np

def to_greedy(transition_matrix):
    """
    Turn transition matrix into 'greedy' transition matrix. Only the step with
    the highest positive fitness difference is allowed (prob. = 1), all other
    steps are not permitted (prob. = 0).
    Parameters
    ----------
    transition_matrix : 2D numpy.ndarray.
        Transition matrix where the highest value T(i->j) per row i should
        correspond to the step s(i->j) where j is the neighbor of genotype i
        with the highest fitness. Can be obtained using the 'ratio' fixation
        function, where transition probability T(i->j) is simply the ratio of
        fitness j over fitness i.
    Returns
    -------
    M : 2D numpy.ndarray.
        Transition matrix corresponding to a 'greedy random walk' on the
        genotype-phenotype map.
    References
    ----------
    de Visser JA, Krug J. 2014. Empirical fitness landscapes and the
    predictability of evolution. Nature Reviews Genetics 15:480–490.
    """
    T = transition_matrix.copy()
    # Remove self-looping probability/matrix diagonal = 0
    np.fill_diagonal(T, 0)

    # Get column index of max value for each row.
    indices = np.argmax(T, axis=1)
    # Set index pointer (check scipy.sparse.csr_matrix documentation).
    indptr = np.array(range(T.shape[0] + 1))
    # Since there is only on possible greedy step per row, it is assigned probability of 1.
    data = np.ones(T.shape[0])

    M = csr_matrix((data, indices, indptr), shape=T.shape).toarray()

    return M

__description__ = \
"""
Generate a transition matrix for a genotype phenotype map.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-19"

import warnings
from . import generate_tmatrix_python as py

try:
    from . import generate_tmatrix_cython as cy
    cy_available = True

except ImportError:
    cy_available = False
    w = "Could not find cython version of generate_tmatrix. Falling\n"
    w += "back on python version (same functionality, much slower)\n."
    warnings.warn(w)

import numpy as np

def generate_tmatrix(fitness,
                     neighbor_slicer,
                     neighbors,
                     population_size=1000,
                     fixation_model="moran",
                     use_cython=True):
    r"""
    Generate a stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    Parameters
    ----------
    fitness : numpy.ndarray
        num_genotypes-long float array containing fitness of each genotype
    neighbor_slicer : numpy.ndarray
        num_genotypes-long int array containing number of neighbors accessible
        for each genotype (excluding self)
    neighbors : numpy.ndarray
        1D numpy int array storing a jagged array with neighbors for each
        genotype. neighbor_slicer is used to look up where each genotype's
        neighbors are in this array
    population_size : int
        population size for fixation calculation. note that this parameter is
        ignored for the sswm model
    fixation_model : str
        model to use for calculating fixation probabilities. should be moran,
        mcclandish, or sswm (strong-selection, weak mutation).
    use_cython : bool
        use faster cython implementation if available.

    Returns
    -------
    T : nump.ndarray
        N x N row stochastic transition matrix

    Notes
    -----
    For a genotype $i$ with $n$ neighbors, we first calculate the probability
    of genotype $i$ mutating to genotype $j$ (instead of $k$, $l$, tc.). This i
    given by $P_{mutate\ i \rightarrow j} = 1/n$. We then calculate the
    probability that the mutation fixes using the fixation model
    ($P_{fix\ i\rightarrow j}$). In the transition matrix, we record the joint
    probability of mutation and fixation:
    $P_{ij} = P_{mutate\ i \rightarrow j} P_{fix\ i\rightarrow j}$. To
    get the self probability $P_{ii}$, we subtract the probability of all
    non-self moves: $P_{ii} = 1 - \sum_{j<n} P_{ij}$.
    """

    # Check sanity of fitness vector
    try:
        fitness = np.array(fitness,dtype=float)
        if np.sum(np.isnan(fitness)) != 0:
            err = "fitness vector contains NaN\n"
            raise ValueError(err)
        if np.sum(fitness < 0):
            err = "fitness vector contains values < 0\n"
            raise ValueError(err)

    except TypeError:
        err = "fitness vector must contain float values\n"
        raise TypeError(err)

    try:
        # Make sure fitness and neighbors have same length
        if fitness.shape[0] != neighbor_slicer.shape[0]:
            raise ValueError
    except (AttributeError,IndexError):
        err = "fitness and neighbor_slicer must be numpy arrays of the same\n"
        err += "length. (neighors and neighbor_slicer should generally be\n"
        err += "made with the flatten_neighbors function).\n"
        raise ValueError(err)

    # Make sure population size is an integer of 1 or more
    try:
        population_size = int(population_size)
        if population_size < 1:
            raise ValueError

    except (TypeError,ValueError):
        err = f"population size must be an int >= 1.\n"
        raise ValueError(err)

    # Fixation model (external c function; see f_type def above)
    fixation_models = ["moran","mcclandish","sswm"]
    if fixation_model not in fixation_models:
        err = f"fixation model '{fixation_model}' not recognized.\n"
        err += f"Should be one of {','.join(fixation_models)}.\n"
        raise ValueError(err)

    # Figure out which implementation to use
    if use_cython and not cy_available:
        w = "Could not find cython version of generate_tmatrix. Falling\n"
        w += "back on python version (same functionality, much slower)\n."
        warnings.warn(w)

    if use_cython:
        return cy.generate_tmatrix_cython(fitness,
                                          neighbor_slicer,
                                          neighbors,
                                          population_size,
                                          fixation_model)
    else:
        return py.generate_tmatrix_python(fitness,
                                          neighbor_slicer,
                                          neighbors,
                                          population_size,
                                          fixation_model)

from gpvolve.check import gpm_sanity
from gpvolve.utils import flatten_neighbors
from gpvolve.markov import base
from gpvolve.phenotype_to_fitness import linear, sigmoid, exponential, step
import numpy as np


def apply_fitness_function(gpm, fitness_function, **params):
    """
    Compute fitness values from a user-defined phenotype-fitness function.
    A few basic functions can be found in gpvolve.phenotype_to_fitness. For a direct
    mapping of phenotype to fitness, use one_to_one without additional
    parameters.

    Parameters
    ----------
    gpm : a GenotypePhenotypeMap object
    fitness_function: function.
        A python function that takes phenotypes and additional parameters
        (optional) and returns a list of fitness values (type=float).

    Returns
    -------
    Nothing: None
        The computed fitness values are automatically stored in underlying
        dataframe of gpm object, (i.e. gpm.data.loc[:, "fitness"]).
    """
    # Get neighbors and check minimum requirements
    gpm.get_neighbors()
    gpm_sanity(gpm)

    # Calculate fitness parameters
    fitness = fitness_function(gpm.data.loc[:, "phenotype"], **params)

    # Add fitness values column to gpm.data pandas data frame.
    gpm.data.loc[:, "fitness"] = fitness

    return gpm


def build_transition_matrix(gpm, fixation_model='moran', **params):
    """
    Calculate fixation probability along all edges and build transition
    matrix. Tries to use cython first, and falls back to python if needed.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object
    fixation_model : Fixation model used for transition matrix.
    fitness_model : If fitness values are not present, generate them using a given
                    model (options are linear, sigmoid, exponential, and step).
    Returns
    -------
    T : Stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    """
    # Get neighbors and check minimum requirements
    gpm.get_neighbors()
    gpm_sanity(gpm)

    # Check a fitness column is present in gpm.data dataframe
    if 'fitness' in gpm.data.columns:
        pass
    else:
        err = f"fitness column not in gpm.data\n"
        err += "dataframe\n"
        raise KeyError(err)

    # Convert fitness to an array
    fitness = np.array(gpm.fitness)

    # Get flat neighbors
    neighbor_slicer, neighbors = flatten_neighbors(gpm)

    try:
        use_cython = True
        t = base.generate_tmatrix(fitness,
                                  neighbor_slicer,
                                  neighbors,
                                  fixation_model=fixation_model,
                                  use_cython=use_cython,
                                  **params)
    except ImportError:
        use_cython = False
        t = base.generate_tmatrix(fitness,
                                  neighbor_slicer,
                                  neighbors,
                                  fixation_model=fixation_model,
                                  use_cython=use_cython,
                                  **params)

    return t



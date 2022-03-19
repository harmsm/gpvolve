
from gpvolve.utils import flatten_neighbors
from ._generate_tmatrix import generate_tmatrix

import numpy as np

from gpvolve.check import gpm_sanity

def build_transition_matrix(gpm, fitness_column="fitness", fixation_model="moran", population_size=1000):
    """
    Calculate fixation probability along all edges and build transition
    matrix. Tries to use cython first, and falls back to python if needed.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object
    fitness_column : str
        column in gpm.data that has fitness value for each genotype
    fixation_model : str
        model to use for calculating fixation probabilities. should be moran,
        mcclandish, or sswm (strong-selection, weak mutation).
    population_size : int
        population size for fixation calculation. note that this parameter is
        ignored for the sswm model

    Returns
    -------
    T : Stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    """
    # Get neighbors and check minimum requirements
    gpm.get_neighbors()
    gpm_sanity(gpm)

    # Check population_size
    try:
        population_size = int(population_size)
        if population_size < 1:
            raise ValueError
    except (ValueError,TypeError):
        err = f"population_size must be an integer > 0.\n"
        raise ValueError(err)

    # Get fitness data
    try:
        fitness = np.array(gpm.data.loc[:,fitness_column],dtype=float)
        if np.min(fitness) < 0:
            raise ValueError
        if np.sum(np.isnan(fitness)) > 0:
            raise ValueError
    except KeyError:
        err = f"fitness_column '{fitness_column}' not in gpm.data\n"
        err += "dataframe\n"
        raise KeyError(err)
    except (TypeError,ValueError):
        err = "fitness_column must point to a column in gpm.data that can\n"
        err += "be coerced as a float, where the minimum is >= 0 and that does\n"
        err += "not have nan.\n"
        raise ValueError(err)

    # Get flat neighbors
    neighbor_slicer, neighbors = flatten_neighbors(gpm)

    t = generate_tmatrix(fitness,
                         neighbor_slicer,
                         neighbors,
                         fixation_model=fixation_model,
                         population_size=population_size)

    return t

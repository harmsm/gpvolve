__description__ = \
"""
Function for applying an exponential transform to phenotype data, yielding
fitness.
"""

import numpy as np


def exponential(phenotypes, a=1,b=0,k=1):
    r"""
    Apply exponential fitness function to phenotypes. Applies model:

    $$F = a \times P^{k} + b$$

    Parameters
    ----------
    phenotypes : 1D numpy.ndarray.
        Phenotype values (dtype=float).
    a,b,k : float,float,float
        transform parameters.

    Returns
    -------
    fitnesses : 1D numpy.ndarray.
        List of fitnesses.
    """

    

    # Raising 0 phenotypes to a power will result in NaN, hence we change any NaN to 0.
    fitnesses = np.nan_to_num(fitnesses ** exponent)

    return fitnesses

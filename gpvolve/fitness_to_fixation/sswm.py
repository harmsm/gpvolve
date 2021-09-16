__description__ = \
"""
Strong selection, weak mutation fixation model from Gillespie 1984.
"""
__author__ = "Zach Sailer, Michael J. Harms"
__date__ = "2021-09-01"

import numpy as np

def sswm(fitness_i, fitness_j):
    r"""
    Strong selection, weak mutation model. From Gillespie 1984.

    Parameters
    ----------
    fitness_i : float
        fitness of the source genotype (wildtype). Must be greater than zero.
    fitness_j : float
        fitness of the target genotype (mutant). Must be greater than zero.

    Returns
    -------
    fixation_probability : float

    Notes
    -----
    + Fixation probability given by:
        $$s_{ij} = \frac{f_{j} - f_{i}}{f_{i}}$$
        $$\pi_{i \rightarrow j} = 1 - e^{s_{ij}}$$

    + Gives real valued answers for all finite inputs of fitness_i and fitness_j.
    """

    try:
        if float(fitness_i) <= 0 or float(fitness_j) <= 0:
            raise TypeError
    except TypeError:
        err = "fitness values must be > 0\n"
        raise ValueError(err)

    maxexp = np.finfo(float).maxexp

    a = fitness_j - fitness_i
    if a <= 0:
        return 0.0
    else:
        ratio = np.log2(a) - np.log2(fitness_i)
        if ratio > maxexp:
            return 1.0
        sij = np.power(2,ratio)

    return 1 - np.exp(-sij)

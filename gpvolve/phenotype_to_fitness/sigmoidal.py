import numpy as np


def sigmoid(P, a=0, b=1):
    """
    Apply sigmoid fitness function to phenotypes. Applies model:

    $$F = a* \frac{1}{1+e^{-P}} + b$$

    Parameters
    ----------
    phenotypes : 1D numpy.ndarray.
        Phenotype values (dtype=float).
    a,b : float,float,float
        transform parameters.

    Returns
    -------
    fitness : 1D numpy.ndarray.
        List of fitness values.
    """

    fitness = a*(1/(1+np.exp(-P)))+b

    return fitness


def linear(phenotypes, a=1, b=1):
    r"""
    Apply linear fitness function to phenotypes. Applies model:

    $$F = a*P + b$$

    Parameters
    ----------
    phenotypes : 1D numpy.ndarray.
        Phenotype values (dtype=float).
    a,b: float,float
        transform parameters.

    Returns
    -------
    fitness : 1D numpy.ndarray.
        List of fitness values.
    """

    # Raising 0 phenotypes to a power will result in NaN, hence we change any NaN to 0.
    # Calculating fitness
    fitness = a * phenotypes + b

    return fitness

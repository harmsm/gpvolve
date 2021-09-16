__description__ = \
"""
Calculate fixation probability using model described by Sella & Hirsch 2005.
"""
__author__ = "Zach Sailer, Michael J. Harms"
__date__ = "2021-09-01"

import numpy as np

def moran(fitness_i, fitness_j, population_size):
    r"""
    Calculate fixation probability using moran model proposed by Sella and
    Hirsch, 2005.

    Parameters
    ----------
    fitness_i : float
        fitness of the source genotype (wildtype). Must be greater than zero.
    fitness_j : float
        fitness of the target genotype (mutant). Must be greater than zero.
    population_size : number
        population size (must be 1 or more.)

    Returns
    -------
    fixation_probability : float

    Notes
    -----
    + Fixation probability given by:

    $$\pi_{i \rightarrow j} = \frac{1 - \Big ( \frac{f_{i}}{f_{j}} \Big ) }{1 - \Big ( \frac{f_{i}}{f_{j}} \Big )^{N} }$$

    + Function gives real valued answers for all finite inputs of fitness_i,
      fitness_j, and population_size.

    References
    ----------
    G. Sella, A. E. Hirsh: The application of statistical physics to evolutionary biology, Proceedings of the National
    Academy of Sciences Jul 2005, 102 (27) 9541-9546.
    """

    try:
        if float(fitness_i) <= 0 or float(fitness_j) <= 0 or float(population_size) < 1:
            raise TypeError
    except TypeError:
        err = "fitness values must be > 0 and population_size must be >= 1\n"
        raise ValueError(err)

    # Will be 1.0 for population size of 1, regardless of fitness difference
    if population_size == 1:
        return 1.0

    maxexp = np.finfo(float).maxexp

    # If fitness is identical, generate parameter set with fitness_i slightly
    # smaller and fitness_j slightly smaller. Since these are all ratios, this
    # will sample just above and just below the infinity of fi == fj. If the
    # fitness are not same, just record them both as param sets. Do the
    # calculation for all params in param_sets and take the mean at the end.
    if fitness_i == fitness_j:
        param_sets = [(fitness_i*0.99999,fitness_j),(fitness_i,fitness_j*0.99999)]
    else:
        param_sets = [(fitness_i,fitness_j)]

    results = []
    for fi,fj in param_sets:

        a = np.log2(fi) - np.log2(fj)

        # If |a|*N is *huge* we can't store their multiple.
        if np.log2(np.abs(a)) + np.log2(population_size) > maxexp:

            # (1 - 2^a)/(1-2^(a*N)).
            # If 2^a is really big compared to 1, this reduces to:
            #      -2^a/-2^(a*N) --> 1/2^N.
            # If 2^N is really big compared to 2^a, this reduces to ...
            #       1/2^N.
            # If 2^N is bigger than we can calculate, just return 0.
            if a > 0:
                if population_size > maxexp:
                    results.append(0.0)
                else:
                    results.append(1/(np.power(2,population_size)))
                continue

            # If a < 0...
            # We know that |a|*N is HUGE, so denominator (1-2^N*2^a --> 1.0).
            else:
                if -a > maxexp:
                    results.append(1.0)
                else:
                    results.append(1 - np.power(2,a))
                continue


        b = population_size*a

        # Too big to do calculation explicitly, but that's okay. We need to do
        # (1 - 2^a)/(1 - 2^b), but we now know that 2^a and 2^b are >> 1. So this
        # reduces to -2^a/-2^b, which simplifies to 2^(a-b). If b is big it must
        # be positive.  Since it is larger than a by a factor of N, b-a will positive, it
        # is larger than a by a factor of N, so this will lead to a safe negative
        # number going into power.

        if b > maxexp:
            results.append(np.power(2,a-b))
            continue
        else:

            num = 1 - np.power(2,a)
            den = 1 - np.power(2,b)

            results.append(num/den)
            continue

    return np.mean(results)

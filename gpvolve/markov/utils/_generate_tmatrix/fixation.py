__description__ = \
"""
Calculate fixation probabilites using a variety of population genetic models.
"""
__author__ = "Zach Sailer, Michael J. Harms"
__date__ = "2021-09-01"

import numpy as np

def mcclandish(fitness_i, fitness_j, population_size):
    r"""
    Calculate fixation probability using model proposed by McClandish, 2011.

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

    $$\pi_{i \rightarrow j} = \frac{1-e^{-2(f_{j}-f_{i})}}{1-e^{-2N(f_{j}-f_{i})}}$$

    + Function gives real valued answers for all finite inputs of fitness_i,
      fitness_j, and population_size.

    References
    ----------
    McCandlish, D. M. (2011), VISUALIZING FITNESS LANDSCAPES. Evolution, 65: 1544-1558.
    """

    try:
        if float(fitness_i) <= 0 or float(fitness_j) <= 0 or float(population_size) < 1:
            raise TypeError
    except TypeError:
        err = err = "fitness values must be > 0 and population_size must be >= 1\n"
        raise ValueError(err)

    # Will be 1.0 for population size of 1, regardless of fitness difference
    if population_size == 1:
        return 1.0

    maxexp = np.finfo(float).maxexp
    power_coeff = -2*np.log2(np.e)
    l2_power_coeff = np.log2(2*np.log2(np.e))

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

        # Will not hit overflow because each bounded by (0,max]
        a = fj - fi

        # Try to calcualate
        unable_to_calculate = 4
        neg2a, exp_neg2a = None, None
        neg2aN, exp_neg2aN = None, None

        # Can we do -2*(fj - fi) and exp(-2*(fj - fi))?
        if (np.log2(np.abs(a)) + l2_power_coeff) <= maxexp:
            neg2a = a*power_coeff
            unable_to_calculate -= 1
            if neg2a <= maxexp:
                exp_neg2a = np.power(2,neg2a)
                unable_to_calculate -= 1

        # Can we do -2*(fj - fi)*N and exp(-2*(fj - fi)*N)?
        if (np.log2(np.abs(a)) + np.log2(population_size) + l2_power_coeff) <= maxexp:
            neg2aN = a*population_size*power_coeff
            unable_to_calculate -= 1
            if neg2aN <= maxexp:
                exp_neg2aN = np.power(2,neg2aN)
                unable_to_calculate -= 1

        # Something was too big to calculate
        if unable_to_calculate > 0:

            # If a is positive (and knowning N is always positive),
            # exp(-2*a*N) -> 0 and the denominator (1 - exp(-2*a*N)) --> 1.
            if a > 0:

                # If exp(-2*a) is overflowing when a is positive, numerator --> 1.0.
                # Otherwise calcualte numerator exactly.
                if exp_neg2a is None:
                    results.append(1.0)
                else:
                    results.append(1 - exp_neg2a)

            # If a is negative...
            elif a < 0:

                # If a is super negative, exp(-2a) becomes large and positive
                # and dominates the 1 in front. This means the equation becomes
                # exp(-2*a)/exp(-2*a*N) which simplifies to 1/exp(-2*a*(N-1)).

                # If we got an overflow in the exp(-2aN), the denominator is
                # huge and this --> 0.

                # Huge negative selection...
                if exp_neg2a is None:
                    # With a huge population ...
                    if exp_neg2aN is None:
                        # No fixation.
                        results.append(0.0)

                    # If we're here, we have huge negative selection with a
                    # tiny population size. This basically won't happen unless
                    # we allow population size < 1.0. Assert no fixation
                    # because scenario is so outlandish.
                    else:
                        results.append(0.0)

                # If exp_neg2a is finite, the overflow had to come in the
                # denominator.  This means denominator is huge and value --> 0.
                else:
                    results.append(0.0)

            # This should not be possible because we ensured fi != fj above.
            else:
                e = "It should not be possible to have difference in fitness\n"
                e += "of zero... Something is dreadfully wrong."
                raise RuntimeError(e)

        else:
            numerator = 1 - exp_neg2a
            denominator = 1 - exp_neg2aN
            if numerator == 0:
                results.append(0)
            else:
                results.append(numerator/denominator)

    return np.mean(results)


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
        param_sets = [(fitness_i*0.9999999999,fitness_j),(fitness_i,fitness_j*0.9999999999  )]
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


def sswm(fitness_i, fitness_j, population_size):
    r"""
    Strong selection, weak mutation model. From Gillespie 1984.

    Parameters
    ----------
    fitness_i : float
        fitness of the source genotype (wildtype). Must be greater than zero.
    fitness_j : float
        fitness of the target genotype (mutant). Must be greater than zero.
    population_size : int
        parameter is ignored (here to keep compatiability with other fixation
        models).

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

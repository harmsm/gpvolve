__description__ = \
"""
Python implementation for generating row-stochastic transition matrix from
fitness and neighbor data given some fixation model.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

from .fixation import moran, mcclandish, sswm
import numpy as np

def generate_tmatrix_python(fitness,
                            neighbor_slicer,
                            neighbors,
                            population_size,
                            fixation_model):

    """
    Generate a stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    This function should not be called directly. Instead, use generate_tmatrix
    wrapper. Wrapper has argument docs and does argument sanity checking.
    """

    # Get fixation model
    fixation_models = {"moran":moran,"mcclandish":mcclandish,"sswm":sswm}
    fixation_model = fixation_models[fixation_model]

    # Get number of genotypes
    num_genotypes = len(fitness)

    # Create output transition matrix
    T = np.zeros((num_genotypes,num_genotypes),dtype=float)

    # Go through every genotype
    for i in range(num_genotypes):

        # Go through its neighbors. If no neighbors, set prob staying self to
        # 1.0.
        num_neighbors = neighbor_slicer[i,1] - neighbor_slicer[i,0]
        if num_neighbors == 0:
            T_view[i,i] = 1.0
        else:
            Pi_out = 0.0
            for j in range(num_neighbors):

                j_n = neighbors[neighbor_slicer[i,0] + j]

                # Calculate fixation probability for i -> j
                Pij_fix = fixation_model(fitness[i],fitness[j_n],population_size)

                # Pij is Pmutate * Pfix = 1/n*Pfix
                T[i,j_n] = Pij_fix/num_neighbors
                Pi_out += T[i,j_n]

            # Probability of remaining is 1 - total probability of leaving.
            T[i,i] = 1 - Pi_out

    return T

def _moran_tester(f1,f2,N):
    """
    Wrap moran to allow pytest to access. Not generally used by users.
    """
    return moran(f1,f2,N)

def _mcclandish_tester(f1,f2,N):
    """
    Wrap mcclandish to allow pytest to access. Not generally used by users.
    """

    return mcclandish(f1,f2,N)

def _sswm_tester(f1,f2,N):
    """
    Wrap sswm to allow pytest to access. Not generally used by users.
    """

    return sswm(f1,f2,N)

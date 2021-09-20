#cython: language_level=3
__description__ = \
"""
Cython implementation for generating row-stochastic transition matrix from
fitness and neighbor data given some fixation model.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

# Load in external C code for fixation functions
cdef extern from "fixation.h":
    double moran(double fitness_i, double fitness_j, long population_size)
    double mcclandish(double fitness_i, double fitness_j, long population_size)
    double sswm(double fitness_i, double fitness_j, long population_size)

import numpy as np
cimport numpy as cnp
cimport cython

from cpython.pycapsule cimport PyCapsule_GetPointer
from cython.cimports.cpython.mem import PyMem_Free

# Make a type for passing functions
ctypedef double (*f_type)(double, double, long)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Deactivate divide-by-zero checking
def generate_tmatrix_cython(fitness,
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

    cdef long population_size_int = <long>population_size

    # Fixation model (external c function; see f_type def above)
    cdef f_type fixation_model_ftype
    if fixation_model == "moran":
        fixation_model_ftype = <f_type>moran
    elif fixation_model == "mcclandish":
        fixation_model_ftype = <f_type>mcclandish
    elif fixation_model == "sswm":
        fixation_model_ftype = <f_type>sswm
    else:
        err = f"fixation model '{fixation_model}' not recognized.\n"
        err += "Should be one of 'moran', 'mcclandish', or 'sswm'.\n"
        raise ValueError(err)

    # Get number of genotypes
    num_genotypes = len(fitness)
    cdef long num_genotypes_int = <long>num_genotypes

    # Create output transition matrix
    T = np.zeros((num_genotypes,num_genotypes),dtype=float)

    # Sundry counters
    cdef int i, j, j_n
    cdef int num_neighbors
    cdef double Pi_out, Pij_fix

    # efficient views of numpy arrays
    cdef long[:] neighbors_view = neighbors
    cdef long[:,:] neighbor_slicer_view = neighbor_slicer
    cdef double[:] fitness_view = fitness
    cdef double[:,:] T_view = T

    # Go through every genotype
    for i in range(num_genotypes_int):

        # Go through its neighbors. If no neighbors, set prob staying self to
        # 1.0.
        num_neighbors = neighbor_slicer_view[i,1] - neighbor_slicer_view[i,0]
        if num_neighbors == 0:
            T_view[i,i] = 1.0
        else:
            Pi_out = 0.0
            for j in range(num_neighbors):

                j_n = neighbors[neighbor_slicer[i,0] + j]

                # Calculate fixation probability for i -> j
                Pij_fix = fixation_model_ftype(fitness_view[i],
                                               fitness_view[j_n],
                                               population_size_int)

                # Pij is Pmutate * Pfix = 1/n*Pfix
                T_view[i,j_n] = Pij_fix/num_neighbors
                Pi_out += T_view[i,j_n]

            # Probability of remaining is 1 - total probability of leaving.
            T_view[i,i] = 1.0 - Pi_out

    return T

def _moran_tester(f1,f2,N):
    """
    Wrap moran C-implementation to allow pytest to access. Not generally used
    by users.
    """

    cdef double f1_dbl = <double>f1
    cdef double f2_dbl = <double>f2
    cdef long N_long = <long>N

    return moran(f1_dbl,f2_dbl,N_long)

def _mcclandish_tester(f1,f2,N):
    """
    Wrap mcclandish C-implementation to allow pytest to access. Not generally used
    by users.
    """

    cdef double f1_dbl = <double>f1
    cdef double f2_dbl = <double>f2
    cdef long N_long = <long>N

    return mcclandish(f1_dbl,f2_dbl,N_long)

def _sswm_tester(f1,f2,N):
    """
    Wrap sswm C-implementation to allow pytest to access. Not generally used
    by users.
    """

    cdef double f1_dbl = <double>f1
    cdef double f2_dbl = <double>f2
    cdef long N_long = <long>N

    return sswm(f1_dbl,f2_dbl,N_long)

#cython: language_level=3
__description__ = \
"""
Cython implementation of Wright Fisher simulation.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

import numpy as np
cimport numpy
cimport cython

from cpython.pycapsule cimport PyCapsule_GetPointer
from cython.cimports.cpython.mem import PyMem_Free

from numpy.random cimport bitgen_t
from numpy.random import PCG64
from numpy.random.c_distributions cimport (random_standard_uniform,
                                           random_bounded_uint64,
                                           random_poisson)



@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True) # Deactivate divide-by-zero checking
def wf_engine_cython(pops,
                     mutation_rate,
                     fitness,
                     neighbor_slicer,
                     neighbors):
    """
    A cython implementation of the Wright Fisher engine.

    This function should not be called directly. Instead, use wf_engine
    wrapper. Wrapper has argument docs and does argument sanity checking.
    """

    cdef double denominator
    cdef int i, j, k, num_populated_genotypes, min_index, max_index
    cdef int num_to_mutate_int

    # Get number of genoptypes, population size, and number of steps
    num_genotypes = len(fitness)
    pop_size = sum(pops[0,:])
    num_steps = len(pops)

    # If zero population, don't bother with simulation
    if pop_size == 0:
        return pops

    # Create c-versions of the simulation parameters
    cdef int num_steps_int = <int>num_steps
    cdef int num_genotypes_int = <int>num_genotypes
    cdef int pop_size_int = <int>pop_size
    cdef double pop_size_dbl = <double>pop_size
    cdef double expected_num_mutations = <double>(pop_size_dbl*mutation_rate)

    # efficient views of numy arrays
    cdef double[:] fitness_view = fitness
    cdef long[:,:] neighbor_slicer_view = neighbor_slicer
    cdef long[:] neighbors_view = neighbors
    cdef long[:,:] pops_view = pops

    # Initialize random number generator
    cdef bitgen_t *bitgen_state
    bitgen_state = <bitgen_t *>PyCapsule_GetPointer(PCG64().capsule,
                                                    "BitGenerator")

    # Some temporary vectors
    choice_vector = np.zeros(num_genotypes,dtype=int)
    prob_vector = np.zeros(num_genotypes,dtype=float)
    cdef long[:] choice_vector_view = choice_vector
    cdef double[:] prob_vector_view = prob_vector

    for i in range(1,num_steps_int,1):

        denominator = 0;
        k = 0;
        for j in range(0,num_genotypes_int,1):

            if pops_view[i-1,j] >  0:
                choice_vector_view[k] = j
                prob_vector_view[k] = fitness_view[j]*pops_view[i-1,j]
                denominator = denominator + prob_vector_view[k]
                k += 1

        # These are how many genotypes were populated
        num_populated_genotypes = k;

        if abs(denominator) < 1E-20:
            for j in range(0,num_populated_genotypes,1):
                prob_vector_view[j] = pops_view[i-1,choice_vector_view[j]]/pop_size_dbl
        else:
            for j in range(0,num_populated_genotypes,1):
                prob_vector_view[j] = prob_vector_view[j]/denominator

        # How many to mutate this generation?
        num_to_mutate_int = random_poisson(bitgen_state, expected_num_mutations)

        # Go through the population
        for j in range(0,pop_size_int,1):

            # Use random weighted number to choose which genotype this new
            # critter will have
            cum_sum = 0.0
            rand_value = random_standard_uniform(bitgen_state)
            for k in range(0,num_populated_genotypes,1):
                cum_sum += prob_vector_view[k]
                if cum_sum >= rand_value:
                    break

            k = choice_vector_view[k]

            # If this critter is slated to mutate...
            if j < num_to_mutate_int:

                # Figure out the minimal and maximum values for this genotype's
                # neighbors
                min_index = neighbor_slicer_view[k,0]
                max_index = neighbor_slicer_view[k,1]

                # Get random integer between 0 and max_index - min_index -- which
                # neighbor to grab
                k = random_bounded_uint64(bitgen_state,
                                          min_index,max_index-min_index-1,
                                          0,0)
                # Get genotype index corresponding to that neighbor choice
                k = neighbors_view[k]

            # Update next generation population with new genotype k
            pops_view[i,k] += 1

    return pops

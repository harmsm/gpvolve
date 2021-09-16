__description__ = \
"""
Python implementation of Wright Fisher simulation.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

import numpy as np

def wf_engine(num_steps,
              num_genotypes,
              pop_size,
              num_to_mutate,
              fitness,
              neighbor_slicer,
              neighbors,
              pops):
    """
    A python implementation of the Wright Fisher engine. The arguments here
    match those in the c-extension version of the engine.

    Parameters
    ----------
    num_steps : int
        number of time steps to run
    num_genotypes : int
        number of genotypes in the whole map
    pop_size : int
        size of the population
    num_to_mutate : int
        number of genotypes to mutate to a neighbor each time step
    fitness : numpy.ndarray
        num_genotypes-long float array containing fitness of each genotype
    neighbor_slicer : numpy.ndarray
        num_genotypes-long int array containing number of neighbors accessible
        for each genotype (excluding self)
    neighbors : numpy.ndarray
        1D numpy int array storing a jagged array with neighbors for each
        genotype. neighbor_slicer is used to look up where each genotype's
        neighbors are in this array
    pops : numpy.ndarray
        num_steps + 1 by num_genotypes 2D int array that stores the population
        of each genotype for each step in the simulation. The first row holds
        the initial population of all genotypes.

    Returns
    -------
    pops : nump.ndarray
        num_steps + 1 x num_genotypes 2D int array that stores the population
        of each genotype for each step in the simulation.
    """

    indexes = np.arange(num_genotypes,dtype=int)
    for i in range(1,num_steps+1):

        # Look at non-zero genotypes
        mask = indexes[pops[i-1,:] != 0]
        local_fitness = fitness[mask]
        local_pop = pops[i-1,mask]

        # If all fitness are 0 for the populated genotypes, probability of
        # reproducing depends only on how often each genotype occurs.
        if np.sum(local_fitness) == 0:
            prob = local_pop

        # In most cases, reproduction probability is given by how many of each
        # genotype times its fitness
        else:
            prob = local_pop*local_fitness

        # Normalize prob
        prob = prob/np.sum(prob)

        # New population selected based on relative fitness
        new_pop = np.random.choice(mask,size=pop_size,p=prob,replace=True)

        # Introduce mutations
        for j in range(num_to_mutate):
            k = new_pop[j]
            new_pop[j] = np.random.choice(neighbors[neighbor_slicer[k,0]:neighbor_slicer[k,1]],size=1)[0]

        # Count how often each genotype occurs and store in pops array
        idx, counts = np.unique(new_pop,return_counts=True)
        pops[i,idx] = counts

    return pops

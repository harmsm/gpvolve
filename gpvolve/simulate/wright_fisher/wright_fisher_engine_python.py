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
              num_neighbors,
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
    num_neighbors : numpy.ndarray
        num_genotypes-long int array containing number of neighbors accessible
        for each genotype (excluding self)
    neighbors : list of numpy.ndarray
        num_genotypes-long list of int arrays, where each array stores the
        indexes of neighboring genotypes for the genotype.
    pops : numpy.ndarray
        num_steps + 1 x num_genotypes 2D int array that stores the population
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

        # If all fitness are 0 for the populated genotypes, probability of
        # reproducing depends only on how often each genotype occurs.
        if np.sum(fitness[mask]) == 0:
            prob = pops[i-1,mask]

        # In most cases, reproduction probability is given by how many of each
        # genotype times its fitness
        else:
            prob = pops[i-1,mask]*fitness[mask]

        # Normalize prob
        prob = prob/np.sum(prob)

        # New population selected based on relative fitness
        new_pop = np.random.choice(mask,size=pop_size,p=prob,replace=True)

        # Introduce mutations
        for j in range(num_to_mutate):
            new_pop[j] = np.random.choice(neighbors[new_pop[j]],size=1)[0]

        # Count how often each genotype occurs and store in pops array
        idx, counts = np.unique(new_pop,return_counts=True)
        pops[i,idx] = counts

    return pops

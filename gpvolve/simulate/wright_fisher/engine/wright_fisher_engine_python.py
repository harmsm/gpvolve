__description__ = \
"""
Python implementation of Wright Fisher simulation.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

import numpy as np

def wf_engine_python(pops,
                     mutation_rate,
                     fitness,
                     neighbor_slicer,
                     neighbors):
    """
    A python implementation of the Wright Fisher engine.

    This function should not be called directly. Instead, use wf_engine
    wrapper. Wrapper has argument docs and does argument sanity checking.
    """

    # If zero population, don't bother with simulation
    if np.sum(pops[0,:]) == 0:
        return pops

    # Get number of genoptypes, population size, and expected number of mutations
    # each generation
    num_genotypes = len(fitness)
    population_size = sum(pops[0,:])
    expected_num_mutations = mutation_rate*population_size
    num_generations = len(pops)

    indexes = np.arange(num_genotypes,dtype=int)
    for i in range(1,num_generations):

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
        new_pop = np.random.choice(mask,size=population_size,p=prob,replace=True)

        # Introduce mutations
        num_to_mutate = np.random.poisson(expected_num_mutations)

        # If we have a ridiculously high mutation rate, do not mutate each
        # genotype more than once.
        if num_to_mutate > population_size:
            num_to_mutate = population_size

        for j in range(num_to_mutate):
            k = new_pop[j]

            # If neighbor_slicer[k,0] == -1, this genotype *has* no neighbors.
            # Mutation should lead to self.
            if neighbor_slicer[k,0] != -1:
                a = neighbors[neighbor_slicer[k,0]:neighbor_slicer[k,1]]
                new_pop[j] = np.random.choice(a,size=1)[0]

        # Count how often each genotype occurs and store in pops array
        idx, counts = np.unique(new_pop,return_counts=True)
        pops[i,idx] = counts

    return pops

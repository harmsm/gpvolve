
import pytest
import gpmap
import gpvolve
from gpvolve.markov import utils

import numpy as np

import itertools

def test_generate_tmatrix():

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors()

    fitness = np.array(gpm.fitness)
    neighbor_slicer, neighbors = gpvolve.utils.flatten_neighbors(gpm)

    for use_cython in [False,True]:
        print("Using cython?",use_cython)

        T = utils.generate_tmatrix(fitness,
                                   neighbor_slicer,
                                   neighbors,
                                   use_cython=use_cython)

        assert T.shape == (4,4)

        # Make sure it's row stochastic
        S = np.sum(T,axis=1)
        for i in range(4):
            assert np.isclose(S[i],1.0)

        bad_fitness = ["stupid",
                       [1.0,1.0,1.0,np.nan],
                       [-1,1,1,1],
                       [1,1,1],None]
        for b in bad_fitness:
            with pytest.raises(ValueError):
                T = utils.generate_tmatrix(b,
                                           neighbor_slicer,
                                           neighbors,
                                           use_cython=use_cython)

        bad_population_size = ["stupid",None,-1,0]
        for b in bad_population_size:
            with pytest.raises(ValueError):
                T = utils.generate_tmatrix(fitness,
                                           neighbor_slicer,
                                           neighbors,
                                           population_size=b,
                                           use_cython=use_cython)

        bad_model = ["stupid",None,-1,(),[]]
        for b in bad_model:
            with pytest.raises(ValueError):
                T = utils.generate_tmatrix(fitness,
                                           neighbor_slicer,
                                           neighbors,
                                           fixation_model=b,
                                           use_cython=use_cython)



        models = ["moran","mcclandish","sswm"]
        population_size = 10**np.arange(0,15,dtype=int)
        for p in population_size:
            for m in models:
                T = utils.generate_tmatrix(fitness,
                                           neighbor_slicer,
                                           neighbors,
                                           population_size=p,
                                           fixation_model=m,
                                           use_cython=use_cython)
                S = np.sum(T,axis=1)
                for i in range(4):
                    assert np.isclose(S[i],1.0)

    # Generate slightly bigger map with different fitness values (near 1 for
    # all)
    genotype = ["".join(g) for g in itertools.product(["0","1"],repeat=5)]
    fitness = 1.0 + (np.random.random(len(genotype)) - 0.5)*0.1
    gpm = gpmap.GenotypePhenotypeMap(genotype,fitness=fitness)
    gpm.get_neighbors()
    neighbor_slicer, neighbors = gpvolve.utils.flatten_neighbors(gpm)
    num_genotypes = len(neighbor_slicer)

    # Try cython and python impelementation
    for use_cython in [False,True]:

        # Fixation models...
        for m in ["moran","mcclandish"]:

            # Set pop size relatively low to allow mutations...
            T = utils.generate_tmatrix(gpm.fitness,
                                       neighbor_slicer,
                                       neighbors,
                                       population_size=10,
                                       fixation_model=m,
                                       use_cython=use_cython)

            # Make sure it has the right shape
            assert T.shape == (num_genotypes,num_genotypes)

            # Make sure it is stochastic
            S = np.sum(T,axis=1)
            for i in range(num_genotypes):
                assert np.isclose(S[i],1.0)

            # Make sure self and neighbors have non-zero entries; all others
            # should have zero
            for i in range(len(neighbor_slicer)):

                i_neighbors = neighbors[neighbor_slicer[i,0]:neighbor_slicer[i,1]]
                for j in range(len(neighbor_slicer)):
                    if j in i_neighbors:
                        assert T[i,j] != 0
                    elif i == j:
                        assert not np.isclose(T[i,j],0)
                    else:
                        assert np.isclose(T[i,j],0)

            # With pop size of 1, this should make diagonal zero. *Some*
            # mutation will happen and fix...
            T = utils.generate_tmatrix(gpm.fitness,
                                       neighbor_slicer,
                                       neighbors,
                                       population_size=1,
                                       fixation_model=m,
                                       use_cython=use_cython)
            for i in range(num_genotypes):
                assert np.isclose(T[i,i],0)

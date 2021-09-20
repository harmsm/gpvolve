
import pytest
import gpmap
import numpy as np

import gpvolve
from gpvolve.markov import utils

def test_generate_tmatrix():

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors()

    fitness = np.array(gpm.fitness)
    neighbor_slicer, neighbors = gpvolve.utils.flatten_neighbors(gpm)

    for use_cython in [False,True]:

        T = utils.generate_tmatrix(fitness,
                                   neighbor_slicer,
                                   neighbors,
                                   use_cython=use_cython)

        assert T.shape == (4,4)

        S = np.sum(T,axis=0)
        for i in range(4):
            assert np.isclose(S[i],1.0)





    # fitness,
    #                       neighbor_slicer,
    #                       neighbors,
    #                       population_size=1000,
    #                       fixation_model="moran",
    #                       use_cython=True)

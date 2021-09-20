import pytest

import gpmap

import gpvolve
from gpvolve.utils import flatten_neighbors

import numpy as np

def test_flatten_neighbors():

    bad_gpm = ["stupid",1,(),None] #,gpm]
    for b in bad_gpm:
        with pytest.raises(TypeError):
            flatten_neighbors(b)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    with pytest.raises(ValueError):
        flatten_neighbors(gpm)

    gpm.get_neighbors(neighbor_function="hamming",cutoff=1)
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)
    assert np.array_equal(neighbor_slicer[:,0],[0,2,4,6])
    assert np.array_equal(neighbor_slicer[:,1],[2,4,6,8])
    assert np.array_equal(neighbors,[1,2,0,3,0,3,1,2])

    

    gpm.get_neighbors(neighbor_function="hamming",cutoff=0)
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)
    print(neighbor)
    print(neighbor_slicer)

    #ssert np.array_equal(neighbor_slicer[:,0],[0,2,4,6])
    #assert np.array_equal(neighbor_slicer[:,1],[2,4,6,8])
    #assert np.array_equal(neighbors,[1,2,0,3,0,3,1,2])
    assert False

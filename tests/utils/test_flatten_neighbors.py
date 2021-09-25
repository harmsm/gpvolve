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

    # Self and non-self
    gpm.get_neighbors(neighbor_function="hamming",cutoff=1)
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)
    assert np.array_equal(neighbor_slicer[:,0],[0,3,6,9])
    assert np.array_equal(neighbor_slicer[:,1],[3,6,9,12])
    assert np.array_equal(neighbors,[0,1,2,0,1,3,0,2,3,1,2,3])

    # Only self
    gpm.get_neighbors(neighbor_function="hamming",cutoff=0)
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)

    assert np.array_equal(neighbor_slicer[:,0],[0,1,2,3])
    assert np.array_equal(neighbor_slicer[:,1],[1,2,3,4])
    assert np.array_equal(neighbors,[0,1,2,3])

    # One self neighbor
    gpm.get_neighbors(neighbor_function="hamming",cutoff=0)
    gpm.neighbors.loc[:,"include"] = False
    gpm.neighbors.loc[0,"include"] = True
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)

    assert np.array_equal(neighbor_slicer[:,0],[0,-1,-1,-1])
    assert np.array_equal(neighbor_slicer[:,1],[1,-1,-1,-1])
    assert np.array_equal(neighbors,[0])

    # No neighbors
    gpm.get_neighbors(neighbor_function="hamming",cutoff=0)
    gpm.neighbors.loc[:,"include"] = False
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)

    assert np.array_equal(neighbor_slicer[:,0],[-1,-1,-1,-1])
    assert np.array_equal(neighbor_slicer[:,1],[-1,-1,-1,-1])
    assert np.array_equal(neighbors,[])

    # One non-self neighbor
    gpm.get_neighbors(neighbor_function="hamming",cutoff=1)
    gpm.neighbors.loc[:,"include"] = False
    gpm.neighbors.loc[1,"include"] = True
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)

    assert np.array_equal(neighbor_slicer[:,0],[0,-1,-1,-1])
    assert np.array_equal(neighbor_slicer[:,1],[1,-1,-1,-1])
    assert np.array_equal(neighbors,[1])

    # Only non-self
    gpm.get_neighbors(neighbor_function="hamming",cutoff=1)
    mask = gpm.neighbors.loc[:,"target"] == gpm.neighbors.loc[:,"source"]
    gpm.neighbors.loc[:,"include"] = True
    gpm.neighbors.loc[mask,"include"] = False
    neighbor_slicer, neighbors = flatten_neighbors(gpm)
    assert neighbor_slicer.shape == (4,2)
    assert np.array_equal(neighbor_slicer[:,0],[0,2,4,6])
    assert np.array_equal(neighbor_slicer[:,1],[2,4,6,8])
    assert np.array_equal(neighbors,[1,2,0,3,0,3,1,2])

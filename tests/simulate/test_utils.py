import pytest

import gpvolve
from gpvolve.simulate.utils import check_simulation_parameter_sanity as csps
import gpmap

import numpy as np

import itertools, copy


def test_check_simulation_parameter_sanity():

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    dummy_good = {"gpm":gpm,
                  "max_generations":100,
                  "mutation_rate":0.01,
                  "population_size":10,
                  "fitness_column":"fitness"}

    # ------------------------------------------------------------------------
    # Check gpm
    # ------------------------------------------------------------------------

    gpm_dummy = copy.deepcopy(dummy_good)
    gpm_dummy.pop("gpm")

    with pytest.raises(TypeError):
        csps("stupid",**gpm_dummy)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm._data = "stupid"
    with pytest.raises(ValueError):
        csps(gpm,**gpm_dummy)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    with pytest.raises(ValueError):
        csps(gpm,**gpm_dummy)

    gpm._neighbors = "stupid"
    with pytest.raises(ValueError):
        csps(gpm,**gpm_dummy)

    # Screw up target column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["target"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        csps(gpm,**gpm_dummy)

    # Screw up source column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["source"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        csps(gpm,**gpm_dummy)

    # This should work
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    out = csps(gpm,**gpm_dummy)
    assert out[0] is gpm

    # ------------------------------------------------------------------------
    # Check num step
    # ------------------------------------------------------------------------
    num_gen_dummy = copy.deepcopy(dummy_good)
    num_gen_dummy.pop("max_generations")

    bad_num_generations = ["stupid",-1,(1,3)]
    for n in bad_num_generations:
        with pytest.raises(ValueError):
            csps(max_generations=n,**num_gen_dummy)

    # Make sure num_generations translated correctly
    out = csps(max_generations=10,**num_gen_dummy)
    assert out[1] == 10

    # Make sure num_generations translated correctly
    out = csps(max_generations=0,**num_gen_dummy)
    assert out[1] == 0

    # ------------------------------------------------------------------------
    # Check mutation rate
    # ------------------------------------------------------------------------
    mu_dummy = copy.deepcopy(dummy_good)
    mu_dummy.pop("mutation_rate")

    bad_mutation_rates = ["stupid",-1,2,(1,3)]
    for m in bad_mutation_rates:
        with pytest.raises(ValueError):
            csps(mutation_rate=m,**mu_dummy)

    # Should work
    out = csps(mutation_rate=0.1,**mu_dummy)
    assert out[2] == 0.1

    # ------------------------------------------------------------------------
    # population_size
    # ------------------------------------------------------------------------

    N_dummy = copy.deepcopy(dummy_good)
    N_dummy.pop("population_size")

    bad_pops = [-1,0,"stupid",(1,1)]
    for p in bad_pops:
        with pytest.raises(ValueError):
            csps(population_size=p,**N_dummy)

    out = csps(population_size=10,**N_dummy)
    assert out[3] == 10

    # ------------------------------------------------------------------------
    # check fitness column
    # ------------------------------------------------------------------------
    fitness_dummy = copy.deepcopy(dummy_good)
    fitness_dummy.pop("fitness_column")

    with pytest.raises(KeyError):
        csps(fitness_column="stupid",**fitness_dummy)

    bad_fitness = [[-1,0,0,1],[np.nan,0,0,0]]
    for f in bad_fitness:
        gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                         fitness=f)
        gpm.get_neighbors()
        fitness_dummy_bad = copy.deepcopy(fitness_dummy)
        fitness_dummy_bad["gpm"] = gpm
        with pytest.raises(ValueError):
            csps(fitness_column="fitness",**fitness_dummy_bad)

    out = csps(fitness_column="fitness",**fitness_dummy)
    assert np.array_equal(out[4],[0.1,0.2,0.2,0.3])

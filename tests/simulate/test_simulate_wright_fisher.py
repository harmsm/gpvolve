
import pytest

import gpvolve
from gpvolve.simulate.wright_fisher import simulate
import gpmap

import numpy as np

import itertools

def test_wright_fisher_simulate_argparse():

    # ------------------------------------------------------------------------
    # Check gpm
    # ------------------------------------------------------------------------

    with pytest.raises(TypeError):
        simulate("stupid",0.1)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm._data = "stupid"
    with pytest.raises(ValueError):
        simulate(gpm,0.1)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    with pytest.raises(ValueError):
        simulate(gpm,0.1)

    gpm._neighbors = "stupid"
    with pytest.raises(ValueError):
        simulate(gpm,0.1)

    # Screw up target column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["target"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        simulate(gpm,0.1,num_steps=1)

    # Screw up source column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["source"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        simulate(gpm,0.1,num_steps=1)

    # This should work
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    simulate(gpm,0.1,num_steps=1)

    # ------------------------------------------------------------------------
    # Check mutation rate
    # ------------------------------------------------------------------------
    bad_mutation_rates = ["stupid",-1,2,(1,3)]
    for m in bad_mutation_rates:
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=m)

    # should work
    good_mutation_rates = [0,0.5,1.0]
    for m in good_mutation_rates:
        simulate(gpm,mutation_rate=m,num_steps=1)

    # ------------------------------------------------------------------------
    # Check num step
    # ------------------------------------------------------------------------
    bad_num_steps = ["stupid",-1,(1,3)]
    for n in bad_num_steps:
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=0.1,num_steps=n)

    # should work
    good_num_steps = [0,1,10]
    for n in good_num_steps:
        simulate(gpm,0.1,num_steps=n)

    # ------------------------------------------------------------------------
    # initial_pop_column
    # ------------------------------------------------------------------------

    # Send in bad pop column
    with pytest.raises(KeyError):
        simulate(gpm,mutation_rate=0.1,num_steps=1,initial_pop_column="stupid")

    # Send in bad pop column values
    bad_pops = [[0,0,0,0],[1,0,0,-1],["my","favorite","bad","column"]]
    for b in bad_pops:
        with pytest.raises(ValueError):
            gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                             fitness=[0.1,0.2,0.2,0.3])
            gpm.get_neighbors()
            gpm.data.loc[:,"pops"] = b
            simulate(gpm,mutation_rate=0.1,num_steps=1,initial_pop_column="pops")

    # Send in pops array
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.data.loc[:,"pops"] = [0,0,1000,0]
    pops = simulate(gpm,mutation_rate=0.1,num_steps=0,initial_pop_column="pops")
    assert np.array_equal(pops[0,:],gpm.data.loc[:,"pops"])

    # No pops, should die b/c name column not in df
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.data.drop(labels="name",axis=1,inplace=True)
    with pytest.raises(ValueError):
        pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=10,
                                  initial_pop_column=None)

    # No pops, should die b/c wildtype not in data df
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.data.loc[:,"name"] = ["not","good","name","scheme"]
    with pytest.raises(ValueError):
        pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=10,
                                  initial_pop_column=None)

    # No pops --> stick pop_size on wildtype
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    pop_size = 10
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=pop_size,
                              initial_pop_column=None)
    assert np.array_equal(pops[0,:],[pop_size,0,0,0])

    # ------------------------------------------------------------------------
    # pop_size
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    bad_pops = [-1,0,"stupid",(1,1)]
    for p in bad_pops:
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=p)

    pops = simulate(gpm,mutation_rate=0.1,num_steps=0,pop_size=10)
    assert np.sum(pops[0]) == 10

    # ------------------------------------------------------------------------
    # check fitness column
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    with pytest.raises(KeyError):
        simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=10,fitness_column="stupid")

    bad_fitness = [[-1,0,0,1],[np.nan,0,0,0]]
    for f in bad_fitness:
        gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                         fitness=f)
        gpm.get_neighbors()
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=10,fitness_column="fitness")

    # ------------------------------------------------------------------------
    # check num_replicate_sims
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    bad_num_replicate_sims = ["test",(1,3),0]
    for n in bad_num_replicate_sims:
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=10,num_replicate_sims=n)

    results = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=1,num_replicate_sims=2)
    assert type(results) is list
    assert len(results) == 2


    # ------------------------------------------------------------------------
    # check num_threads
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    bad_num_threads = ["test",(1,3),0]
    for n in bad_num_threads:
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=10,num_threads=n)

    # Run multithreaded. Other runs tested default (num_threads = 1)
    results = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=1,num_threads=2,num_replicate_sims=2)
    assert type(results) is list
    assert len(results) == 2



def test_wright_fisher_simulate():

    # ------------------------------------------------------------------------
    # Switch into really testing sim method
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    # NOT SURE HOW TO ACTUALLY TEST THIS GIVEN CURRENT IMPLEMENTATION
    # # Make sure we make right number of attempts give mutation_rate and pop_size
    # pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=100)
    # assert np.sum(list(attempts.values())) == 10
    #
    # pops = simulate(gpm,mutation_rate=1,num_steps=1,pop_size=100)
    # assert np.sum(list(attempts.values())) == 100
    #
    # pops = simulate(gpm,mutation_rate=0,num_steps=1,pop_size=100)
    # assert np.sum(list(attempts.values())) == 0
    #
    # # Make sure we don't ever try a self -> self move
    # pops = simulate(gpm,mutation_rate=1,num_steps=1,pop_size=100)
    # for k in attempts.keys():
    #     assert k[0] != k[1]

    # Run simulation multiple times with equal fitness and equal pops. Should
    # end up basically equal at end.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.1,0.1,0.1],
                                     pops=[200,200,200,200])
    gpm.get_neighbors()
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1000,initial_pop_column="pops")
    result = np.sum(pops,axis=0)
    assert 1 - np.min(result)/np.max(result) < 0.10

    # make sure conservation of mass holds
    assert len(np.unique(np.sum(pops,axis=1))) == 1
    assert np.unique(np.sum(pops,axis=1))[0] == 800

    # Start at 00. Give 11 *way* higher fitness. With low mutation rate and many
    # steps, expect final step to be 99 in 11, 1 in other (due to mutation)
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,2,2,1000],
                                     pops=[100,0,0,0])
    gpm.get_neighbors()
    pops = simulate(gpm,mutation_rate=0.01,num_steps=5000,initial_pop_column="pops")
    assert pops[-1,-1] >= 98

    # make sure conservation of mass holds
    assert len(np.unique(np.sum(pops,axis=1))) == 1
    assert np.unique(np.sum(pops,axis=1))[0] == 100


    # Make sure memory size allocation is right
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=2**8 - 1)
    assert isinstance(pops[-1,0],np.uint8)
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=2**8)
    assert isinstance(pops[-1,0],np.uint16)
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=2**16-1)
    assert isinstance(pops[-1,0],np.uint16)
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1,pop_size=2**16)
    assert isinstance(pops[-1,0],np.uint32)


def test_wright_fisher_simulate_larger():

    # Do a generic run for a large-ish map.
    genotype = ["".join(g) for g in itertools.product("01",repeat=8)]
    fitness = np.random.random(len(genotype))
    gpm = gpmap.GenotypePhenotypeMap(genotype=genotype,fitness=fitness)
    gpm.get_neighbors()
    pops = simulate(gpm,mutation_rate=0.1,num_steps=1000,pop_size=500)
    assert isinstance(pops[-1,0],np.uint16)
    assert pops.shape[0] == 1001
    assert pops.shape[1] == len(genotype)

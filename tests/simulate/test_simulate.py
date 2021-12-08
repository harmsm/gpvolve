
import pytest

import gpvolve
from gpvolve.simulate import simulate
import gpmap

import numpy as np

import itertools

def test_simulate_argpass():

    # ------------------------------------------------------------------------
    # Check gpm
    # ------------------------------------------------------------------------

    with pytest.raises(TypeError):
        simulate("stupid")

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm._data = "stupid"
    with pytest.raises(ValueError):
        simulate(gpm)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    with pytest.raises(ValueError):
        simulate(gpm)

    gpm._neighbors = "stupid"
    with pytest.raises(ValueError):
        simulate(gpm)

    # Screw up target column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["target"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        simulate(gpm)

    # Screw up source column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["source"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        simulate(gpm)

    # This should work
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    pops = simulate(gpm,num_steps=1)
    assert pops.shape == (2,4)

    # ------------------------------------------------------------------------
    # Check engine
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()


    bad_engine = ["stupid",1,[],None]
    for b in bad_engine:
        with pytest.raises(ValueError):
            simulate(gpm,engine=b)

    # This should work
    pops = simulate(gpm,num_steps=1,engine="wf")

    # ------------------------------------------------------------------------
    # Check mutation rate
    # ------------------------------------------------------------------------
    bad_mutation_rates = ["stupid",-1,2,(1,3)]
    for m in bad_mutation_rates:
        with pytest.raises(ValueError):
            simulate(gpm,mutation_rate=m)

    # Should work
    simulate(gpm,num_steps=1,mutation_rate=0.1)

    # ------------------------------------------------------------------------
    # Check num step
    # ------------------------------------------------------------------------
    bad_num_steps = ["stupid",-1,(1,3)]
    for n in bad_num_steps:
        with pytest.raises(ValueError):
            simulate(gpm,num_steps=n)

    # Make sure num_steps translated correctly
    pops = simulate(gpm,num_steps=10)
    pops.shape == (11,4)

    # Make sure num_steps translated correctly
    pops = simulate(gpm,num_steps=0)
    pops.shape == (1,4)

    # ------------------------------------------------------------------------
    # initial_pop_column
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    # Send in bad pop column
    with pytest.raises(KeyError):
        simulate(gpm,initial_pop_column="stupid")

    # Send in bad pop column values
    bad_pops = [[0,0,0,0],[1,0,0,-1],["my","favorite","bad","column"]]
    for b in bad_pops:
        with pytest.raises(ValueError):
            gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                                 fitness=[0.1,0.2,0.2,0.3])
            gpm.get_neighbors()
            gpm.data.loc[:,"pops"] = b
            simulate(gpm,initial_pop_column="pops")


    # No pops, should die b/c no wildtype can be found to assign initial
    # population to (name column not in df)
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                         fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.data.drop(labels="name",axis=1,inplace=True)
    with pytest.raises(ValueError):
        pops = simulate(gpm,pop_size=10,initial_pop_column=None)

    # No pops, should die b/c wildtype not in data df
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                         fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.data.loc[:,"name"] = ["not","good","name","scheme"]
    with pytest.raises(ValueError):
        pops = simulate(gpm,pop_size=10,initial_pop_column=None)

    # Make sure pops passed in correctly if specified by column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3],
                                     pops=[100,100,0,0])
    gpm.get_neighbors()
    pops = simulate(gpm,num_steps=10,initial_pop_column="pops")
    assert pops.shape == (11,4)
    assert np.sum(pops[0,:]) == 200
    assert np.sum(pops[-1,:]) == 200
    assert np.array_equal(pops[0,:],[100,100,0,0])

    # Make sure we get all pop on wildtype if we specify a pop_size but no
    # initial pop column.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3],wildtype="00")
    gpm.get_neighbors()
    pop_size = 10
    pops = simulate(gpm,num_steps=10,pop_size=pop_size)
    assert np.array_equal(pops[0,:],[pop_size,0,0,0])

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3],wildtype="11")
    gpm.get_neighbors()
    pop_size = 10
    pops = simulate(gpm,num_steps=10,pop_size=pop_size)
    assert np.array_equal(pops[0,:],[0,0,0,pop_size])


    # ------------------------------------------------------------------------
    # pop_size
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    bad_pops = [-1,0,"stupid",(1,1)]
    for p in bad_pops:
        with pytest.raises(ValueError):
            simulate(gpm,pop_size=p)

    pops = simulate(gpm,num_steps=1,pop_size=10)
    assert np.sum(pops[0,:]) == 10
    assert np.sum(pops[1,:]) == 10

    # ------------------------------------------------------------------------
    # check fitness column
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    with pytest.raises(KeyError):
        simulate(gpm,fitness_column="stupid")

    bad_fitness = [[-1,0,0,1],[np.nan,0,0,0]]
    for f in bad_fitness:
        gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                         fitness=f)
        gpm.get_neighbors()
        with pytest.raises(ValueError):
            simulate(gpm,fitness_column="fitness")

    # Run a simulation where second position should have vast majority of
    # population based on fitness if fitness passed properly
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,100,0.1,0.1])
    gpm.get_neighbors()
    pops = simulate(gpm,fitness_column="fitness",pop_size=100,num_steps=100,
                    mutation_rate=0.1)
    assert np.argmax(pops[0,:]) == 0
    assert np.argmax(pops[-1,:]) == 1

    # ------------------------------------------------------------------------
    # check num_replicate_sims
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    bad_num_replicate_sims = ["test",(1,3),0]
    for n in bad_num_replicate_sims:
        with pytest.raises(ValueError):
            simulate(gpm,num_steps=1,pop_size=10,num_replicate_sims=n)

    results = simulate(gpm,num_replicate_sims=2)
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
            simulate(gpm,num_threads=n)

    # Run multithreaded. Other runs tested default (num_threads = 1)
    results = simulate(gpm,num_steps=1,pop_size=1,num_threads=2,num_replicate_sims=2)
    assert type(results) is list
    assert len(results) == 2

    # ------------------------------------------------------------------------
    # check use_cython
    # ------------------------------------------------------------------------

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()

    pops = simulate(gpm,num_steps=10,pop_size=10,use_cython=False)
    assert pops.shape == (11,4)
    assert np.sum(pops[-1,:]) == 10

    pops = simulate(gpm,num_steps=10,pop_size=10,use_cython=True)
    assert pops.shape == (11,4)
    assert np.sum(pops[-1,:]) == 10

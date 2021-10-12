import itertools
import gpmap
import numpy as np
from gpvolve import utils
from gpvolve.simulate import wright_fisher as wf


def _test_wf_engine(use_cython=False):
    """
    Test simulation protocols for a wide variety of sane values.

    The engines do *not* do sanity checking, so that has to be taken care of
    upstream in the code.
    """

    print("Using cython?",use_cython)

    gpm = gpmap.GenotypePhenotypeMap(["00","10","01","11"])
    gpm.get_neighbors()

    initial_pops = [np.array([1,0,0,0],dtype=int)*1000,
                    np.array([1,1,1,1],dtype=int)*250,
                    np.array([1,0,0,1],dtype=int)*500,
                    np.array([0,0,1,1],dtype=int)*500,
                    np.array([1,0,0,0],dtype=int),
                    np.array([10,0,0,0],dtype=int),
                    np.array([0,0,0,0],dtype=int)]

    possible_pops = []
    for p in initial_pops:
        possible_pops.append(np.zeros((11,4),dtype=int))
        possible_pops[-1][0,:] = p

    possible_mu = [0.0,0.001,0.01,0.1,1.0,10.0]
    possible_fitness = [np.array([1,1,1,1],dtype=float),
                        np.array([0,0,0,0],dtype=float),
                        np.array([.1,1,.1,.1],dtype=float)]

    neighbor_slicer, neighbors = utils.flatten_neighbors(gpm)



    for i, p in enumerate(possible_pops):
        for m in possible_mu:
            for f in possible_fitness:

                pops = wf.wf_engine(pops=p,
                                    mutation_rate=m,
                                    fitness=f,
                                    neighbor_slicer=neighbor_slicer,
                                    neighbors=neighbors,
                                    use_cython=use_cython)

                # Pops array should be self, not copy
                assert pops is p

                # Initial population should be same as what we hoped
                assert np.array_equal(pops[0,:],initial_pops[i])

def _test_numerical(use_cython=False):

    print("Using cython?",use_cython)

    # ------------------------------------------------------------------------
    # Switch into really testing sim method
    # ------------------------------------------------------------------------

    # Run simulation with equal fitness and equal pops. Should end up basically
    # equal at end.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.1,0.1,0.1],
                                     pops=[200,200,200,200])
    gpm.get_neighbors()
    neighbor_slicer, neighbors = utils.flatten_neighbors(gpm)

    pops = np.zeros((5001,len(gpm.pops)),dtype=int)
    pops[0,:] = gpm.pops

    pops = wf.wf_engine(pops=pops,
                        mutation_rate=0.1,
                        fitness=gpm.fitness,
                        neighbor_slicer=neighbor_slicer,
                        neighbors=neighbors,
                        use_cython=use_cython)

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
    neighbor_slicer, neighbors = utils.flatten_neighbors(gpm)

    pops = np.zeros((5001,len(gpm.pops)),dtype=int)
    pops[0,:] = gpm.pops

    pops = wf.wf_engine(pops=pops,
                        mutation_rate=0.01,
                        fitness=gpm.fitness,
                        neighbor_slicer=neighbor_slicer,
                        neighbors=neighbors,
                        use_cython=use_cython)

    assert pops[-1,-1] >= 95

    # make sure conservation of mass holds
    assert len(np.unique(np.sum(pops,axis=1))) == 1
    assert np.unique(np.sum(pops,axis=1))[0] == 100

def _test_simulate_larger(use_cython=False):

    print("Using cython?",use_cython)

    # Do a generic run for a large-ish map.
    genotype = ["".join(g) for g in itertools.product("01",repeat=8)]
    fitness = np.random.random(len(genotype))
    gpm = gpmap.GenotypePhenotypeMap(genotype=genotype,fitness=fitness)
    gpm.get_neighbors()

    neighbor_slicer, neighbors = utils.flatten_neighbors(gpm)

    pops = np.zeros((1001,len(gpm.data)),dtype=int)
    pops[0,0] = 500

    pops = wf.wf_engine(pops=pops,
                        mutation_rate=0.1,
                        fitness=fitness,
                        neighbor_slicer=neighbor_slicer,
                        neighbors=neighbors,
                        use_cython=use_cython)

    assert pops.shape[0] == 1001
    assert pops.shape[1] == len(genotype)

def test_wf_engine_python():
    _test_wf_engine(use_cython=False)

def test_wf_engine_cython():
    _test_wf_engine(use_cython=True)

def test_numerical_python():
    _test_numerical(use_cython=False)

def test_numerical_cython():
    _test_numerical(use_cython=True)

def test_simulate_larger_python():
    _test_simulate_larger(use_cython=False)

def test_simulate_larger_cython():
    _test_simulate_larger(use_cython=True)

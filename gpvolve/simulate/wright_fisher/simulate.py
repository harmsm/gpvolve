__description__ = \
"""
Run Wright Fisher simulation on a genotype phentoype map.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

from .engine import wf_engine
import gpvolve.check as check

import gpvolve
import gpvolve.utils as utils

import gpmap

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import multiprocessing as mp
import warnings, os

def _sim_on_thread(args):
    """
    Run a simulation on a thread. Constructs a pops array for storing simulation
    results and then runs simulation_engine(pops,*args[5:]).

    Parameters
    ----------
    args : tuple
        arg[0] is simulation engine
        arg[1] is queue for storing results. If None, return results directly.
        arg[2] is index uniquely identifying thread run.
        arg[3] is initial population array
        arg[4] is number of steps to run
        Remaining args are sent to wf_engine.
    """

    # Parse args tuple
    engine_function = args[0]
    queue = args[1]
    index = args[2]
    initial_pop = args[3]
    num_generations = args[4]

    # Make pops array to store results
    pops = np.zeros((num_generations+1,len(initial_pop)),dtype=int)
    pops[0] = initial_pop

    # Construct args to send to wf_engine, making appopriate pops array
    wf_args = [pops]
    wf_args.extend(args[5:])
    wf_args = tuple(wf_args)

    # Do calculation
    pops = engine_function(*wf_args)

    # If multithreaded (queue not None), append to queue. Otherwise, just
    # return
    if queue is None:
        return pops
    else:
        queue.put(pops)


def simulate(gpm,
             engine="wf",
             num_generations=1000,
             mutation_rate=0.001,
             population_size=100,
             fitness_column="fitness",
             initial_pop_column=None,
             num_replicate_sims=1,
             num_threads=1,
             use_cython=True):
    """
    Simulate evolution across a GenotypePhenotypeMap.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap
        genotype phenotype map to use for the simulation
    engine : str
        simulation engine to use. "wf": Wright-Fisher. Currently only engine
        available.
    num_generations : int
        number of steps to run the simulation (must be >= 0)
    mutation_rate : float
        how probable is it that a genotype mutates over a generation?
    population_size : int
        population size. must be int > 0. if initial_pop_column is set, this
        overrides the population size
    fitness_column : str
        column in gpm.data that has fitness value for each genotype
    initial_pop_column : str
        column in gpm.data that has initial population for each genotype. If
        None, assign the wildtype genotype a population population_size and set rest
        to 0.
    num_replicate_sims : int
        number of replicate simulations to run. must be int > 0.
    num_threads : int
        number of threads to run (one simulation per thread). if None, use all
        available cpus. If set, must be an int > 0.
    use_cython : bool
        use faster cython implementation if available.

    Returns
    -------
    results : np.ndarray or list
        if num_replicate_sims == 1 (default), return num_generations + 1 x num_genotypes
        array with genotype counts over steps. If num_replicate_sims > 1, return
        a list of arrays, one for each replicate simulation.
    """

    check.gpm_sanity(gpm)

    # Get engine function
    engine_functions = {"wf":wf_engine}
    try:
        engine_function = engine_functions[engine]
    except (KeyError,TypeError):
        err = f"engine '{engine}' not recognized. Should be one of:\n"
        for k in engine_functions:
            err += "    {k}\n"
        raise ValueError(err)

    # Get initial population vector
    initial_pop = None
    if initial_pop_column is not None:

        try:
            initial_pop = np.array(np.round(gpm.data.loc[:,initial_pop_column],0),
                                   dtype=int)
            if np.min(initial_pop) < 0:
                raise ValueError
            if np.sum(initial_pop) < 1:
                raise ValueError
        except KeyError:
            err = f"initial_pop_column '{initial_pop_column}' not in gpm.data\n"
            err += "dataframe\n"
            raise KeyError(err)
        except (TypeError,ValueError):
            err = "initial_pop_column must point to a data column that can be\n"
            err += "coerced as an int, whose smallest value is >= 0, and that\n"
            err += "has at least one row with a value > 0.\n"
            raise ValueError(err)

        # Population size from column overwrites whatever came in as a kwarg
        population_size  = np.sum(initial_pop)

    # Make sure simulation parameters are sane
    gpm, num_generations, mutation_rate, population_size, fitness = \
        gpvolve.simulate.utils.check_simulation_parameter_sanity(gpm,
                                                                 num_generations,
                                                                 mutation_rate,
                                                                 population_size,
                                                                 fitness_column)

    # If we have not made an initial_pop array above, make one with the wildtype
    # genotype as population_size and all other genotypes as 0
    if initial_pop is None:

        initial_pop = np.zeros(len(gpm.data),dtype=int)

        err = None
        try:
            wt_index = np.arange(len(gpm.data),dtype=int)[gpm.data.name == "wildtype"][0]
        except AttributeError:
            err = "gpm.data should have a 'name' column.\n"
        except IndexError:
            err = "gpm.data 'name' column does not contain 'wildtype'.\n"

        if err is not None:
            err += "If no initial_pop_column is specified, the 'wildtype'\n"
            err += "genotype is assigned a population of 'population_size' and all\n"
            err += "other genotypes are assigned population of 0.\n"
            raise ValueError(err)

        initial_pop[wt_index] = population_size

    # Figure out number of threads to use
    if num_threads is None:
        try:
            num_threads = mp.cpu_count()
        except NotImplementedError:
            num_threads = os.cpu_count()
            if num_threads is None:
                warnings.warn("Could not determine number of cpus. Using single thread.\n")
                num_threads = 1

    # Sanity check on number of threads
    try:
        num_threads = int(num_threads)
        if num_threads < 1:
            raise ValueError
    except (ValueError,TypeError):
        err = "num_threads should be an integer > 1 or None. (If None, the \n"
        err += "number of threads is set to the number of cpus)\n"
        raise ValueError(err)

    # Sanity check on number of replicate simulations
    try:
        num_replicate_sims = int(num_replicate_sims)
        if num_replicate_sims < 1:
            raise ValueError
    except (ValueError,TypeError):
        err = "num_replicate_sims should be an integer > 1.\n"
        raise ValueError(err)

    # Make sure use_cython can be cast as bool (should pretty much always work)
    try:
        use_cython = bool(use_cython)
    except TypeError:
        err = "use_cython must be 'True' or 'False'\n"
        raise TypeError(err)

    # -------------------------------------------------------------------------
    # Prep for calculation
    # -------------------------------------------------------------------------

    # Check for poor connectivity and throw a warning if such nodes exist.
    utils.check_neighbor_connectivity(gpm,warn=True)

    # Get neighbors into useful form
    neighbor_slicer, neighbors = utils.flatten_neighbors(gpm)

    # Decide whether we are running replicates on multiple threads (meaning
    # we need a queue) or just running on one thread.
    if num_threads == 1:
        queue = None
    else:
        # Kick off queue for capturing results
        queue = mp.Manager().Queue()

    # Build a list of args to pass for each replicate. This is formatted in
    # this way so it can be passed via multiprocessing.imap call.
    all_args = []
    for i in range(num_replicate_sims):

        args = [engine_function,queue,i,initial_pop,num_generations,
                mutation_rate,fitness,neighbor_slicer,neighbors,
                use_cython]

        # Append this to args
        all_args.append(tuple(args))

    # If only one thread, call without dealing with Pool/queue complexity
    if num_threads  == 1:
        results = []
        for a in all_args:
            results.append(_sim_on_thread(a))

    # If more than one thread, put under control of Pool
    else:
        with mp.Pool(num_threads) as pool:

            # pool.imap() runs a function on elements in iterable, filling threads
            # as each job finishes. (Calls _sim_on_thread on every args tuple in
            # all_args). This black magic call inserts a tqdm progress bar
            # into this process. It means every time a thread finishes, the
            # progress bar updates.
            list(tqdm(pool.imap(_sim_on_thread,all_args),
                      total=len(all_args)))

        # Get pops arrays from queue object
        results = []
        while not queue.empty():
            results.append(queue.get())

    # If we are running a single replicate, just return the sim
    if num_replicate_sims == 1:
        return results[0]

    # Otherwise, return a list of sims
    return results

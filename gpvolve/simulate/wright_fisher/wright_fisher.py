__description__ = \
"""
Run Wright Fisher simulation on a genotype phentoype map.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

# Figure out if we are using c or python wright fisher engine
#try:
#from .wright_fisher_engine_ext import wf_engine
#except ImportError:
#    from .wright_fisher_engine_python import wf_engine
from .wright_fisher_engine_python import wf_engine

import gpmap

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import multiprocessing as mp
import warnings, os

def _wf_engine_thread(args):
    """
    Run a wright fisher engine on a thread.

    Parameters
    ----------
    args : tuple
        arg[0] is queue for storing results. if None, return results directly.
        arg[1] is index uniquely identifying thread run.
        arg[2] is simulation data type
        arg[3] is initial population array
        arg[4] is number of steps to run
        arg[5] is number of genoytpes

        Remaining args are sent to wf_engine (no error checking).
    """

    # Parse front part of args tuple
    queue = args[0]
    index = args[1]
    sim_dtype = args[2]
    initial_pop = args[3]
    num_steps = args[4]
    num_genotypes = args[5]

    # Make pops array to store results
    pops = np.zeros((num_steps+1,num_genotypes),dtype=sim_dtype)
    pops[0] = initial_pop

    # Construct args to send to wf_engine, making appopriate pops array
    wf_args = list(args[4:])
    wf_args.append(pops)
    wf_args = tuple(wf_args)

    # Do calculation
    pops = wf_engine(*wf_args)

    # If multithreaded (queue not None), append to queue. Otherwise, just
    # return
    if queue is None:
        return pops
    else:
        queue.put(pops)

def simulate(gpm,
             mutation_rate,
             num_steps=1000,
             pop_size=100,
             fitness_column="fitness",
             initial_pop_column=None,
             num_replicate_sims=1,
             num_threads=1):
    """
    Parameters
    ----------
    gpm : GenotypePhenotypeMap
        genotype phenotype map to use for the simulation
    mutation_rate : float
        how probable is it that a genotype mutates over a generation?
    num_steps : int
        number of steps to run the simulation (must be >= 0)
    pop_size : int
        population size. must be int > 0. if initial_pop_column is set, this
        overrides the population size
    fitness_column : str
        column in gpm.data that has relative fitness values for each genotype
    initial_pop_column : str
        column in gpm.data that has initial population for each genotype. If
        None, assign the wildtype genotype a population pop_size and set rest
        to 0.
    num_replicate_sims : int
        number of replicate simulations to run. must be int > 0.
    num_threads : int
        number of threads to run (one simulation per thread). if None, use all
        available cpus. If set, must be an int > 0.

    Returns
    -------
    results : np.ndarray or list
        if num_replicate_sims == 1 (default), return num_steps + 1 x num_genotypes
        array with genotype counts over steps. If num_replicate_sims > 1, return
        a list of arrays, one for each replicate simulation.
    """

    # -------------------------------------------------------------------------
    # Parse arguments and validate sanity
    # -------------------------------------------------------------------------

    # Check gpm instance
    if not isinstance(gpm,gpmap.GenotypePhenotypeMap):
        err = "gpm must be a gpmap.GenotypePhenotypeMap instance\n"
        raise TypeError(err)

    # Look for gpm.data dataframe
    try:
        if not isinstance(gpm.data,pd.DataFrame):
            raise AttributeError
    except (AttributeError,TypeError):
        err = "gpm must have .data attribute that is a pandas DataFrame\n"
        raise ValueError(err)

    # Look for gpm.neighbors dataframe
    try:
        if not isinstance(gpm.neighbors,pd.DataFrame):
            raise AttributeError

        gpm.neighbors.loc[:,"source"]
        gpm.neighbors.loc[:,"target"]

    except (KeyError,AttributeError):
        err = "gpm must have .neighbors attribute that is a pandas\n"
        err += "DataFrame with source and target columns. Have you run\n"
        err += "gpm.get_neighbors()?\n"
        raise ValueError(err)

    # Check mutation_rate
    try:
        mutation_rate = float(mutation_rate)
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError
    except (ValueError,TypeError):
        err = "mutation_rate must be a float >= 0 and <= 1.\n"
        raise ValueError(err)

    # Check number of steps
    try:
        num_steps = int(num_steps)
        if num_steps < 0:
            raise ValueError
    except (ValueError,TypeError):
        err = "num_steps must be an integer >= 0.\n"
        raise ValueError(err)

    # Get initial population vector
    initial_pop = None
    if initial_pop_column is not None:

        try:
            initial_pop = np.array(np.round(gpm.data.loc[:,initial_pop_column],0),dtype=int)
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
        pop_size  = np.sum(initial_pop)

    # Check pop_size
    try:
        pop_size = int(pop_size)
        if pop_size < 1:
            raise ValueError
    except (ValueError,TypeError):
        err = "pop_size must be an integer > 0.\n"
        raise ValueError(err)

    # Get fitness data
    try:
        fitness = np.array(gpm.data.loc[:,fitness_column],dtype=float)
        if np.min(fitness) < 0:
            raise ValueError
        if np.sum(np.isnan(fitness)) > 0:
            raise ValueError
    except KeyError:
        err = f"fitness_column '{fitness_column}' not in gpm.data\n"
        err += "dataframe\n"
        raise KeyError(err)
    except (TypeError,ValueError):
        err = "fitness_column must point to a column in gpm.data that can\n"
        err += "be coerced as a float, where the minimum is >= 0, and that\n"
        err += "does not have nan.\n"
        raise ValueError(err)

    # If we have not made an initial_pop array above, make one with the wildtype
    # genotype as pop_size and all other genotypes as 0
    if initial_pop is None:

        initial_pop = np.zeros(len(gpm.data),dtype=int)

        err = None
        try:
            wt_index = np.arange(len(gpm.data),dtype=int)[gpm.data.name == "wildtype"]
        except AttributeError:
            err = "gpm.data should have a 'name' column.\n"
        except IndexError:
            err = "gpm.data 'name' column does not contain 'wildtype'.\n"

        if err is not None:
            err += "If no initial_pop_column is specified, the 'wildtype'\n"
            err += "genotype is assigned a population of 'pop_size' and all\n"
            err += "other genotypes are assigned population of 0.\n"
            raise ValueError(err)

        initial_pop[wt_index] = pop_size

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

    # -------------------------------------------------------------------------
    # Set up values used to run calculation
    # -------------------------------------------------------------------------

    # How many individuals to mutate each generation
    num_to_mutate = int(np.round(mutation_rate*pop_size,0))

    # Get genotype loc indexes
    genotype_indexes = np.array(gpm.data.loc[:,"name"].index,dtype=int)

    # Dictionary for converting dataframe loc indexes to iloc indexes. Sims are
    # going to be done using iloc indexes for efficiency; however, neighbors
    # .source and .target use loc indexing.
    loc_to_iloc = dict(zip(genotype_indexes,
                           np.arange(len(genotype_indexes),dtype=int)))

    # Create list of numpy arrays containing neighbors. These use the iloc
    # indexing scheme
    neighbors = []
    for g in genotype_indexes:

        iloc_g = loc_to_iloc[g]

        # Non-self neighbors
        mask = np.logical_and(gpm.neighbors.source == g,
                              gpm.neighbors.target != g)
        neighbor_list = []
        for n in gpm.neighbors.target[mask]:
            iloc_n = loc_to_iloc[n]
            neighbor_list.append(iloc_n)

        neighbors.append(np.array(neighbor_list,dtype=int))

    # array containing length of each neighbors array.
    num_neighbors = np.array([len(n) for n in neighbors],dtype=int)

    # Figure out what sort of array to use to store the populations
    sim_dtype = None
    for d in [np.uint8,np.uint16,np.uint32,np.uint64]:
        if np.iinfo(d).max >= pop_size:
            sim_dtype = d
            break
    if sim_dtype is None:
        err = f"population size '{pop_size}' > max allowed '{np.iinfo(np.uint64).max}'\n'"
        raise OverflowError(err)

    # Get number of genotypes
    num_genotypes = len(genotype_indexes)

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

        # Make args list
        args = [queue,i,sim_dtype,initial_pop,
                num_steps,num_genotypes,pop_size,num_to_mutate,
                fitness,num_neighbors,neighbors]

        # Append this to args
        all_args.append(tuple(args))

    # If only one thread, call without Pool/queue complexity
    if num_threads == 1:
        results = []
        for a in all_args:
            results.append(_wf_engine_thread(a))

    # If more than one thread, call under control of pool
    else:
        with mp.Pool(num_threads) as pool:

            # pool.imap() runs a function on elements in iterable, filling threads
            # as each job finishes. (Calls _wf_engine_thread on every args tuple in
            # all_args).

            list(pool.imap(_wf_engine_thread,all_args))

        # Get pops arrays from queue object
        results = []
        while not queue.empty():
            results.append(queue.get())

    # If we are running a single replicate, just return the sim
    if num_replicate_sims == 1:
        return results[0]

    # Otherwise, return a list of sims
    return results

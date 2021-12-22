__description__ = \
"""
"""
__date__ = "2021-12-10"
__author__ = "Clara Rehmann"

import gpvolve.check as check
from gpmap import GenotypePhenotypeMap

import pyslim, tskit, gpmap
import numpy as np, pandas as pd

import os

def make_stepframe(stpdict,num_generations):

    # establish dataframe for transition counts each generation
    stepframe = pd.DataFrame(columns = [str(k) for k in stpdict.keys()], index = range(num_generations+1))
    stepframe = stepframe.fillna(0)

    # fill counts at each generation
    for key, value in stpdict.items():
        for v in value:
            stepframe[str(key)][v] += 1

    return stepframe

def make_fluxdict(gpm, genotypehistory):
    """
    Parse histdict output from get_hist() into a dictionary of scaled edge weights.

    PARAMETERS:
    -----------
    gpm (GenotypePhenotypeMap object) : gpmap used to run the simulation
    genotypehistory (dict) : histdict output from get_hist()

    RETURNS:
    --------
    fluxdict (dict) : nested dictionary of steps between ancestral and derived genotype,
        edge transitions at that step, and their scaled # of observations (weights)
        format = {step: {(ancestral, derived): weight}}

    """
    gpm.get_neighbors()
    pwy_count = genotypehistory.get(np.where(gpm.data.n_mutations == gpm.length)[0][0])
    pwys = list(pwy_count.keys())
    binary = np.array([list([int(s) for s in g]) for g in gpm.data.binary])

    fluxdict = {}

    for i in range(gpm.length): # for each required step between ancestral and derived
        counts = {}
        for p in pwys: # for unique path taken, which step did they take here?
            # establish binary derived genotype
            gt_bin_dev = [0]*gpm.length
            for site in range(i, -1, -1):
                gt_bin_dev[p[site]] = 1
            if i==0:
                gt_bin_anc = [0]*gpm.length # binary ancestral genotype
            else:
                gt_bin_anc = [0]*gpm.length
                for site in range(i-1, -1, -1):
                    gt_bin_anc[p[site]] = 1
            gt_dev = np.where((binary == gt_bin_dev).all(axis=1))[0][0] # derived genotype gpmap index
            gt_anc = np.where((binary == gt_bin_anc).all(axis=1))[0][0] # ancestral genotype gpmap index

            # number of individuals who took that (anc, dev) step
            counts.update({(gt_anc,gt_dev):(int(0 if counts.setdefault((gt_anc,gt_dev)) is None
                                            else counts.setdefault((gt_anc,gt_dev)))
                                        +pwy_count[p])})
        # scale
        s = sum(counts.values())
        for key, value in counts.items():
            counts[key] = value / s
        fluxdict.update(counts)

    return fluxdict

def check_simulation_parameter_sanity(gpm,
                                      max_generations,
                                      mutation_rate,
                                      population_size,
                                      fitness_column):

    # Check gpm sanity
    check.gpm_sanity(gpm)
    gpm.get_neighbors()

    # Check number of steps
    try:
        max_generations = int(max_generations)
        if max_generations < 0:
            raise ValueError
    except (ValueError,TypeError):
        err = "num_generations must be an integer >= 0.\n"
        raise ValueError(err)

    # Check mutation_rate
    try:
        mutation_rate = float(mutation_rate)
        if mutation_rate < 0 or mutation_rate > 1:
            raise ValueError
    except (ValueError,TypeError):
        err = "mutation_rate must be a float >= 0 and <= 1.\n"
        raise ValueError(err)

    # Check population_size
    try:
        population_size = int(population_size)
        if population_size < 1:
            raise ValueError
    except (ValueError,TypeError):
        err = f"population_size must be an integer > 0.\n"
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
        err += "be coerced as a float, where the minimum is >= 0 and that does\n"
        err += "not have nan.\n"
        raise ValueError(err)

    return gpm, max_generations, mutation_rate, population_size, fitness

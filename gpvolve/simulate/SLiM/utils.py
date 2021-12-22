__description__ = \
"""
"""
__date__ = "2021-12-10"
__author__ = "Clara Rehmann"

import pyslim, tskit, gpmap
from pathlib import Path
from gpmap import GenotypePhenotypeMap

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

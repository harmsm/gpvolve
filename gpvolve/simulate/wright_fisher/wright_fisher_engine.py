__description__ = \
"""
Code for simulating evolution using a Wright-Fisher process.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-15"

import numpy as np

import warnings

import gpvolve.simulate.wright_fisher.wright_fisher_engine_python as py
try:
    import gpvolve.simulate.wright_fisher.wright_fisher_engine_cython as cy
    cy_available = True
except ImportError:

    cy_available = False
    w = "Could not load cython version of wright_fisher_engine. Falling\n"
    w += "back on python version (same functionality, much slower)\n."
    warnings.warn(w)

def wf_engine(pops,
              mutation_rate,
              fitness,
              neighbor_slicer,
              neighbors,
              use_cython=True):
    """
    Simulate evolution across a GenotypePhenotypeMap using a Wright-Fisher
    process. This engine should generally be called via the .simulate function.

    Parameters
    ----------
    pops : numpy.ndarray
        num_steps + 1 by num_genotypes 2D int array that stores the population
        of each genotype for each step in the simulation. The first row holds
        the initial population of all genotypes.
    mutation_rate : float
        mutation rate for each generation
    fitness : numpy.ndarray
        num_genotypes-long float array containing fitness of each genotype
    neighbor_slicer : numpy.ndarray
        num_genotypes-long int array containing number of neighbors accessible
        for each genotype (excluding self)
    neighbors : numpy.ndarray
        1D numpy int array storing a jagged array with neighbors for each
        genotype. neighbor_slicer is used to look up where each genotype's
        neighbors are in this array
    use_cython : bool
        use faster cython implementation if available.

    Returns
    -------
    pops : numpy.ndarray
        num_steps + 1 x num_genotypes 2D int array that stores the population
        of each genotype for each step in the simulation.
    """

    if use_cython:
        return cy.wf_engine_cython(pops,
                                   mutation_rate,
                                   fitness,
                                   neighbor_slicer,
                                   neighbors)
    else:
        return py.wf_engine_python(pops,
                                   mutation_rate,
                                   fitness,
                                   neighbor_slicer,
                                   neighbors)

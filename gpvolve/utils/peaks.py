from matplotlib import pyplot as plt
import numpy as np


def find_peaks(gpm, return_peaks=False):
    """
    Finds local maxima and adds a binary column to neighbors
    dataframe showing whether given source node is a peak.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object
    return_peaks : Whether to return binary list indicating
                   nodes are a peak or not.

    Returns
    -------
    is_peak : Binary list indicating whether node with given
              index is a peak. Only returned if return_peaks
              is set to 'True'.
    """
    gpm.get_neighbors()
    is_peak = []

    for count, i in enumerate(gpm.neighbors.source):

        # Phenotypes of everybody that is adjacent to genotype i
        adjacent = list(gpm.data.loc[gpm.neighbors.loc[gpm.neighbors.source == i, "target"], "phenotype"])

        # Is genotype i a peak? False = No
        if np.max(adjacent) <= gpm.data.loc[i, 'phenotype']:
            # If it's a peak, turn binary entry into True
            is_peak.append(True)
        else:
            is_peak.append(False)

    gpm.neighbors['peaks'] = is_peak

    if return_peaks:
        return is_peak


def soft_peaks(gpm, error, return_peaks=False):
    """
    Finds local maxima and adds a binary column to neighbors
    dataframe showing whether given source node is a peak.

    Takes into account error, e.g. if fitness1 has one neighbor
    (fitness2) with higher fitness, fitness1 is still considered a peak if
    fitness1 + error is higher than or equal to fitness2 - error.
    Can return a plot of the local peaks found.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object
    error : Error in phenotypic measurement.
    return_peaks : Whether to return binary list indicating
                   nodes are a peak or not.

    Returns
    -------
    is_peak : Binary list indicating whether node with given
              index is a peak. Only returned if return_peaks
              is set to 'True'.
    """
    gpm.get_neighbors()
    is_peak = []

    for count, i in enumerate(gpm.neighbors.source):

        # Phenotypes of everybody that is adjacent to genotype i
        adjacent = list(gpm.data.loc[gpm.neighbors.loc[gpm.neighbors.source == i, "target"], "phenotype"])

        # Is genotype i a peak? False = No
        if np.max(adjacent) - error <= gpm.data.loc[i, 'phenotype'] + error:
            # If it's a peak, turn binary entry into True
            is_peak.append(True)
        else:
            is_peak.append(False)

    gpm.neighbors['soft_peaks'] = is_peak

    if return_peaks:
        return is_peak


def eigenvalues(T):
    """
    Get the eigenvalues of the transition matrix.

    Parameters:
    -----------
    T : Stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    Returns:
    --------
    eigvals : list of eigenvalues.
    """
    eigvals = np.linalg.eigvals(T)

    return eigvals


def eigenvectors(T):
    """
    Get the eigenvectors of the transition matrix.

    Paremeters:
    T : Stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    Returns:
    --------
    eigv : eigenvectors
    -----------
    T : a transition probability matrix
    """
    eigv = np.linalg.eig(T)[1]

    return eigv
from matplotlib import pyplot as plt
import numpy as np


def find_peaks(gpm, name_of_phenotype='phenotype', return_plot=False):
    """
    Finds local maxima and add to underlying dataframe for any phenotypic
    data passed to function (i.e.,'fitness','phenotype').
    Can return a plot of the local peaks found.

    Parameters:
    -----------
    gpm : GenotypePhenotypeMap object
    """
    gpm.data['peaks'] = gpm.data.loc[:, name_of_phenotype][
        (gpm.data.loc[:, name_of_phenotype].shift(1) < gpm.data.loc[:, name_of_phenotype]) & (
                    gpm.data.loc[:, name_of_phenotype].shift(-1) < gpm.data.loc[:, name_of_phenotype])]

    if return_plot:
        # Plot results
        plt.scatter(plt.data.index, plt.data['peaks'], c='g')
        gpm.data.loc[:, name_of_phenotype].plot()
    else:
        pass


def soft_peaks(gpm, error, name_of_phenotype='phenotype', return_plot=False):
    """
    Finds local maxima and add to underlying dataframe for any phenotypic
    data passed to function (i.e.,'fitness','phenotype').
    Takes into account error, e.g. if fitness1 has one neighbor
    (fitness2) with higher fitness, fitness1 is still considered a peak if
    fitness1 + error is higher than or equal to fitness2 - error.
    Can return a plot of the local peaks found.

    Parameters:
    -----------
    gpm : GenotypePhenotypeMap object
    """
    gpm.data['soft_peaks'] = gpm.data.loc[:, name_of_phenotype][
        (gpm.data.loc[:, name_of_phenotype].shift(1)+error < gpm.data.loc[:, name_of_phenotype]-error) & (
                    gpm.data.loc[:, name_of_phenotype].shift(-1)+error < gpm.data.loc[:, name_of_phenotype]-error)]

    if return_plot:
        # Plot results
        plt.scatter(plt.data.index, plt.data['soft_peaks'], c='g')
        gpm.data.loc[:, name_of_phenotype].plot()
    else:
        pass


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
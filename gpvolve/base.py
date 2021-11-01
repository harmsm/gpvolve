from gpvolve.check import gpm_sanity
from gpvolve.utils import flatten_neighbors
from gpvolve.markov import utils
from gpvolve.phenotype_to_fitness import linear, sigmoid, exponential, step
from matplotlib import pyplot as plt
import numpy as np


def apply_fitness_function(gpm, fitness_function, **params):
    """
    Compute fitness values from a user-defined phenotype-fitness function.
    A few basic functions can be found in gpsolve.fitness. For a direct
    mapping of phenotype to fitness, use one_to_one without additional
    parameters.

    Parameters
    ----------
    gpm : a GenotypePhenotypeMap object with with a phenotype values specified
            beforehand.
    fitness_function: function.
        A python function that takes phenotypes and additional parameters
        (optional) and returns a list of fitness values (type=float).

    Returns
    -------
    Nothing: None
        The computed fitness values are automatically stored in underlying
        dataframe of gpm object, (i.e. gpm.data.loc[:, "fitness"]).
    """
    # Check minimum requirements (is it gpmap object, does it have tmatrix)
    gpm_sanity(gpm)

    # Calculate fitness parameters
    fitness = fitness_function(gpm.data.loc[:, "phenotype"], **params)

    # Add fitness values column to gpm.data pandas data frame.
    gpm.data.loc[:, "fitness"] = fitness

    return gpm


def build_transition_matrix(gpm, fixation_model='moran', fitness_model='linear', **params):
    """
    Calculate fixation probability along all edges and build transition
    matrix. Tries to use cython first, and falls back to python if needed.

    Parameters
    ----------
    gpm : GenotypePhenotypeMap object
    fixation_model : Fixation model used for transition matrix.
    fitness_model : If fitness values are not present, generate them using a given
                    model (options are linear, sigmoid, exponential, and step).
    Returns
    -------
    T : Stochastic transition matrix for evolution across between genotypes
    given the fitness of each genotype, their connectivity, the population size
    and a fixation model.

    """
    # Get neighbors and check minimum requirements
    gpm.get_neighbors()
    gpm_sanity(gpm)

    # Get parameters
    if 'fitness' in list(gpm.data):
        fitness = np.array(gpm.fitness)
    if fitness_model == 'linear':
        apply_fitness_function(gpm, fitness_function=linear)
        fitness = np.array(gpm.fitness)
    elif fitness_model == 'exponential':
        apply_fitness_function(gpm, fitness_function=exponential)
        fitness = np.array(gpm.fitness)
    elif fitness_model == 'sigmoid':
        apply_fitness_function(gpm, fitness_function=sigmoid)
        fitness = np.array(gpm.fitness)
    else:
        print("Unexpected name for phenotype-to-fitness function. Valid"
              "options include: 'linear','exponential','sigmoid',and 'step'.")
    neighbor_slicer, neighbors = flatten_neighbors(gpm)

    try:
        use_cython = True
        t = utils.generate_tmatrix(fitness,
                                   neighbor_slicer,
                                   neighbors,
                                   fixation_model=fixation_model,
                                   use_cython=use_cython)
    except ImportError:
        use_cython = False
        t = utils.generate_tmatrix(fitness,
                                   neighbor_slicer,
                                   neighbors,
                                   fixation_model=fixation_model,
                                   use_cython=use_cython)

    return t


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

    Paremeters:
    -----------
    T : a transition probability matrix
    """
    eigvals = np.linalg.eigvals(T)

    return eigvals


def eigenvectors(T):
    """
    Get the eigenvectors of the transition matrix.

    Paremeters:
    -----------
    T : a transition probability matrix
    """
    eigv = np.linalg.eig(T)[1]

    return eigv

###################################################################################################
# Incomplete functions or functions with questionable usefulness
# ------>> Check which can be converted to utils
# or which are already present as utils
###################################################################################################
#
# def timescales(self):
#     "These are gpvolve's additional properties"
#     return self._gpv
#
# def timescales(self):
#     """
#     Get the relaxation timescales corresponding to the eigenvalues in
#     arbitrary units.
#     """
#     if isinstance(self._timescales, np.ndarray):
#         return self._timescales
#     else:
#         self._timescales = timescales(self.transition_matrix)
#         return self._timescales
#
#
# def forward_committor(self, source=None, target=None):
#     """
#     If no new source and target provided, return existing forward committor
#     values, else, calculate them.
#     """
#     if not source and not target:
#         if isinstance(self._forward_committor, np.ndarray):
#             return self._forward_committor
#
#         else:
#             raise Exception('No forward committor calculated and no source and target provided.')
#
#     elif source and target:
#         self._forward_committor = self.calc_committor(self.transition_matrix, source, target,
#                                                       forward=True)
#         return self._forward_committor
#
#
# def backward_committor(self, source=None, target=None):
#     """
#     If no new source and target provided, return existing backward committor
#     values, else, calculate them.
#     """
#     if not source and not target:
#         if isinstance(self._backward_committor, np.ndarray):
#             return self._backward_committor
#
#         else:
#             raise Exception('No forward committor calculated and no source and target provided.')
#
#     elif isinstance(source, list) and isinstance(target, list):
#         self._backward_committor = self.calc_committor(self.transition_matrix, source, target,
#                                                        forward=False)
#         return self._backward_committor
#
#
# def calc_committor(self, T, source, target, forward=None):
#     """
#     Calculate forward or backward committor for each node between source
#     and target.
#
#     Parameters
#     ----------
#     T : 2D numpy.ndarray.
#         Row stochastic transition matrix.
#
#     source : list.
#         Source of probability flux. Committor value i will be the
#         probability of leaving source and reaching node i before reaching
#         target or source again.
#
#     target : list.
#         Sink of probability flux. Committor value i will be the probability
#         of reaching target from node i before reaching source.
#
#     forward : bool.
#         If True, forward committor is calculated. If False, backward
#         committor is calculated.
#
#     Returns
#     -------
#     committor : 1D numpy.ndarray.
#         Committor values in order of transition matrix.
#     """
#     committor = self.calc_committor(T, source, target, forward=forward)
#     return committor

#
# def step_function(gpm):
#     """
#     A function that bins phenotypes and allows one to define neutral
#     networks in g-p-maps with continuous phenotypes
#     """
#     pass
#
#
# def neutral_network(gpm):
#     """
#     Find neutral network. Look for connected components among phenotypes
#     with same value or value within the same pre-defines bin.
#     """
#     pass
# def stationary_distribution(gpm):
#     """
#     The stationary distribution of the genotype-phenotype-map.
#     """
#     stat_dist = nx.get_node_attributes(gpm, name="stationary_distribution")
#     if stat_dist:
#         return stat_dist
#     else:
#         stat_dist = {node: prob for node, prob in enumerate(stationary_distribution(self.transition_matrix))}
#         nx.set_node_attributes(self, name="stationary_distribution", values=stat_dist)
#
#         return nx.get_node_attributes(self, name="stationary_distribution")


# def stationary_distribution(self, stat_dist_raw):
#     if isinstance(stat_dist_raw, dict):
#         nx.set_node_attributes(self, name="stationary_distribution", values=stat_dist_raw)
#     else:
#         stat_dist = {node: prob for node, prob in enumerate(stat_dist_raw)}
#         nx.set_node_attributes(self, name="stationary_distribution", values=stat_dist)

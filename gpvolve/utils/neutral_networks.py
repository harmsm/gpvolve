
###################################################################################################
# Incomplete functions or functions with questionable usefulness
# ------>> Check which can be converted to base
# or which are already present as base
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

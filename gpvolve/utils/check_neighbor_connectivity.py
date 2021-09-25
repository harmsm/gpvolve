
from gpvolve import check
import warnings
import numpy as np

def check_neighbor_connectivity(gpm,warn=False):
    """
    Check for poorly connected genotypes, returning genotypes that are targets
    but not sources, sources but not targets, and neither sources nor targets.
    This excludes self edges. This only includes edges for which
    gpm.neighbors.include == True

    Parameters
    ----------
    gpm : gpmap.GenotypePhenotypeMap
        GenotypePhenotypeMap with neighbors dataframe.
    warn : bool
        whether or not to throw a warning if there are poorly connected
        genotypes.

    Returns
    -------
    not_a_source : np.ndarray
        genotypes that are targeted by non-self but are not ever a non-self
        source. These will act as sinks/traps. (int, index in gpm.data)
    not_a_target : np.ndarray
        genotypes that are not targeted by non-self but act as a non-self
        source. These will never be visited unless initially populated (int,
        index in gpm.data)
    isolated : np.ndarray
        genotypes that are neither non-self targets nor non-self sources. These
        will never be visited unless initially populated. Any individual that
        starts with this genotype will never mutate. (int, index in gpm.data)
    """

    check.gpm_sanity(gpm)

    genotypes = np.array(gpm.data.index,dtype=int)

    try:
        include_mask = gpm.neighbors.loc[:,"include"] == True
    except KeyError:
        include_mask = np.ones(len(gpm.neighbors),dtype=bool)

    non_self_mask = gpm.neighbors.loc[:,"target"] != gpm.neighbors.loc[:,"source"]
    non_self_mask = np.logical_and(include_mask,non_self_mask)

    source = np.unique(gpm.neighbors.loc[non_self_mask,"source"])
    target = np.unique(gpm.neighbors.loc[non_self_mask,"target"])
    all_neighbors = np.union1d(source,target)

    isolated = np.setdiff1d(genotypes,all_neighbors)
    not_a_source = np.setdiff1d(np.setdiff1d(genotypes,source),isolated)
    not_targeted = np.setdiff1d(np.setdiff1d(genotypes,target),isolated)

    num_not_a_source = not_a_source.shape[0]
    num_not_targeted = not_targeted.shape[0]
    num_isolated = isolated.shape[0]

    w = None
    if num_isolated > 0 or num_not_a_source > 0 or num_not_targeted > 0:
        w = "Some genotypes do not have non-self neighbors. Genotypes with no\n"
        w += "neighbors will be isolated, either never visited or trapping\n"
        w += "individuals that start with that genotype. To speed the \n"
        w += "calculation, consider removing individuals with no neigbors.\n"
        w += "Genotypes that only act as sources but not targets will never\n"
        w += "be visited unless they are in the initial population; genotypes\n"
        w += "that act as targets but never sources are sinks that trap\n"
        w += "individuals who acquire that genotype.\n\n"
        if num_isolated > 0:
            w += f"    number isolated: {num_isolated}\n"
        if num_not_targeted > 0:
            w += f"    number never targeted: {num_not_targeted}\n"
        if num_not_a_source > 0:
            w += f"    number never a source: {num_not_a_source}\n"
        w += "\n"

    if w is not None and warn:
        warnings.warn(w)

    return not_a_source, not_targeted, isolated

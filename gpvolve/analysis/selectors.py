from itertools import combinations
from gpmap.utils import hamming_distance
from gpvolve import monotonic_incr


def adaptive_paths(paths, fitnesses):
    
    adaptive_paths = []
    for path in paths:
        if monotonic_incr(path, fitnesses):
            adaptive_paths.append(path)

    return adaptive_paths


def forward_paths(paths, msm, source, target):

    fp = []

    comb = combinations(source, target)
    min_dist = hamming_distance(msm.gpm.data.binary[source[0]], msm.gpm.data.binary[target[0]])

    for path in paths:
        if len(path) - 1 == min_dist:
            fp.append(path)

    return fp


def paths_that_contain(paths, nodes, bool_and=False):
    """Return paths that contain at least one or all of the nodes.

    Parameters
    ----------
    paths : list.
        list of paths where each path is a tuple of integers. Example : [(1,2,3), (1,3,4)]

    nodes : list.
        List of nodes that the paths are going to be searched for.

    bool_and : bool.
        If True, the sufficient requirement is that all nodes are in a path. If False, the sufficient
        requirement is that at least one of the nodes is in a path.

    Returns
    -------
    paths_ : list.
        list of paths that contain at least one or all of the nodes in 'nodes'.
    """
    paths_ = []

    # Must contain all nodes.
    if bool_and:
        for path in paths:
            contains = True
            for node in nodes:
                if node in path:
                    continue
                else:
                    contains = False
                    break
            # If no breaks happen, all nodes are in path. (Sufficient requirement.)
            if contains:
                paths_.append(path)

    # Must contain at least one of the nodes.
    elif not bool_and:
        for path in paths:
            for node in nodes:
                if node in path:
                    paths_.append(path)
                    break
    return paths_


def paths_that_do_not_contain(paths, nodes, bool_and=True):
    """Return paths that do not contain at least one or all of the nodes.

    Parameters
    ----------
    paths : list.
        list of paths where each path is a tuple of integers. Example : [(1,2,3), (1,3,4)]

    nodes : list.
        List of nodes that the paths are going to be searched for.

    bool_and : bool.
        If True, the sufficient requirement is that all nodes are not in a path. If False, the sufficient
        requirement is that at least one of the nodes is not in a path.

    Returns
    -------
    paths_ : list.
        list of paths that do not contain at least one or all of the nodes in 'nodes'.
    """
    paths_ = []

    # Must not contain all nodes.
    if bool_and:
        for path in paths:
            doesnt_contain = True
            for node in nodes:
                if node not in path:
                    continue
                else:
                    doesnt_contain = False
                    break
            # If no breaks, all nodes are not in path. (Sufficient requirement.)
            if doesnt_contain:
                paths_.append(path)

    # Must not contain at least one of the nodes.
    elif not bool_and:
        for path in paths:
            for node in nodes:
                if node not in path:
                    paths_.append(path)
                    break
    return paths_


def fraction_of_paths(paths_dict, fraction=1.):
    """Get fraction of strongest paths whose probability sum to a certain fraction.

    Parameters
    ----------
    paths_dict : dict.
        Dictionary of paths (tuple) and probabilities (float). Should be normalized, otherwise fraction might not
        actually get the fraction.

    fraction : float/int (default=1.).
        Find most likely paths which have a summed probability of at least 'fraction'.

    Returns
    -------
    new_dict : dict.
        Dictionary of most likely paths which have a summed probability of at least 'fraction'.
    """
    # Sort paths and probababilties from highest to lowest probability.
    sorted_probs, sorted_paths = zip(*sorted(zip(paths_dict.values(), paths_dict.keys()), reverse=True))
    probsum = 0

    for i, prob in enumerate(sorted_probs):
        probsum += prob

        # Enough paths to reach fraction?
        if probsum >= fraction:
            new_dict = dict(zip(sorted_paths[:i + 1], sorted_probs[:i + 1]))
            return new_dict
    # Not enough paths in whole dictionary to reach fraction.
    new_dict = paths_dict
    return new_dict

def get_sub_paths(paths, start, end):
    """
    Get part of path between 'start' node and 'end' node.
    Parameters
    ----------
    paths : dict.
        dict of paths and probabiltiites. Paths have to tuples of integers.

    start : any single list element.
        Element with which sub-path should start.

    end : any single list element.
        Element with which sub-path should start.

    Returns
    -------
    subpaths : dict.
        Dict of subpaths. Some subpaths might be identical, which will be treated as one and the probabilities summed.

        """
    subpaths = {}
    for path, prob in paths.items():
        p = list(path)
        try:
            s = p.index(start)
        except ValueError:
            raise Exception("%s not in path %s" % (s, path))
        try:
            e = p.index(end)
        except ValueError:
            raise Exception("%s not in path %s" % (e, path))

        try:
            subpaths[tuple(p[s:e + 1])] += prob
        except KeyError:
            subpaths[tuple(p[s:e + 1])] = prob

__description__ = \
    """
Functions for checking sanity of inputs.
"""
__author__ = "Michael J. Harms"
__date__ = "2021-09-25"

import gpmap
import pandas as pd


def gpm_sanity(gpm):
    """
    Check sanity of a gpmap.GenotypePhenotypeMap instance for gpvolve, throwing
    error if it's bad.

    Parameters
    ----------
    gpm : ?
        possible gpmap.GenotypePhenotypeMap to validate

    Returns
    -------
    None
    """

    # Check gpm instance
    if not isinstance(gpm, gpmap.GenotypePhenotypeMap):
        err = "gpm must be a gpmap.GenotypePhenotypeMap instance\n"
        raise TypeError(err)

    # Look for gpm.data dataframe
    try:
        if not isinstance(gpm.data, pd.DataFrame):
            raise AttributeError
    except (AttributeError, TypeError):
        err = "gpm must have .data attribute that is a pandas DataFrame\n"
        raise ValueError(err)

    # Look for gpm.neighbors dataframe
    try:
        if not isinstance(gpm.neighbors, pd.DataFrame):
            raise AttributeError

        gpm.neighbors.loc[:, "source"]
        gpm.neighbors.loc[:, "target"]

    except (KeyError, AttributeError):
        err = "gpm must have .neighbors attribute that is a pandas\n"
        err += "DataFrame with source and target columns. Have you run\n"
        err += "gpm.get_neighbors()?\n"
        raise ValueError(err)

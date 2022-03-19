import pytest
import gpvolve.check as check
import gpmap
import numpy as np
from gpvolve.utils import find_peaks


def test_check_gpm():
    with pytest.raises(TypeError):
        check.gpm_sanity("stupid")

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00", "10", "01", "11"],
                                     fitness=[0.1, 0.2, 0.2, 0.3])
    gpm._data = "stupid"
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00", "10", "01", "11"],
                                     fitness=[0.1, 0.2, 0.2, 0.3])
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    gpm._neighbors = "stupid"
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    # Screw up target column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00", "10", "01", "11"],
                                     fitness=[0.1, 0.2, 0.2, 0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["target"], axis=1, inplace=True)
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    # Screw up source column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00", "10", "01", "11"],
                                     fitness=[0.1, 0.2, 0.2, 0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["source"], axis=1, inplace=True)
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    # This should work
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00", "10", "01", "11"],
                                     fitness=[0.1, 0.2, 0.2, 0.3])
    gpm.get_neighbors()
    check.gpm_sanity(gpm)



# def test_find_peaks():
#     # Generate a random map
#     gpm = gpmap.simulate.generate_random()
#
#     # Find maximum values
#     find_peaks(gpm)
#
#     # Check that 'peaks' column was created and added to dataframe
#     assert 'peaks' in list(gpm.neighbors)
#
#     # Check that maximum value of phenotypes is one of the peaks
#     # Collect index of neighbors on main dataframe (source edge)
#     edges = []
#     for i, v in enumerate(gpm.neighbors.peaks):
#         if v:
#             edges.append(gpm.neighbors.loc[i, 'edge'][0])
#
#     # Check max phenotype is one of the peaks
#     assert np.max(gpm.data.loc[:, 'phenotype']) <= any(gpm.data.loc[edges, 'phenotype'])

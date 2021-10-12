
import gpmap
import numpy as np
import pytest
import gpvolve.utils as utils


def test_check_neighbor_connectivity():

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])

    # Bad gpm
    with pytest.raises(ValueError):
        utils.check_neighbor_connectivity(gpm)

    # Should work and not warn, connectivity fine
    gpm.get_neighbors()
    utils.check_neighbor_connectivity(gpm)

    # Warn because poor connectivity
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors(cutoff=0)
    with pytest.warns(UserWarning):
        utils.check_neighbor_connectivity(gpm,warn=True)

    # make sure no warning raised if no warning requested
    with pytest.warns(None) as record:
        not_a_source, not_targeted, isolated = utils.check_neighbor_connectivity(gpm,warn=False)
    assert not record
    assert len(not_a_source) == 0
    assert len(not_targeted) == 0
    assert len(isolated) == 4

    # Turn off 0 as a source.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors()
    mask = gpm.neighbors.source == 0
    pred_num_no_source = np.sum(mask)
    gpm.neighbors.loc[mask,"include"] = False

    not_a_source, not_targeted, isolated = utils.check_neighbor_connectivity(gpm,warn=False)
    assert np.array_equal(not_a_source,[0])
    assert len(not_targeted) == 0
    assert len(isolated) == 0

    # Turn off 1 as a target.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors()
    mask = gpm.neighbors.target == 1
    gpm.neighbors.loc[mask,"include"] = False

    not_a_source, not_targeted, isolated = utils.check_neighbor_connectivity(gpm,warn=False)
    assert len(not_a_source) == 0
    assert np.array_equal(not_targeted,[1])
    assert len(isolated) == 0

    # Turn off 2 as both a source *and* a target.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors()
    mask = gpm.neighbors.target == 2
    gpm.neighbors.loc[mask,"include"] = False
    mask = gpm.neighbors.source == 2
    gpm.neighbors.loc[mask,"include"] = False

    not_a_source, not_targeted, isolated = utils.check_neighbor_connectivity(gpm,warn=False)
    assert len(not_a_source) == 0
    assert len(not_targeted) == 0
    assert np.array_equal(isolated,[2])

    # Turn off 0 as a source, and turn off 2 as both a source *and* a target.
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[1.0,1.0,1.0,1.0])
    gpm.get_neighbors()
    mask = gpm.neighbors.source == 0
    gpm.neighbors.loc[mask,"include"] = False
    mask = gpm.neighbors.target == 2
    gpm.neighbors.loc[mask,"include"] = False
    mask = gpm.neighbors.source == 2
    gpm.neighbors.loc[mask,"include"] = False

    not_a_source, not_targeted, isolated = utils.check_neighbor_connectivity(gpm,warn=False)
    assert np.array_equal(not_a_source,[0])
    assert len(not_targeted) == 0
    assert np.array_equal(isolated,[2])

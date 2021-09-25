
import pytest
import gpvolve.check as check
import gpmap

def test_check_gpm():

    with pytest.raises(TypeError):
        check.gpm_sanity("stupid")

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm._data = "stupid"
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    gpm._neighbors = "stupid"
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    # Screw up target column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["target"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    # Screw up source column
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    gpm.neighbors.drop(labels=["source"],axis=1,inplace=True)
    with pytest.raises(ValueError):
        check.gpm_sanity(gpm)

    # This should work
    gpm = gpmap.GenotypePhenotypeMap(genotype=["00","10","01","11"],
                                     fitness=[0.1,0.2,0.2,0.3])
    gpm.get_neighbors()
    check.gpm_sanity(gpm)

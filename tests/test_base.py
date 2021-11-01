import pytest
import gpvolve.check as check
import gpmap
import numpy as np
from gpvolve.base import apply_fitness_function
from gpvolve.phenotype_to_fitness import linear
from gpvolve import find_peaks


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


def test_apply_fitness_function_linear():
    # Apply random parameters, anything between 1-10
    a = np.random.randint(1, 10)
    b = np.random.randint(1, 10)
    gpm = gpmap.simulate.generate_random()

    # Apply linear fitness function
    apply_fitness_function(gpm, linear, a=a, b=b)

    # Check that fitness are actually linear
    np.testing.assert_array_equal(gpm.data.loc[:, 'fitness'], a * gpm.data.loc[:, 'phenotype'] + b)


def test_find_peaks():
    # Generate a random map
    gpm = gpmap.simulate.generate_random()

    # Find maximum values
    find_peaks(gpm,name_of_phenotype='phenotype')

    # Check that 'peaks' column was created and added to dataframe
    assert 'peaks' in list(gpm.data)

    # Check that maximum value of phenotypes is one of the peaks
    assert np.max(gpm.data.loc[:, 'phenotype']) in list(gpm.data.loc[:, 'peaks'])
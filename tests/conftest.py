import pytest
from gpmap import GenotypePhenotypeMap
import numpy as np


@pytest.fixture(scope="module")
def gpvolve_gpm():

    wildtype = "AAA"

    genotypes = [
        "AAA",
        "AAB",
        "ABA",
        "BAA",
        "ABB",
        "BAB",
        "BBA",
        "BBB"
    ]

    mutations = {
        0: ["A", "B"],
        1: ["A", "B"],
        2: ["A", "B"],
    }
    phenotypes = np.random.rand(len(genotypes))
    gpm = GenotypePhenotypeMap(wildtype=wildtype,
                               genotype=genotypes,
                               phenotype=phenotypes)

    return gpm

@pytest.fixture(scope="module")
def number_data():

    return {"max_float":np.finfo(float).max,
            "tiny_float":np.finfo(float).tiny,
            "max_int":np.iinfo(int).max}

@pytest.fixture(scope="module")
def pop_gen_scenarios():

    scenarios = []
    for f1 in 10**np.arange(-10,11,1,dtype=float):
        for f2 in 10**np.arange(-10,11,1,dtype=float):
            for pop in 10**np.arange(0,10,dtype=int):
                scenarios.append((f1,f2,pop))

    return scenarios

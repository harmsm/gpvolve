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

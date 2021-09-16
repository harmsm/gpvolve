import pytest

from gpvolve import fitness_to_fixation

import numpy as np

import itertools

def test_sswm(number_data):

    def local_sswm(f1,f2):
        """
        Stupidly simple local sswm model without error checking or fancy numerical
        log stuff to make sure it's doing what it's supposed to do.
        """

        sij = (f2 - f1) / f1
        if sij < 0:
            sij = 0
        return 1 - np.exp(-sij)

    # Make sure it runs
    fitness_to_fixation.sswm(1,1)

    # Test some known answers
    args = [(0.1,1.0),(1.0,0.1)]
    for a in args:
        assert np.isclose(fitness_to_fixation.sswm(*a),
                          local_sswm(*a))

    # Test argument checking
    bad_args = [("stupid","strings"),
                (1,"bad"),("bad",1),
                (-1,1),(1,-1),(-1,-1),
                (0,1),(1,0)]
    for a in bad_args:
        with pytest.raises(ValueError):
            fitness_to_fixation.sswm(*a)

    # Test numerical overflow checking
    num_args = itertools.product([number_data["max_float"],
                                  number_data["tiny_float"]],repeat=2)
    for a in num_args:
        fitness_to_fixation.sswm(*a)



def test_moran(number_data):

    def local_moran(f1,f2,N):
        """
        Stupidly simple local moran model without error checking or fancy numerical
        log stuff to make sure it's doing what it's supposed to do.
        """

        if f1 == f2:
            f2 = f2 - f1/1000

        return np.nan_to_num((1 - (f1 / f2)) / (1 - (f1 / f2) ** N))

    # Make sure it runs
    fitness_to_fixation.moran(1,1,1)

    # Test some known answers
    args = [[0.1,1.0,i] for i in [1,3,10,30,100]]
    args.extend([[1.0,0.1,i] for i in [1,3,10,30,100]])
    for a in args:
        assert np.isclose(fitness_to_fixation.moran(*a),local_moran(*a))

    # Test argument checking
    bad_args = [("stupid","strings","badder"),
                (1,1,"bad"),(1,"bad",1),("bad",1,1),
                (1,1,0),(1,0,1),(0,1,1),
                (1,1,-1),(1,-1,1),(0,-1,1)]
    for a in bad_args:
        with pytest.raises(ValueError):
            fitness_to_fixation.moran(*a)

    # Test numerical overflow checking
    num_args = itertools.product([number_data["max_float"],
                                  number_data["tiny_float"]],repeat=3)
    for a in num_args:
        local_a = list(a)
        # Set population_size to 1.0 if tiny
        if local_a[2] < 1:
            local_a[2] = 1

        fitness_to_fixation.moran(*local_a)


def test_mcclandish(number_data):

    def local_mcclandish(f1,f2,N):
        """
        Stupidly simple local mcclandish model without error checking or fancy numerical
        log stuff to make sure it's doing what it's supposed to do.
        """

        numer = 1 - np.exp(-2 * (f2 - f1))
        denom = 1 - np.exp(-2 * N * (f2 - f1))
        sij = numer/denom
        return sij

    # Make sure it runs
    fitness_to_fixation.mcclandish(1,1,1)

    # Test some known answers
    args = [[0.1,1.0,i] for i in [1,3,10,30,100]]
    args.extend([[1.0,0.1,i] for i in [1,3,10,30,100]])
    for a in args:
        assert np.isclose(fitness_to_fixation.mcclandish(*a),local_mcclandish(*a))

    # Test argument checking
    bad_args = [("stupid","strings","badder"),
                (1,1,"bad"),(1,"bad",1),("bad",1,1),
                (1,1,0),(1,0,1),(0,1,1),
                (1,1,-1),(1,-1,1),(0,-1,1)]
    for a in bad_args:
        with pytest.raises(ValueError):
            fitness_to_fixation.mcclandish(*a)

    # Test numerical overflow checking
    num_args = itertools.product([number_data["max_float"],
                                  number_data["tiny_float"]],repeat=3)
    for a in num_args:
        local_a = list(a)
        # Set population_size to 1.0 if tiny
        if local_a[2] < 1:
            local_a[2] = 1

        fitness_to_fixation.mcclandish(*local_a)

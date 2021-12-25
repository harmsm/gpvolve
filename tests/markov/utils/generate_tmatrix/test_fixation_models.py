import pytest

import gpvolve.markov._generate_tmatrix.generate_tmatrix_cython as cy
import gpvolve.markov._generate_tmatrix.generate_tmatrix_python as py

import numpy as np

import itertools

# Gonna want to *catch* some overflows in test functions.
import warnings
#warnings.filterwarnings("error")

def test_sswm(number_data,pop_gen_scenarios):

    def local_sswm(f1,f2,N):
        """
        Stupidly simple local sswm model without error checking or fancy numerical
        log stuff to make sure it's doing what it's supposed to do.
        """
        sij = (f2 - f1) / f1
        if sij < 0:
            sij = 0
        return 1 - np.exp(-sij)

    for module in [py,cy]:
        print("testing",module)

        limits = number_data

        # Make sure it runs
        module._sswm_tester(1,1,1)

        # Test some known answers
        args = [(0.1,1.0,1),(1.0,0.1,1)]
        for a in args:
            assert np.isclose(module._sswm_tester(*a),
                              local_sswm(*a))

        # Test numerical overflow checking
        num_args = itertools.product([limits["max_float"],
                                      limits["tiny_float"]],repeat=2)
        for a in num_args:
            for n in [1,limits["max_int"]]:
                b = list(a)
                b.append(n)
                module._sswm_tester(*b)

        # Run a wide variety of sceanrios
        for s in pop_gen_scenarios:
            real = module._sswm_tester(*s)
            local = local_sswm(*s)
            if not np.isnan(local):
                assert np.isclose(real,local)
            assert not np.isnan(real)


def test_moran(number_data,pop_gen_scenarios):

    def local_moran(f1,f2,N):
        """
        Stupidly simple local moran model without error checking or fancy numerical
        log stuff to make sure it's doing what it's supposed to do.
        """
        if f1 == f2:
            f2 = f2 - f1/1000

        return (1 - (f1 / f2)) / (1 - (f1 / f2) ** N)


    for module in [py,cy]:
        print("testing",module)

        limits = number_data

        # Make sure it runs
        module._moran_tester(1,1,1)

        # Test numerical overflow checking
        num_args = itertools.product([limits["max_float"],
                                      limits["tiny_float"]],repeat=2)
        for a in num_args:
            for n in [1,limits["max_int"]]:
                b = list(a)
                b.append(n)
                module._moran_tester(*b)

        # Run a wide variety of sceanrios
        for s in pop_gen_scenarios:
            real = module._moran_tester(*s)
            local = local_moran(*s)
            if not np.isnan(local):
                # Relatively loose tolerance here
                assert np.isclose(real,local,rtol=1e-3,atol=1e-3)

            assert not np.isnan(real)

def test_mcclandish(number_data,pop_gen_scenarios):

    def local_mcclandish(f1,f2,N):
        """
        Stupidly simple local mcclandish model without error checking or fancy numerical
        log stuff to make sure it's doing what it's supposed to do.
        """

        numer = 1 - np.exp(-2 * (f2 - f1))
        denom = 1 - np.exp(-2 * N * (f2 - f1))
        sij = numer/denom

        return sij


    for module in [py,cy]:
        print("testing",module)

        limits = number_data

        # Make sure it runs
        module._mcclandish_tester(1,1,1)

        # Test numerical overflow checking
        num_args = itertools.product([limits["max_float"],
                                      limits["tiny_float"]],repeat=2)
        for a in num_args:
            for n in [1,limits["max_int"]]:
                b = list(a)
                b.append(n)
                module._mcclandish_tester(*b)

        # Run a wide variety of sceanrios
        for s in pop_gen_scenarios:
            real = module._mcclandish_tester(*s)
            local = local_mcclandish(*s)
            if not np.isnan(local):
                assert np.isclose(real,local)
            assert not np.isnan(real)

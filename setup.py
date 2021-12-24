#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Note: To use the 'upload' functionality of this file, you must:
#   $ pip install twine

import io, os, sys, glob
from shutil import rmtree
from setuptools import dist

# hack necessary to allow setup.py install..
dist.Distribution().fetch_build_eggs(['Cython>=3.0.0a9', 'numpy>=1.21.1'])

import numpy as np
import Cython.Compiler.Options
from Cython.Build import cythonize, build_ext
from setuptools.extension import Extension
from os.path import join, dirname, abspath
import glob
from numpy.distutils.misc_util import get_info
from setuptools import find_packages, setup, Command

Cython.Compiler.Options.annotate = True

# Very annoying way of linking random libraries from numpy that works on all operating systems
path = dirname(__file__)
src_dir = join(dirname(path), '..', 'src')
defs = [('NPY_NO_DEPRECATED_API', 0)]
inc_path = np.get_include()
# not so nice. We need the random/lib library from numpy
lib_path = [abspath(join(np.get_include(), '..', '..', 'random', 'lib'))]
lib_path += get_info('npymath')['library_dirs']

# Package meta-data.
NAME = 'gpvolve'
DESCRIPTION = "A python package for simulating and analyzing evolutionary trajectories through genotype-phenotype-maps"
URL = 'https://github.com/harmslab/gpvolve'
EMAIL = 'harmsm@gmail.com'
AUTHOR = 'Leander D. Goldbach, Michael J. Harms'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = '0.3.1'

# What packages are required for this module to be executed?
REQUIRED = ['numpy', 'cython', 'networkx', 'msmtools', 'matplotlib',
            'pyslim','tskit','imageio']

# What packages are optional?
EXTRAS = {}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


# list of cython extensions as tuples of (module,path)
extensions = [('gpvolve.simulate.wright_fisher.engine.wright_fisher_engine_cython',
               'gpvolve/simulate/wright_fisher/engine/'),
              ('gpvolve.markov.base._generate_tmatrix.generate_tmatrix_cython',
               'gpvolve/markov/base/_generate_tmatrix')]

# Create extensions to compile
ext_modules = []
for e in extensions:
    # Get pyx files
    model_pyx_files = glob.glob(os.path.join(e[1], "*.pyx"))

    # Get c files, making sure not to grab .c files corresponding to .pyx files
    # that may have been generated previously by cython commands. If we
    # include those, we get duplicate definitions of functions
    for c in glob.glob(os.path.join(e[1], "*.c")):
        if ".".join(c.split(".")[:-1]) + ".pyx" not in model_pyx_files:
            model_pyx_files.append(c)

    ext_modules.append(Extension(e[0],
                                 model_pyx_files,
                                 include_dirs=[inc_path,
                                               np.get_include(),
                                               join(path, '..', '..')],
                                 library_dirs=lib_path,
                                 libraries=['npyrandom', 'npymath'],
                                 define_macros=[('NPY_NO_DEPRECATED_API', 0)]),
                       )

# Make sure these are included with the package
all_c_files = list(glob.glob("**/*.c", recursive=True))
all_c_files.extend(list(glob.glob("**/*.h", recursive=True)))
all_c_files.extend(list(glob.glob("**/*.pyx", recursive=True)))
all_c_files.extend(list(glob.glob("**/*.pxd", recursive=True)))
all_c_files.extend(list(glob.glob("**/*.eidos", recursive=True)))

# Where the magic happens:
setup(
    name=NAME,
    version=about['__version__'],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    package_data={"": all_c_files},
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    ext_modules=ext_modules,
    # $ setup.py publish support.
    cmdclass={
        'upload': UploadCommand,
        'build_ext': build_ext,
    },
)

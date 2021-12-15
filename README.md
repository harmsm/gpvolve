# gpvolve

[![PyPI version](https://badge.fury.io/py/gpvolve.svg)](https://badge.fury.io/py/gpvolve)
[![](https://readthedocs.org/projects/gpvolve/badge/?version=latest)](https://gpvolve.readthedocs.io/en/latest/?badge=latest)
[![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/lperezmo/gpvolve/blob/master/examples/introduction.ipynb)

**A python package for extracting evolutionary trajectories from genotype-phenotype-maps**

A Python API for the simulation and analysis of evolution in genotype-phenotype space.
You can use this library to:

   1. Build a markov state model from a genotype-phenotype-map.
   2. Find clusters of genotypes that represent metastable states of the system, using PCCA+.
   3. Compute fluxes and pathways between pairs of genotypes and/or clusters of interest, using Transition Path Theory.
   4. Visualize the outputs of all of the above.

## Basic Example

Create a random genotype-phenotype map using `gpvolve` package
```python
from gpmap.simulate import generate_nk
import numpy as np

# Generate random genotype-phenotype map
gpm = generate_nk(gpm_output_column='phenotype',
                  k=0,
                  num_sites=4)
   
# Find all neighbors between nodes
gpm.get_neighbors()              
```
Assign customized fitness values to the genotype-phenotype map (required in order for `gpvolve` to work)
```python
# Define some values for fitness, it can be anything you want

# EXAMPLE 1 : random fitness values based on how many genotypes there are
fitnesses = np.random.random(size=len(gpm.data.genotype))
# EXAMPLE 2 : fitness values based on a phenotype value
fitnesses = np.log(gpm.data.phenotype)

# Add fitness values column to gpm.data pandas data frame using pandas in-place operations
gpm.data.loc[:,"fitness"] = fitnesses
```

Find the optimal number of clusters to divide your genotypes.
```python
from gpvolve.cluster import optimize
optimal_num_clusters = optimize(T, criterion='Spectral gap')
```
Plot and color the nodes of the genotypes using any attribute within the genotype-phenotype map.
```python
from matplotlib import pyplot as plt
from gpmap import GenotypePhenotypeGraph

# Remove self-looping edges
gpm.neighbors.loc[gpm.neighbors.direction != 1,"include"] = False

# Create GenotypePhenotypeGraph object
G = GenotypePhenotypeGraph()
G.add_gpm(gpm)

# Add custom node data and labels, with chosen colormaps
G.add_node_cmap(data_column="phenotype")
G.add_node_labels(data_column="genotype")

# Plot map
G, fig, ax = plot(G,
                  figsize=(15,10),
                  plot_node_labels=True, 
                  node_options={'node_size':5000}, 
                  edge_options={'arrows':'black',
                                'arrowsize':1})
ax.set_title("Map nodes labeled by genotype and colored by fitness", 
             fontsize=20)
plt.show()
```

<img src="docs/img/example_map.png" width="700">


## Install

To install from PyPI:
```
pip install gpvolve
```
To install a development version:
```
git clone https://github.com/harmslab/gpvolve
cd gpvolve
pip install  -e .
```
To install a development version from testpypi:
```
pip install --upgrade cython msmtools tqdm
pip install -i https://test.pypi.org/simple/ gpmap==0.8.3
pip install -i https://test.pypi.org/simple/ gpvolve==0.3.1
```


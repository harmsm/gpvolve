__description__ = \
"""
"""
__date__ = "2021-12-10"
__author__ = "Clara Rehmann"

import gpmap
from gpvolve.simulate import utils

import imageio

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

## PLOTTING FUNCTIONS

def plot_gen(gen, gpm, countframe, stepframe, cmap):
    """
    Plot one-generation snapshot of gpmap, for animate_flux() function
    """

    gpm.get_neighbors()

    # get rid of self loops
    mask = gpm.neighbors.direction != 1
    gpm.neighbors.loc[mask, 'include'] = False

    # set node size based on pop. size
    gpm.data['slim_size'] = countframe.loc[gen]

    # set edge size based on # of steps
    gpm.neighbors['slim_weight'] = [0]*len(gpm.neighbors)
    maximum=0
    for column in stepframe.columns:
        if stepframe[column].max() > maximum:
            maximum = stepframe[column].max()

    for col in stepframe.columns:
        gpm.neighbors.loc[gpm.neighbors.edge == eval(col), 'slim_weight'] = (stepframe[col][gen]/(100*maximum))

    # graph parameters
    G = gpmap.GenotypePhenotypeGraph()
    G.add_gpm(gpm)


    if np.mean(gpm.neighbors['slim_weight']) > 0:
        G.add_edge_sizemap(data_column='slim_weight', size_min=0.001, size_max = 5)
    else:
        G.edge_options['width'] = 1
        G.edge_options['edge_color'] = 'w'
    G.add_node_sizemap(data_column='slim_size', size_min=10, size_max=5000)
    G.add_node_cmap(data_column='slim_fitness', cmap=cmap)

    # plot
    p = gpmap.plot(G, edge_options={'arrows':None, 'arrowsize' : .00001}, figsize=(10,10))
    return p


class SimulationResult:
    """
    """

    def __init__(self,
                 gpm,
                 max_generations,
                 mutation_rate,
                 population_size,
                 node_counts,
                 edge_counts,
                 node_history):

        # set gpmap, get neighbors
        self._gpm = gpm

        self._max_generations = max_generations
        self._mutation_rate = mutation_rate
        self._population_size = population_size

        self._node_counts = node_counts
        self._edge_counts = edge_counts
        self._node_history = node_history

        self._generations = len(self._node_counts) #max(self._node_counts.keys())
        self._edge_weights = utils.make_fluxdict(self._gpm, self._node_history)


    @property
    def gpm(self):
        """
        genotype-phenotype map object used to run slimulation
        """
        return self._gpm

    @property
    def node_counts(self):
        """
        Pandas DataFrame representation of occupants at each node over the simulation
        """
        return self._node_counts

    @property
    def edge_counts(self):
        """
        Pandas DataFrame representation of edge steps taken @ each generation
        """
        return self._edge_counts

    @property
    def edge_weights(self):
        ## MAYBE STORE IN GPM???
        return self._edge_weights

    @property
    def generations(self):
        return self._generations

    @property
    def population_size(self):
        return self._population_size

    @property
    def mutation_rate(self):
        return self._mutation_Rate

    # ----------------------------------
    # Plotting functions
    # ----------------------------------

    def plot_flux(self, cmap = 'plasma'):
        """
        Plot relative edge flux over simulation.

        Returns:
        --------
        g : GenotypePhenotypeGraph object

        """

        for key, value in self._edge_weights.items():
            self._gpm.neighbors.loc[self._gpm.neighbors['edge']==key, 'slim_weight'] = value

        G = gpmap.GenotypePhenotypeGraph()
        G.add_gpm(self._gpm)

        # get rid of self loops
        mask = self._gpm.neighbors.direction != 1
        self._gpm.neighbors.loc[mask, 'include'] = False

        G.edge_options["arrows"] = None
        G.edge_options['arrowsize']=0.001
        G.add_edge_sizemap(data_column = 'slim_weight')
        G.add_node_cmap(data_column = 'slim_fitness', cmap = cmap)
        g = gpmap.plot(G)
        return g

    def plot_gt(self):
        """
        Plot proportion of genotypes over each generation.

        Returns:
        --------
        plot : MatPlotLib object (?)

        """
        df = self.node_counts
        plot = plt.stackplot(df.index, df.values.T, labels = df.index)
        plt.legend()
        return plot

    def animate_flux(self, outpath, cmap='plasma'):
        """
        make animated gif of flux over gpmap landscape
        """

        # load data
        countframe = self.node_counts
        stepframe = utils.make_stepframe(self._edge_counts, self._generations)
        gpm = self._gpm

        # make plots
        for gen in range(1, len(countframe)):
            p = plot_gen(gen, gpm, countframe, stepframe, cmap=cmap)
            plt.savefig(outpath+'_'+str(gen)+'.png')
            plt.close()

        # animate
        with imageio.get_writer(outpath+'.gif', mode='I') as writer:
            for gen in range(1, len(countframe)):
                image = imageio.imread(outpath+'_'+str(gen)+'.png')
                writer.append_data(image)
        writer.close()

    # ----------------------------------
    # Read/write functions
    # ----------------------------------
    def to_json(self, outpath):
        """
        Write SLiMsim as a json for future reference
        """
        # update keys to be json-friendly
        edge_counts = dict(zip([str(k) for k in self._edge_counts.keys()], list(self._edge_counts.values())))
        edge_weights = dict(zip([str(k) for k in self._edge_weights.keys()], list(self._edge_weights.values())))

        def convert(o):
            if isinstance(o, np.int64): return int(o)
            raise TypeError

        f = open(outpath+'.json', 'w')
        jsslim = {
                'gpmap': self._gpm.to_dict(),
                'node_counts': self._node_counts,
                'edge_counts': edge_counts,
                'edge_weights': edge_weights,
                'generations': self._generations,
                'population_size': self._population_size,
                'mutation_rate': self._mutation_rate,
                  }
        json.dump(jsslim, f, default=convert)
        f.close()

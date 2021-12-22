import gpmap
from gpmap import GenotypePhenotypeMap
from matplotlib import pyplot as plt
import pyslim, tskit
import numpy as np
import pandas as pd
import subprocess, time, os, imageio, json

from . import utils

class GenotypePhenotypeSLiM():
    """
    Class for building and analyzing population genetics models

    PARAMETERS:
    -----------
    gpm (GenotypePhenotypeMap object) : gpmap to use for the simulation

    ATTRIBUTES:
    -----------
    node_counts (DataFrame) : Pandas DataFrame containing the number of individuals occupying
        each node (columns) at each generation (rows)
    edge_weights (dict) : Dictionary of edges (keys) and the relative proportion of steps taken on that edge (values)
    edge_counts (DataFrame) : Pandas Dataframe containing the number of edge steps (columns)
        observed in each generation (rows)
    generations (int) : Number of generations the simulation was run for
    population_size (int) : Number of individuals in the simulated population
    mutation_rate (float) : Mutation rate used in the simulation

    """

    def __init__(self, gpm, max_generation=None, population_size=None, mutation_rate=None):
        # set gpmap, get neighbors
        self._gpm = gpm
        self._gpm.get_neighbors()

        # SLiM outputs empty
        self._node_counts = None
        self._edge_weights = None
        self._edge_counts = None
        self._generations = None

        # metadata
        self._max_generation = max_generation
        self._population_size = population_size
        self._mutation_rate = mutation_rate

    def slimulate(self, population_size, mutation_rate, max_generation, outpath, overwrite=False, slim_path='slim', haploid=False, fitness_column='fitness'):
        """
        run one SLiM simulation

        Parameters
        ----------
        population_size (int) : Number of individuals in the population
        mutation_rate (float) : Mutation rate
        max_generation (int) : Maximum number of generations to run simulation
            (simulation will finish if derived state fixes)
        outpath (str) : Path for SLiM working output files
        overwrite (bool) : Overwrite existing SLiM files if they already exist at the specified outpath.
            (default = False, function will exit without executing)
        slim_path (str) : Path to SLiM engine (default = 'slim')
        haploid (bool) : run SLiM on a haploid population (default = False)

        Returns
        -------
        _node_counts : dictionary
            Nested dictionary of nodes (keys) and number of individuals occupying each node (values) at each generation
        _edge_weights : dictionary
            Dictionary of edges (keys) and the relative proportion of steps taken on that edge (values)
        _edge_counts : dictionary
            Dictionary of edges (keys) and the generations in which that step was observed (values)
        _generations : int
            Number of generations the simulation was run for
        _population_size : int
            Number of individuals in the simulated population
        _mutation_rate : float
            Mutation rate used in the simulation
        """

        # make sure slim is installed
        try:
            subprocess.run([slim_path])
        except FileNotFoundError:
            err = "SLiM not in path. Is it installed?"
            raise RuntimeError(err)

        # check if outpath is occupied
        if os.path.exists(outpath+'_gtcount.txt'):
            if overwrite:
                for sfx in ['_gtcount.txt', '_gpmap_SLiM.txt', '_gttransitions.txt', '_treeinds.txt', '.slim', '.trees']:
                    if os.path.exists(outpath+sfx):
                        os.remove(outpath+sfx)
            else:
                raise FileExistsError(f'Outpath {outpath} is already occupied! Exiting...')
                return

        # write SLiM script and SLiM-friendly gpmap
        utils.write_slim(self._gpm.length, max_generation, outpath, haploid=haploid)
        utils.make_gpm(self._gpm, outpath, fitness_column=fitness_column)

        # Construct command
        command = [slim_path,
                   "-d",f"MUTATIONRATE={mutation_rate}",
                   "-d",f"POPULATIONSIZE={population_size}",
                   "-d",f"GPMAP=\\\"{outpath}_gpmap_SLiM.txt\\\"",
                   "-d",f"OUTPATH=\\\"{outpath}\\\"",
                   f"{outpath}.slim"]
        command = " ".join(command) # wacky arg parsing in SLiM; pass as string


        # run SLiM!
        with open(outpath+'_stdout.txt', 'wb') as out, open(outpath+'_stderr.txt', 'wb') as err:
            print(command)
            p = subprocess.run(command, stdout=out, stderr=err, shell=True)

        # save outputs
        self._node_counts = utils.get_gtcount(outpath+'_gtcount.txt', self._gpm)
        self._edge_counts = utils.get_stpdict(outpath+'_gttransitions.txt')
        self._node_history = utils.get_hist(outpath+'_treeinds.txt', pyslim.load(outpath+'.trees'))
        self._edge_weights = utils.make_fluxdict(self._gpm, self._node_history)
        self._generations = max(self._node_counts.keys())
        self._population_size = population_size
        self._mutation_rate = mutation_rate

    # ----------------------------------
    # Get attributes
    # ----------------------------------

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
        nc = pd.DataFrame.from_dict(self._node_counts, orient = 'index')
        return nc

    @property
    def edge_counts(self):
        """
        Pandas DataFrame representation of edge steps taken @ each generation
        """
        ec = utils.make_stepframe(self._edge_counts, self._generations)
        return ec

    @property
    def edge_weights(self):
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
        df = pd.DataFrame.from_dict(self._node_counts, orient = 'index')
        plot = plt.stackplot(df.index, df.values.T, labels = df.index)
        plt.legend()
        return plot

    def animate_flux(self, outpath, cmap='plasma'):
        """
        make animated gif of flux over gpmap landscape
        """

        # load data
        countframe = pd.DataFrame.from_dict(self._node_counts, orient = 'index')
        stepframe = utils.make_stepframe(self._edge_counts, self._generations)
        gpm = self._gpm

        # make plots
        for gen in range(1, len(countframe)):
            p = utils.plot_gen(gen, gpm, countframe, stepframe, cmap=cmap)
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

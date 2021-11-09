from gpmap import GenotypePhenotypeMap
from matplotlib import pyplot as plt
import pyslim, tskit
import subprocess, time, os, imageio
from .SLiM import *

class GenotypePhenotypeSLiM():
    """
    Class for building and analyzing population genetics models
    
    Parameters
    ----------
    gpm : GenotypePhenotypeMap object.
        A genotype-phenotype map object.
        
    Attributes
    ----------
    node_counts : dictionary
        Nested dictionary of nodes (keys) and number of individuals occupying each node (values) at each generation
    edge_weights : dictionary
        Dictionary of edges (keys) and the relative proportion of steps taken on that edge (values)
    edge_counts : dictionary
        Dictionary of edges (keys) and the generations in which that step was observed (values)
    generations : int
        Number of generations the simulation was run for
    population_size : int
        Number of diploid individuals in the simulated population
    mutation_rate : float
        Mutation rate used in the simulation
    
    """    
    
    def __init__(self, gpm, max_generation=None, population_size=None, mutation_rate=None):
        self._gpm = gpm
        self._gpm.get_neighbors()
        self._node_counts = None
        self._edge_weights = None
        self._edge_counts = None
        self._generations = None
        self._max_generation = max_generation
        self._population_size = population_size
        self._mutation_rate = mutation_rate
    
    def slimulate(self, population_size, mutation_rate, max_generation, outpath, slim_path='slim'):
        """
        run one SLiM simulation
        
        Parameters
        ----------
        population_size : integer
            Number of diploid individuals in the population
        mutation_rate : float
            Mutation rate
        max_generation : integer
            Maximum number of generations to run simulation for (simulation will finish if derived state fixes)
        outpath : string
            Path for SLiM working output files (NOTE: SLiM outfiles must not already exist at this path)
        slim_path : string (default 'slim')
            Path to SLiM engine
        
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
            Number of diploid individuals in the simulated population
        _mutation_rate : float
            Mutation rate used in the simulation
        
        """
        # write SLiM script and SLiM-friendly gpmap
        utils.write_slim(self._gpm.length, max_generation, outpath)
        utils.scale_fitness(self._gpm, outpath)
        
        # strings to send to SLiM
        map_dec = ' GPMAP=\\"'+outpath+'_gpmap_scaled.txt\\"'
        out_dec = ' OUTPATH=\\"'+outpath+'\\"'
        
        # command to run SLiM
        command = (slim_path +
                " -d MUTATIONRATE="+str(mutation_rate)+
                " -d POPULATIONSIZE="+str(population_size)+
                " -d"+map_dec+
                " -d"+out_dec+" "+
                outpath+".slim")
        
        # run SLiM!
        with open(outpath+'_stdout.txt', 'wb') as out, open(outpath+'_stderr.txt', 'wb') as err:
            p = subprocess.Popen(command, stdout=out, stderr=err, shell=True)
            p.wait()
            
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

    def plot_flux(self):
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
        G.add_node_cmap(data_column = 'phenotype')
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

    def animate_flux(self, outpath):
        """
        make animated gif of flux over gpmap landscape
        """
    
        # load data
        countframe = pd.DataFrame.from_dict(self._node_counts, orient = 'index')
        stepframe = utils.make_stepframe(self._edge_counts, self._generations)
        gpm = self._gpm
    
        # make plots
        for gen in range(1, len(countframe)):
            p = utils.plot_gen(gen, gpm, countframe, stepframe)
            plt.savefig(outpath+'_'+str(gen)+'.png')
            plt.close()
        
        # animate
        with imageio.get_writer(outpath+'.gif', mode='I') as writer:
            for gen in range(1, len(countframe)):
                image = imageio.imread(outpath+'_'+str(gen)+'.png')
                writer.append_data(image)
        writer.close()
    
    
    
        
        
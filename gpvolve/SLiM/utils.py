import pyslim, tskit, gpmap
from pathlib import Path
from gpmap import GenotypePhenotypeMap
import numpy as np, pandas as pd

# utility functions for writing/reading SLiM inputs and outputs

## WRITING FUNCTIONS

def write_slim(L_sites, generations, out, haploid=False):
    """
    Write SLiM script for an L-site epistasis model, calling fitness values from a genotype-phenotype map.
    
    The basic SLiM code blocks are:
        - initialize() callback
            - initialize parameters for simulation parameters
            - initialize chromosome with L sites, each with a unique mutation allowed
            - establish mutation stack groups (only one mutation is allowed at each site, back mutation allowed)
        - first generation callback
            - initialize SLiM subpopulation
            - add 'ancestral' mutation to all sites in all individuals
            - define genotype-phenotype dictionary
        - fitness(NULL) callback
            - defines an individual's fitness based on gpmap values
        - mutation(NULL) callback
            - writes new mutations to a text file as they arise
        - run the simulation
            - gtcount() function writes the census count of populated nodes at each generation to _gtcount.txt
            - background_check() function writes observed edge transitions at each generation to _gttransitions.txt
            - simulation runs until 'generations' is reached or until all mutations fix
            - once finished running, saves tree sequence information
    
    PARAMETERS:
    -----------
    L_sites (int) : number of interacting sites 
    generations (int) : number of generations to run the simulation for
    out (str) : outpath for SLiM script (appended with .slim)
    haploid (bool) : run the simulation on a haploid population (default = False, 
            population is diploid and fitness is averaged between both chromosomes)
    
    RETURNS:
    --------
    None (SLiM script is written to 'out' parameter)
    """
    
    # find gt_to_pt.eidos filepath
    eidospath = Path(__file__).parent / "gt_to_pt.eidos"
    
    out=out+'.slim'
    with open(out, 'w') as f:
        # initialize callback
        f.writelines('\n'.join(['// '+str(L_sites)+'-site epistasis model',
            'initialize() {',
            '\tinitializeSLiMOptions(keepPedigrees=T);',
            '\tsource("'+str(eidospath)+'");',
            #'\tsource("gt_to_pt.eidos");',
            '\tinitializeTreeSeq();',
            '\tinitializeRecombinationRate(0);',
            '\tinitializeMutationRate(MUTATIONRATE);',
            '\n']))
        for site in range(L_sites):
            f.writelines('\n'.join(['\t initializeMutationType("m'+str(site)+'", 1.0, "f", 1.0);',
                                    '\t initializeGenomicElementType("g'+str(site)+'", m'+str(site)+', 1.0);',
                                    '\t initializeGenomicElement(g'+str(site)+', '+str(site)+', '+str(site)+');',
                                    '\t m'+str(site)+'.convertToSubstitution = F;',
                                    '\t m'+str(site)+'.mutationStackPolicy = "l";',
                                    '\t m'+str(site)+'.mutationStackGroup = -1;',
                                    '\n']))
        f.write('\t initializeMutationType("m1000", 1.0, "f", 1.0);\n')
        f.write('\t m1000.mutationStackPolicy = "l";\n')
        f.write('\t m1000.mutationStackGroup = -1;\n')
        f.write('}\n')

        # create population and establish parameters
        f.write('1 {\n')
        f.write('\tsim.addSubpop("p1", POPULATIONSIZE);\n')
        f.write('\tfor (genome in p1.genomes) {\n')
        f.write('\t\tfor (pos in 0:('+str(L_sites)+'-1)) {\n')
        f.write('\t\t\tgenome.addNewMutation(m1000,1.0,pos);\n')
        f.write('\t\t}\n')
        f.write('\t\tfor (mutation in genome.mutations) {\n')
        f.write('\t\t\tmutation.tag=0;\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('\tdefineGlobal("gpmap",gpmap_load(GPMAP));\n')
        f.write('\tdefineGlobal("gtdict",gt_list(GPMAP));\n')
        f.write('}\n')
        
        # epistasis fitness callback
        f.write('// epistasis callback\n')
        f.write('fitness(NULL) {\n')
        f.write('\tmuts1=c(individual.genome1.mutations.mutationType.id);\n')
        f.write('\tmuts2=c(individual.genome2.mutations.mutationType.id);\n')
        f.write('\tgt1=slim_to_mut(muts1);\n')
        f.write('\tgt2=slim_to_mut(muts2);\n')
        f.write('\tph1=gpmap.getValue(gt1);\n')
        f.write('\tph2=gpmap.getValue(gt2);\n')
        if haploid:
            f.write('\treturn (ph1);\n')
        else:
            f.write('\treturn ((ph1+ph2)/2);\n')
        f.write('}\n')
        
        # track background of new mutations
        f.write('mutation(NULL) {\n')
        f.write('// no same mutations at a site\n')
        f.write('\tfor (m in genome.mutations) {\n')
        f.write('\t\tif (m.mutationType == mut.mutationType) {\n')
        f.write('\t\t\treturn F;\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('\tmuts=c(genome.mutations.mutationType.id);\n')
        f.write('\tgt=slim_to_mut(muts);\n')
        f.write('\tgtv=gtdict.getValue(gt);\n')
        f.write('\tmut.tag=(asInteger(gtv));\n')
        f.write('\treturn T;\n')
        f.write('}\n')

        # run until max generation or all mutations fix
        f.write('1:'+str(generations)+' late() {\n')
        f.write('\tgt_count(sim.generation, p1, gtdict); // write genotype count\n')
        f.write('\tif (any(sim.mutations.originGeneration==sim.generation)) { // save mutation transitions\n')
        f.write('\t\tidx=which(sim.mutations.originGeneration==sim.generation); {\n')
        f.write('\t\tfor (i in idx) {\n')
        f.write('\t\t\tmutation=sim.mutations[i];\n')
        f.write('\t\t\tbackground_check(sim.generation, p1, mutation);\n')
        f.write('\t\t\t}\n')
        f.write('\t\t}\n')
        f.write('\t}\n')
        f.write('\tif (check_fix(p1)) {\n')
        f.write('\t\ttree_deets(p1, gtdict);\n')
        f.write('\t\tsim.treeSeqOutput(OUTPATH+".trees");\n')
        f.write('\t\tsim.simulationFinished();\n')
        f.write('\t}\n')
        f.write('}\n')

        f.write(str(generations)+' late() {\n')
        f.write('\ttree_deets(p1, gtdict);\n')
        f.write('\tsim.treeSeqOutput(OUTPATH+".trees");\n')
        f.write('\tsim.simulationFinished();\n')
        f.write('}')
    return None

def make_gpm(gpm, out, fitness_column='fitness'):
    """
    write SLiM-readable gpmap (just a tsv with an extra column defining the fitness values that SLiM should use)
    
    PARAMETERS:
    -----------
    gpm (GenotypePhenotypeMap object) : gpmap to use
    fitness_column (str) : gpmap column for SLiM to use as fitness values. default = 'fitness'
    out (str) : outpath for gpmap tsv (appended with '_gpmap_SLiM.txt')
    
    RETURNS:
    --------
    None (gpmap is written to file)
    """
    # make duplicate column for slim to look at
    gpm.data['slim_fitness'] = gpm.data[fitness_column]
    # save
    gpm.data.to_csv(out+'_gpmap_SLiM.txt', sep='\t')
    return None
    

#def scale_fitness(gpm, out):
#    """
#    DEFUNCT
#    
#    center phenotype values around 1, preserving relative scale
#    (1.0 is neutral in SLiM)
#
#    Parameters
#    __________
#    gpm: genotype phenotype map object
#    out: outpath for gpmap tsv
#    """
#    out=out+'_gpmap_scaled.txt'

#    gpm.data['phenotype'] = gpm.data['phenotype'] + (1.0 - np.mean(gpm.data['phenotype']))

#    gpm.data.to_csv(out, sep='\t')
#    return None

## READING FUNCTIONS

def get_gtcount(filepath, gpm):
    """
    Read 'gtcount' SLiM output into a nested Python dictionary of genotype counts at each generation.
    Each line of 'gtcount' file consists of:
        (generation)\t(' '-separated list of occupied nodes)\t(' '-separated list of respective counts)
    
    PARAMETERS:
    -----------
    filepath (str) : path to _gtcount.txt SLiM output
    gpm (GenotypePhenotypeMap object) : gpmap used to run the simulation

    RETURNS:
    --------
    gtcount (dict) : nested dictionary of genotype counts at each generation. 
        genotype integer ID corresponds to gpmap node. 
        format = {generation: {gt0: count, gt1: count}}
    """

    f = open(filepath, 'r')
    gtcount = {}

    for line in f.readlines():
        line = line.strip().split('\t')
        gen = int(line[0])                  # generation
        gts = line[1].strip().split(' ')    # IDs of present genotypes
        cts = line[2].strip().split(' ')    # count of each
        cdict = dict(zip([int(g) for g in gts], [int(c) for c in cts]))
        gendict = {}
        for ID in range(len(gpm.data)):
            if cdict.get(ID):
                gendict.update({ID: cdict[ID]})
            else:
                gendict.update({ID: 0})
        gtcount.update({gen: gendict})
    return gtcount

def get_stpdict(filepath):
    """
    Read 'gttransitions' SLiM output into a Python dictionary of generations each edge transition was observed in.
    Each line of 'gttransitions' file consists of:
        (generation)(' ')(ancestral node)('/')(derived node)
    
    PARAMETERS:
    -----------
    filepath (str) : path to _gttransitions.txt SLiM output
    
    RETURNS:
    --------
    mutdict: dictionary containing edge transition tuples and a list of generations they occur at.
        format = {(ancestral node, derived node): [generations observed]}
    """

    f = open(filepath, 'r')
    genarray = []
    tsnarray = []

    for line in f.readlines():
        gen, mut = line.strip().split(' ')
        m1, m2 = mut.split('/')
        genarray.append(int(gen))
        tsnarray.append((int(m1), int(m2)))

    tsn = np.unique(tsnarray, axis = 0)
    mutdict = {}
    for t in tsn:
        gidx = np.where(((np.array(tsnarray)) == t).all(axis=1))[0]
        mutdict.update({tuple(t): list(np.array(genarray)[gidx])})

    return mutdict

def get_hist(filepath, treeseq):
    """
    Read the output tree sequence from the finished simulation, parse the mutational trajectories taken by final generation
    
    PARAMETERS:
    -----------
    filepath (str) : path to SLiM _treeinds.txt output (gpmap-relevant metadata for tree sequence)
    treeseq (pyslim SlimTreeSequence object) : tree sequence SLiM output
    
    RETURNS:
    --------
    histdict (dict) : nested dictionary of genotypes at the final generation, 
        edge paths taken to get there, and number of individuals taking that path.
        format = {node: {(ancestral, indermediate, derived): number of individuals}}
    """
    # get mutation table from tree sequence 
    mtable = treeseq.dump_tables().mutations
    # read list of genotypes in last generation
    f = open(filepath, 'r')
    genomes = []
    gtIDs = []
    for line in f.readlines():
        arr = line.strip().split(' ')
        gtIDs.append(int(arr[1])) 
        genomes.append([int(i) for i in arr[2:]])
    # unique genotypes
    g_uniq = np.unique(genomes, axis = 0)
    # hash metadata and load into dictionary 
    mdata_bytes = tskit.unpack_bytes(mtable.metadata, mtable.metadata_offset)
    metadata = {}
    for i in range(len(mdata_bytes)):
        meta = mtable.metadata_schema.decode_row(mdata_bytes[i])['mutation_list'][0]
        for key in list(meta.keys()):
            if (key != 'mutation_type') & (key != 'slim_time'):
                del meta[key]
        meta.update({'site':mtable[i].site})
        metadata.update({i:meta})
    # array of unique mutation states
    state_array = tskit.unpack_strings(mtable.derived_state, mtable.derived_state_offset)
    # array of parent indices
    parent_array = mtable.parent
        
    histdict = {} # initialize our dictionary

    def background_check(mutation, state_array, parent_array):
        """
        function for getting ancestors of a ts mutation (used by get_hist()).
        steps back through recorded mutational states for a final tree sequence node.
        """
        idx = []
        p0 = state_array.index(mutation)
        while parent_array[p0] != -1:   # step back until first generation
            idx.append(p0)
            p0 = state_array.index(state_array[parent_array[p0]])
        return idx
    
    for gt in np.unique(gtIDs):
        pwys = np.unique(np.array(genomes)[np.where(gtIDs == gt)[0]], axis=0) # unique SLiM mutation combinations for this genotype
        if gt == 0: # if ancestral, don't record any edge steps
            histdict.update({gt:(sum([np.count_nonzero((genomes == p).all(axis=1)) for p in pwys]))})
        else:
            pathdict = {}
            counts = []
            morders = []
            for unique_path in pwys: # for each edge step pathway to this genotype
                count = np.count_nonzero((genomes == unique_path).all(axis=1)) # number of genomes that took path
            
                # get background
                bg = [background_check(str(m), state_array, parent_array) for m in unique_path]
                types = [metadata[b]['mutation_type'] for b in np.concatenate(bg).flat]
                sites = [metadata[b]['site'] for b in np.concatenate(bg).flat]
                times = [metadata[b]['slim_time'] for b in np.concatenate(bg).flat]
    
                # what order derived sites occurred in
                tuples = [(t, s, y) for t, s, y in zip(times, sites, types)]
                morder = [t[1] for t in sorted(tuples) if t[2] != 1000]
            
                counts.append(count)
                morders.append(tuple(morder))
            
            for order in np.unique(morders, axis = 0):
                pathdict.update({tuple(order):sum(np.array(counts)[np.where((morders == order).all(axis=1))[0]])})
        
            histdict.update({gt:pathdict})
    return histdict


def make_fluxdict(gpm, genotypehistory):
    """
    Parse histdict output from get_hist() into a dictionary of scaled edge weights.
    
    PARAMETERS:
    -----------
    gpm (GenotypePhenotypeMap object) : gpmap used to run the simulation
    genotypehistory (dict) : histdict output from get_hist()
    
    RETURNS:
    --------
    fluxdict (dict) : nested dictionary of steps between ancestral and derived genotype,
        edge transitions at that step, and their scaled # of observations (weights)
        format = {step: {(ancestral, derived): weight}}
    
    """
    gpm.get_neighbors()
    pwy_count = genotypehistory.get(np.where(gpm.data.n_mutations == gpm.length)[0][0])
    pwys = list(pwy_count.keys())
    binary = np.array([list([int(s) for s in g]) for g in gpm.data.binary])

    fluxdict = {}

    for i in range(gpm.length): # for each required step between ancestral and derived
        counts = {}
        for p in pwys: # for unique path taken, which step did they take here?
            # establish binary derived genotype
            gt_bin_dev = [0]*gpm.length 
            for site in range(i, -1, -1):
                gt_bin_dev[p[site]] = 1
            if i==0:
                gt_bin_anc = [0]*gpm.length # binary ancestral genotype
            else:
                gt_bin_anc = [0]*gpm.length
                for site in range(i-1, -1, -1):
                    gt_bin_anc[p[site]] = 1
            gt_dev = np.where((binary == gt_bin_dev).all(axis=1))[0][0] # derived genotype gpmap index
            gt_anc = np.where((binary == gt_bin_anc).all(axis=1))[0][0] # ancestral genotype gpmap index

            # number of individuals who took that (anc, dev) step
            counts.update({(gt_anc,gt_dev):(int(0 if counts.setdefault((gt_anc,gt_dev)) is None 
                                            else counts.setdefault((gt_anc,gt_dev)))
                                        +pwy_count[p])})
        # scale
        s = sum(counts.values())
        for key, value in counts.items():
            counts[key] = value / s
        fluxdict.update(counts)
    return fluxdict

def make_stepframe(stpdict, gen):
    """
    function to parse stepdict into a gen-by-gen count dataframe 
    """
    # establish dataframe for transition counts each generation
    stepframe = pd.DataFrame(columns = [str(k) for k in stpdict.keys()], index = range(gen+1))
    stepframe = stepframe.fillna(0)

    # fill counts at each generation
    for key, value in stpdict.items():
        for v in value:
            stepframe[str(key)][v] += 1

    return stepframe


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
## script to run SLiM scripts a specified number of times,
## or until equilibrium (change in mean flux is beneath cutoff)

import gpmap, argparse, os, gpvolve
import numpy as np, pandas as pd

parser=argparse.ArgumentParser()
parser.add_argument('--gpmap', help='path to genotype-phenotype map csv')
parser.add_argument('--N', type=int, help='population size')
parser.add_argument('--u', type=float, help='mutation rate')
parser.add_argument('--runs', type=int, default=None, help='number of SLiM runs to execute (otherwise will run until equilibrium cutoff is reached')
parser.add_argument('--cutoff', default=0.001, type=float, help='significance cutoff for equilibrium (default = 0.001)')
parser.add_argument('--working_dir', help='path for slim working files')
parser.add_argument('--slim_out', help='outpath for SLiMsim .json outputs (appended with _run_X if overwrite = False)')
parser.add_argument('--overwrite', default=False, action='store_true', help='overwrite SLiM json files while running? (useful for low-memory issues. default = False')
parser.add_argument('--outpath', help='outpath for summary tsv (appended with "_paths.txt"). columns are "equilibrium" (bool) and edge weights for each simulation (rows).')
args=parser.parse_args()


# functions for running SLiM
def read_gpm(gpm_path):
    gpm = gpmap.read_csv(gpm_path)
    return gpm

def run_slim(gpm, working_dir, N, u):
    sim = gpvolve.slimsim.GenotypePhenotypeSLiM(gpm)
    
    #population_size, mutation_rate, max_generation, outpath, overwrite=False, slim_path='slim', haploid=False):
    
    sim.slimulate(N, u, 100000000, working_dir, overwrite=True)

    return sim
    
# functions for compiling simulation outputs
def get_edgeweights(sim):
    return sim.edge_weights

def append_steps(pathframe, edgeweights):
    pathframe = pathframe.append(edgeweights, ignore_index = True)
    pathframe = pathframe.fillna(0)
    return pathframe

# functions to see if equilibrium is reached
def add_to_mu(mu, val, size):
    # add val to overall mu (size = new length)
    return mu+((val-mu)/size)

def add_to_sigma2(sigma2, val, mu, size):
    # add val to overall sigma2
    return sigma2+(((val-mu)**2)/(size-1))

def update(mu, s2, loc, df):
    oldmu = mu.copy()
    olds2 = s2.copy()
    edges = dict(zip(df.columns, df.loc[loc]))
    for key, value in edges.items():
        newm = add_to_mu(oldmu[key], value, loc+1)
        news = add_to_sigma2(olds2[key], value, oldmu[key], loc+1)
        mu.update({key: newm})
        s2.update({key: news})
    return oldmu, mu, olds2, s2

def find_eq(pathframe):
    mu = dict(zip(pathframe.columns, pathframe.loc[0]))
    s2 = dict(zip(pathframe.columns, [0]*len(pathframe.columns)))

    # establish first gen
    oldmu, mu, olds2, s2 = update(mu, s2, 1, pathframe)
    diff = (np.sum(np.abs(np.array(list(s2.values())) - np.array(list(olds2.values())))))

    # loop thru til eq
    for loc in range(2,len(pathframe)):
        oldmu, mu, olds2, s2 = update(mu, s2, loc, pathframe)
        diff = (np.sum(np.abs(np.array(list(s2.values())) - np.array(list(olds2.values())))))
    return diff

## TIME TO RUN SLIM!

gpm = read_gpm(args.gpmap)

diff = np.Inf
check = diff > args.cutoff
pathframe = pd.DataFrame()
pathframe['equilibrium'] = [False]

# first go
run = 0
sim = run_slim(gpm, args.working_dir, args.N, args.u)
if args.overwrite:
    sim.to_json(args.slim_out)
else:
    sim.to_json(args.slim_out+'_run_'+str(run))
edgeweights = get_edgeweights(sim)
pathframe = append_steps(pathframe, edgeweights)

while check:
    run += 1
    sim = run_slim(gpm, args.working_dir, args.N, args.u)
    if args.overwrite:
        sim.to_json(args.slim_out)
    else:
        sim.to_json(args.slim_out+'_run_'+str(run))
    edgeweights = get_edgeweights(sim)
    pathframe = append_steps(pathframe, edgeweights)
    diff = find_eq(pathframe)
    check = diff > args.cutoff
    pathframe.to_csv(args.outpath + '_paths.txt', sep='\t')
    if args.runs:
        if run > args.runs:
            break

if not args.runs:
    pathframe.loc[pathframe.index[-1], 'equilibrium'] = True
    pathframe.to_csv(args.outpath + '_paths.txt', sep='\t')
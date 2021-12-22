## generate a rough mt fuji landscape with L sites,
## write to csv

import gpmap, argparse
from gpmap import GenotypePhenotypeMap

parser=argparse.ArgumentParser()
parser.add_argument('--L_sites', type=int, help='number of sites')
parser.add_argument('--roughness', type=float, help='roughness of landscape')
parser.add_argument('--out', help='outpath')
args=parser.parse_args()

# make map
gpm = gpmap.simulate.generate_gpm(alphabet='01', num_sites=args.L_sites, num_states_per_site=2)

# generate fuji fitness landscape towards derived state
gpm.data.loc[:, 'phenotype'] = gpmap.simulate.fuji(gpm, ref_genotype = '1'*args.L_sites, roughness = args.roughness)
gpm.data.loc[0, 'phenotype'] = min(gpm.data.loc[:, 'phenotype'])
gpm.data.loc[len(gpm.data)-1, 'phenotype'] = max(gpm.data.loc[:, 'phenotype'])

# write csv
gpm.to_csv(args.out+'_gpmap.txt')

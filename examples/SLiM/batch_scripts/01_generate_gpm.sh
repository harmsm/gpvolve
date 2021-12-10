#!bin/bash
# generate 10 Mt. Fuji landscapes each
# with increasing roughness [0.1, 0.2, 0.3, 0.4, 0.5]
# for L sites [3, 4, 5, 6, 7, 8]

for L in {3..8}; do             # sites
    for r in 0.{1..5}; do       # roughness
        for n in {0..9}; do     # 10 maps
            python generate_gpm.py --L_sites $L --roughness $r --out gpmaps/$L\_site_fuji_r$r\_$n
        done
    done
done

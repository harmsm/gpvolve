#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <stdio.h>

#define PRECISION 1E-20
#define LOG2E 1.44269504088896340736 // log2(e)
#define EULER 2.718281828459045 // e

int is_zero(double value){
    // See if a value is within precision of zero.
    return fabs(value) < PRECISION;
}

int same_number(double value1, double value2){
    // See if value1 and value2 have the same value within precision.

    return fabs(value1 - value2) < PRECISION;
}

double log2(double value){

    return log(value)*LOG2E;

}

double sswm(double fitness_i, double fitness_j, long population_size){

    // Strong selection, weak mutation model. From Gillespie 1984.
    //
    // Parameters
    // ----------
    // fitness_i : float
    //     fitness of the source genotype (wildtype). Must be greater than zero.
    // fitness_j : float
    //     fitness of the target genotype (mutant). Must be greater than zero.
    //
    // Returns
    // -------
    // fixation_probability : float
    //
    // Notes
    // -----
    // + Fixation probability given by:
    //     $$s_{ij} = \frac{f_{j} - f_{i}}{f_{i}}$$
    //     $$\pi_{i \rightarrow j} = 1 - e^{s_{ij}}$$
    //
    // + Gives real valued answers for all finite inputs of fitness_i and fitness_j.
    //

    double a, ratio, sij;

    a = fitness_j - fitness_i;
    if (a <= 0) {
        return 0.0;

    } else {

        ratio = log2(a) - log2(fitness_i);

        if (ratio > DBL_MAX_EXP){
            return 1.0;
        }

        sij = pow(2,ratio);
        return 1 - exp(-sij);

    }

}

double mcclandish(double fitness_i, double fitness_j, long population_size){
    //
    // Calculate fixation probability using model proposed by McClandish, 2011.
    //
    // Parameters
    // ----------
    // fitness_i : float
    //     fitness of the source genotype (wildtype). Must be greater than zero.
    // fitness_j : float
    //     fitness of the target genotype (mutant). Must be greater than zero.
    // population_size : number
    //     population size (must be 1 or more.)
    //
    // Returns
    // -------
    // fixation_probability : float
    //
    // Notes
    // -----
    // + Fixation probability given by:
    //
    // $$\pi_{i \rightarrow j} = \frac{1-e^{-2(f_{j}-f_{i})}}{1-e^{-2N(f_{j}-f_{i})}}$$
    //
    // + Function gives real valued answers for all finite inputs of fitness_i,
    //   fitness_j, and population_size.
    //
    // References
    // ----------
    // McCandlish, D. M. (2011), VISUALIZING FITNESS LANDSCAPES. Evolution, 65: 1544-1558.
    //

    // Control variables
    int i;
    double param_sets[2][2] = {{0,0},{0,0}};
    double num_param_sets = 1;
    double results = 0;
    int bad_exp_neg2a = 0;
    int unable_to_calculate = 4;

    // Interesting constants
    double power_coeff = -2*LOG2E;
    double l2_power_coeff = log2(2*LOG2E);

    double fi, fj, a;
    double exp_neg2a, exp_neg2aN, neg2a, neg2aN;

    // If population size is one, will fix no matter what
    if (population_size == 1){
        return 1.0;
    }

    /* If fitness is identical, generate parameter set with fitness_i slightly
     * smaller and fitness_j slightly smaller. Since these are all ratios, this
     * will sample just above and just below the infinity of fi == fj. Do the
     * calculation for both params in param_sets and take the mean at the end. */
    if (same_number(fitness_i,fitness_j)){
        num_param_sets = 2;
        param_sets[0][0] = fitness_i*0.99999;
        param_sets[0][1] = fitness_j;
        param_sets[1][0] = fitness_i;
        param_sets[1][1] = fitness_j*0.99999;
    } else {
        num_param_sets = 1;
        param_sets[0][0] = fitness_i;
        param_sets[0][1] = fitness_j;
    }

    // Go through number of fitness parameter sets
    for (i = 0; i < num_param_sets; i++){

        // Get fitness i and j
        fi = param_sets[i][0];
        fj = param_sets[i][1];
        a = fj - fi;

        // Can we do -2*(fj - fi) and exp(-2*(fj - fi))?
        if ((log2(fabs(a)) + l2_power_coeff) <= DBL_MAX_EXP){

            neg2a = a*power_coeff;
            unable_to_calculate -= 1;

            if (neg2a <= DBL_MAX_EXP){
                exp_neg2a = pow(2,neg2a);
                unable_to_calculate -= 1;
            } else {
                bad_exp_neg2a = 1;
            }
        }

        // Can we do -2*(fj - fi)*N and exp(-2*(fj - fi)*N)?
        if ((log2(fabs(a)) + log2(population_size) + l2_power_coeff) <= DBL_MAX_EXP){

            neg2aN = a*population_size*power_coeff;
            unable_to_calculate -= 1;

            if (neg2aN <= DBL_MAX_EXP){
                exp_neg2aN = pow(2,neg2aN);
                unable_to_calculate -= 1;
            }
        }

        // Something was too big to calculate
        if (unable_to_calculate > 0){

            // If a is positive (and knowning N is always positive),
            // exp(-2*a*N) -> 0 and the denominator (1 - exp(-2*a*N)) --> 1.
            if (a > 0){

                // If exp(-2*a) is overflowing when a is positive, numerator --> 1.0.
                // Otherwise calcualte numerator exactly.
                if (bad_exp_neg2a){
                    results += 1.0;
                } else {
                    results += (1 - exp_neg2a);
                }

            } else {

                // If a is super negative, exp(-2a) becomes large and positive
                // and dominates the 1 in front. This means the equation becomes
                // exp(-2*a)/exp(-2*a*N) which simplifies to 1/exp(-2*a*(N-1)).

                // If we got an overflow in the exp(-2aN), the denominator is
                // huge and this --> 0.

                // If we have huge negative selection...
                //      With a huge population? --> 0

                //      With tiny population? Numerically, this can't
                //      happen unless population size is < 1.0. Assert no
                //      fixation because scenario is so outlandish. --> 0

                // If exp_neg2a is finite, the overflow had to come in the
                // denominator.  This means denominator is huge and value --> 0.

                // So, no matter what, result is zero.

                results += 0;

            }

        } else {
            results += (1 - exp_neg2a)/(1 - exp_neg2aN);
        }

    }

    return results/num_param_sets;

}

double moran(double fitness_i, double fitness_j, long population_size){

    // Calculate fixation probability using moran model proposed by Sella and
    // Hirsch, 2005.
    //
    // Parameters
    // ----------
    // fitness_i : float
    //     fitness of the source genotype (wildtype). Must be greater than zero.
    // fitness_j : float
    //     fitness of the target genotype (mutant). Must be greater than zero.
    // population_size : number
    //     population size (must be 1 or more.)
    //
    // Returns
    // -------
    // fixation_probability : float
    //
    // Notes
    // -----
    // + Fixation probability given by:
    //
    // $$\pi_{i \rightarrow j} = \frac{1 - \Big ( \frac{f_{i}}{f_{j}} \Big ) }{1 - \Big ( \frac{f_{i}}{f_{j}} \Big )^{N} }$$
    //
    // + Function gives real valued answers for all finite inputs of fitness_i,
    //   fitness_j, and population_size.
    //
    // References
    // ----------
    // G. Sella, A. E. Hirsh: The application of statistical physics to evolutionary biology, Proceedings of the National
    // Academy of Sciences Jul 2005, 102 (27) 9541-9546.
    //

    // Control variables
    int i;
    double param_sets[2][2] = {{0,0},{0,0}};
    double num_param_sets = 1;
    double results = 0;

    double fi, fj, a, b;

    // Will be 1.0 for population size of 1, regardless of fitness difference
    if (population_size == 1){
        return 1.0;
    }

    // If fitness is identical, generate parameter set with fitness_i slightly
    // smaller and fitness_j slightly smaller. Since these are all ratios, this
    // will sample just above and just below the infinity of fi == fj. If the
    // fitness are not same, just record them both as param sets. Do the
    // calculation for all params in param_sets and take the mean at the end.
    if (same_number(fitness_i,fitness_j)){
        num_param_sets = 2;
        param_sets[0][0] = fitness_i*0.99999;
        param_sets[0][1] = fitness_j;
        param_sets[1][0] = fitness_i;
        param_sets[1][1] = fitness_j*0.99999;
    } else {
        num_param_sets = 1;
        param_sets[0][0] = fitness_i;
        param_sets[0][1] = fitness_j;
    }


    for (i = 0; i < num_param_sets; i++){

        fi = param_sets[i][0];
        fj = param_sets[i][1];

        a = log2(fi) - log2(fj);

        // If |a|*N is *huge* we can't store their multiple.
        if ((log2(fabs(a)) + log2(population_size)) > DBL_MAX_EXP){

            // (1 - 2^a)/(1-2^(a*N)).
            // If 2^a is really big compared to 1, this reduces to:
            //      -2^a/-2^(a*N) --> 1/2^N.
            // If 2^N is really big compared to 2^a, this reduces to ...
            //       1/2^N.
            // If 2^N is bigger than we can calculate, just return 0.
            if (a > 0){
                if (population_size > DBL_MAX_EXP){
                    results += 0.0;
                } else {
                    results += (1/(pow(2,population_size)));
                }
                continue;

            // If a < 0...
            // We know that |a|*N is HUGE, so denominator (1-2^N*2^a --> 1.0).
            } else {

                if (-a > DBL_MAX_EXP){
                    results += 1.0;
                } else {
                    results += 1 - pow(2,a);
                }
                continue;
            }
        }

        b = population_size*a;

        // Too big to do calculation explicitly, but that's okay. We need to do
        // (1 - 2^a)/(1 - 2^b), but we now know that 2^a and 2^b are >> 1. So this
        // reduces to -2^a/-2^b, which simplifies to 2^(a-b). If b is big it must
        // be positive.  Since it is larger than a by a factor of N, b-a will
        // positive, it is larger than a by a factor of N, so this will lead to
        // a safe negative number going into power.

        if (b > DBL_MAX_EXP){
            results += pow(2,a-b);
        } else {
            results += (1 - pow(2,a))/(1 - pow(2,b));
        }
    }

    return results/num_param_sets;
}

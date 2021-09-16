#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <numpy/arrayobject.h>
#include <numpy/random/bitgen.h>
#include <numpy/random/distributions.h>

#define PRECISION 1E-20

int is_zero(double value){

    /* See if a value is within precision of zero. */

    return fabs(value) < PRECISION;

}

int random_choice(PyArrayObject *vector, int vector_length, bitgen_t *bitgen_state){

    /* Return a random element from numpy array. uses numpy random
     * number generator */

    int i;
    i = random_bounded_uint64(bitgen_state,0,vector_length,0,0);

    return *((int *)PyArray_GETPTR1(vector,i));

}

int random_weighted_choice(int *vector, double *weight, int vector_length, bitgen_t *bitgen_state){

    /* Return a random element from int array vector, weighting choice by
     * values in double vector weight. Uses numpy random number generator */

    int i;
    double cum_sum, rand_value;

    cum_sum = 0.0;
    rand_value = random_standard_uniform(bitgen_state);
    for (i = 0; i < vector_length; i++){
        cum_sum += weight[i];
        if (cum_sum >= rand_value){
            return vector[i];
        }
    }

    // Only get here if rand_value is numerically one and cum_sum is numerically
    // *slightly* less than one. Return last element in vector.
    return vector[i];

}

int wf_engine(int num_steps,
              int num_genotypes,
              int pop_size,
              int num_to_mutate,
              PyArrayObject *fitness,
              PyArrayObject *num_neighbors,
              PyObject *neighbors,
              PyArrayObject *pops){


    int i, j, k;
    int num_populated_genotypes;

    double *f_ptr;
    int *p_ptr;
    double denominator;

    // Initialize random number generator (grab current state, actually)
    bitgen_t *bitgen_state;

    // Allocate temporary arrays
    int *choice_vector;
    choice_vector = malloc(pop_size*sizeof(int));
    if (choice_vector == NULL){
        fprintf(stderr,"Could not allocate memory for choice_vector\n");
        return 1;
    }

    double *prob_vector;
    prob_vector = malloc(pop_size*sizeof(double));
    if (prob_vector == NULL){
        fprintf(stderr,"Could not allocate memory for prob_vector\n");
        return 1;
    }

    // Go for 1 -> num steps + 1 (0 is starting state defined before sim
    // starts)
    for (i = 1; i < num_steps + 1; i++){

        // Go through every genotype. For every populated genotype, calculate
        // a relative fitness value and store in prob_vector. record its index
        // in choice_vector.
        denominator = 0;
        k = 0;
        for (j = 0; j < num_genotypes; j++){

            // Get pointer to pops[i-1][j]
            p_ptr = (int *)PyArray_GETPTR2(pops,i-1,j);
            if ((*p_ptr) > 0){

                // Get pointer to fitness[j]
                f_ptr = (double *)PyArray_GETPTR1(fitness,j);

                choice_vector[k] = j;
                prob_vector[k] = (*f_ptr)*(*p_ptr);
                denominator += prob_vector[k];

                k += 1;
            }
        }

        // These are how many genotypes were populated
        num_populated_genotypes = k;

        // normalize prob_vector so cumulative sum of weights goes from 0 -> 1.
        // if denominator is zero (because all fitness were zero), calculate
        // prob_vector using only populations of genotypes, not their fitness
        // values
        if (is_zero(denominator)){
            for (j = 0; j < num_populated_genotypes; j++){
                p_ptr = (int *)PyArray_GETPTR2(pops,i-1,choice_vector[j]);
                prob_vector[j] = (*p_ptr)/pop_size;
            }
        } else {
            for (j = 0; j < num_populated_genotypes; j++){
                prob_vector[j] = prob_vector[j]/denominator;
            }
        }

        // Build a new population vector, selecting from previous generation
        // weighted by relative fitness (pop_size*fitness). For the first
        // num_to_mutate genotypes, mutate the genotype before storing.
        for (j = 0; j < pop_size; j++){
            k = random_weighted_choice(choice_vector,
                                       prob_vector,
                                       num_populated_genotypes,
                                       bitgen_state);

            // If this should be mutated, mutate it
            if (j < num_to_mutate) {

                k = random_choice((PyArrayObject *)PyList_GetItem(neighbors, k),
                                  *((int *)PyArray_GETPTR1(num_neighbors,k)),
                                  bitgen_state);
            }

            // Record a new occurence of genotype k in pops[i][k]
            p_ptr = (int *)PyArray_GETPTR2(pops,i,k);
            (*p_ptr) += 1;
        }
    }

    // Clean up.
    free(choice_vector);
    free(prob_vector);

    return 0;
}

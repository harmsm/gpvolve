#ifndef FIXATION_H
#define FIXATION_H

double moran(double fitness_i, double fitness_j, long population_size);
double mcclandish(double fitness_i, double fitness_j, long population_size);
double sswm(double fitness_i, double fitness_j, long population_size);

#endif /* FIXATION_H */

#ifndef WRIGHT_FISHER_ENGINE_H
#define WRIGHT_FISHER_ENGINE_H

struct {
    int **neighbors;
    int num_neighbors;
} neighbors_struct;

int wf_engine(int num_steps,
              int num_genotypes,
              int pop_size,
              int num_to_mutate,
              PyArrayObject *fitness,
              PyArrayObject *num_neighbors,
              PyObject *neighbors,
              PyArrayObject *pops);

#endif /* WRIGHT_FISHER_ENGINE_H */

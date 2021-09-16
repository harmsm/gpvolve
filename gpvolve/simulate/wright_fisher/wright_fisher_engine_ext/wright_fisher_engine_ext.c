#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "wright_fisher_engine.h"

// docstrings
static char module_docstring[] =
    "Run a Wright-Fisher style simulation on a genotype-phenotype map\n";

static char wright_fisher_engine_ext_simulate_docstring[] =
    "Run the Wright-Fisher simulation.\n";

// object prototypes
static PyObject *wright_fisher_engine_ext_simulate(PyObject *self, PyObject *args);

// method definitions
static PyMethodDef module_methods[] = {
    {"wf_engine",
     wright_fisher_engine_ext_simulate,
     METH_VARARGS,
     wright_fisher_engine_ext_simulate_docstring},
    {NULL, NULL}
};

// init function
PyMODINIT_FUNC PyInit_wright_fisher_engine_ext(void)
{
    PyObject *module;
    static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "wright_fisher_engine_ext",
        module_docstring,
        -1,
        module_methods,
        NULL,
        NULL,
        NULL,
        NULL
    };

    module = PyModule_Create(&moduledef);
    if (!module) return NULL;

    // Load numpy funcionality.
    import_array();

    return module;
}

static PyObject *wright_fisher_engine_ext_simulate(PyObject *self, PyObject *args)
{

    int num_steps, num_genotypes, pop_size, num_to_mutate;
    PyObject *neighbors;
    PyArrayObject *fitness, *num_neighbors, *pops;
    int return_value;

    // Parse input
    if (!PyArg_ParseTuple(args,
                          "iiiiOOOO",
                          &num_steps,
                          &num_genotypes,
                          &pop_size,
                          &num_to_mutate,
                          &fitness,
                          &num_neighbors,
                          &neighbors,
                          &pops)){
        return NULL;
    }

    // Run simulation proper
    return_value = wf_engine(num_steps,
                             num_genotypes,
                             pop_size,
                             num_to_mutate,
                             fitness,
                             num_neighbors,
                             neighbors,
                             pops);

    // Make sure it worked
    if (return_value != 0){

        if (return_value == 1){
            PyErr_SetString(PyExc_RuntimeError, "\nCould not allocate memory!\n");
            return NULL;
        }
    }

    // Return None
    Py_INCREF(Py_None);
    return Py_None;

}

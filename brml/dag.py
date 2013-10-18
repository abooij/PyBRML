import numpy
from .potential import Potential

def dag(pot_list):
    """ Generate DAG matrix from list of potentials. """
    all_vars = list(set([var for pot in pot_list for var in pot.variables]))
    all_vars.sort()
    N=len(all_vars)

    matrix = numpy.zeros((N,N))
    for pot in pot_list:
        posteriors = [var for var in pot.variables if var not in pot.priori_variables]
        prior_indices = [all_vars.index(var) for var in pot.priori_variables]
        for var in posteriors:
            index = all_vars.index(var)
            matrix[prior_indices,index]=1
    return matrix

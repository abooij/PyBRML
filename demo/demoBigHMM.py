#!/usr/bin/env python

"""
    DEMOSUMPROD Sum-Product algorithm test
"""
print(__doc__)

import numpy
import numpy.random as npr
import random
import networkx as nx
import matplotlib.pyplot as plt
from brml import TablePotential
from brml import FactorGraph
import profile

print("Libraries loaded")

# variables
H = 5 # random.randint(2,10) #number of H-states
A = npr.random([H,H]) #transition matrix for H-variables
variables = range(1000) #H variables
h_pots = [TablePotential([(i, H)], npr.random(H)) for i in variables] # measurements of hidden H state
h_trans = [TablePotential([(i, H), (i+1, H)], A) for i in variables[:-1]] #transition functions

#construct the factor graph
graph=FactorGraph(h_pots+h_trans)

#compute marginals
sumprod=graph.marginals()
print(sumprod)

#to profile:
#profile.run("graph.marginals()")

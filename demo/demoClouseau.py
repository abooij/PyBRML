#!/usr/bin/env python

# Python implementation of BRMLtoolbox
# Author: Jiuding Duan
# License: GNU license
# def demoClouseau():
"""
    DEMOCLOUSEAU inspector clouseau example
"""
print(__doc__)

import numpy
from brml import TablePotential
from brml import dag

butler=(2,2); maid=(1,2); knife=(0,2) # Variable order is arbitary (3,2,1 for MATLAB)
# Define states, starting from 0. (from 1 for MATLAB)
murderer=0; notmurderer=1
used=0; notused=1

pot = [None, None, None]

pot[butler[0]] = TablePotential([butler], numpy.array([0.6, 0.4]))

pot[maid[0]] = TablePotential([maid], numpy.array([0.2, 0.8]))

pot[knife[0]] = TablePotential([knife, butler, maid], \
    numpy.array([[[0.1, 0.6],
                  [0.2, 0.3]],
                 [[0.9, 0.4],
                  [0.8, 0.7]]]))
pot[knife[0]].priori_variables = [butler, maid]


posterior_pot = pot[knife[0]] * pot[butler[0]] * pot[maid[0]]

print("Joint distribution:")
print(posterior_pot) #joint distribution
DAG = dag(pot)
print("DAG matrix:")
print(DAG)

evidencedpot=posterior_pot.evaluate([(knife, used)])
print("Using evidence that the knife was used (ie. p(butler, maid|knife=used):")
print(evidencedpot)

print("Pot for butler after evidence (ie. p(butler|knife=used)):")
print(evidencedpot.marginalize([maid]))

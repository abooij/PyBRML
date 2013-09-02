#!/usr/bin/env python

"""
%SETPOT sets array potential variables to specified states
% newpot = setpot(pot,variables,evidstates)
%
% set variables in potential to evidential states in evidstates
% Note that the new potential does not contain the evidential variables
"""
import numpy as np
import copy as copy
from brml.potential import Potential
from brml.ismember import ismember
from brml.setstate import setstate
from brml.intersect import intersect
from brml.setstate import setstate
from brml.setminus import setminus
from brml.myzeros import myzeros
from brml.multpots import multpots
from brml.index_to_assignment import index_to_assignment
from brml.assignment_to_index import assignment_to_index


def setpot(pot, evvariables, evidstates):
    #FIXME: data format needed to be unified
    vars = pot.variables
    #vars = np.array(pot.variables) # convert to ndarray format
    #evariables = np.array(evvariables)
    # convert to ndarray format
    #evidstates = np.array(evidstates) # convert to ndarray format
    #print("variables:", vars)
    table = pot.table
    nstates = pot.card
    #print("number of states:", nstates)
    #print("vars:", vars)
    #print("evvariables:", evvariables)
    intersection, iv, iev = intersect(vars, evvariables)
    #iv = np.array(iv)
    #iev = np.array(iev)
    #print("intersection:", intersection)
    #print("iv:", iv)
    #print("iev:", iev)
    #print("iv type:", type(iv))
    #print("number of intersection:", intersection.size)
    if intersection.size == 0:
        newpot = copy.copy(pot)
    else:
        newvar = setminus(vars, intersection)
        dummy, idx = ismember(newvar, vars)
        newns = nstates[idx]
        newpot = Potential()
        newpot.variables = newvar
        newpot.card = newns
        newpot.table = np.zeros(newns)
        #print("idx:", idx)
        #print("iv:", iv)
        for i in range(np.prod(newns)):
            newassign = index_to_assignment(i, newns)
            oldassign = np.zeros(nstates.size, 'int8')
            oldassign[idx] = newassign
            oldassign[iv] = evidstates
            #print("newpot.table.shape:", newpot.table.shape)
            #print("newassign:", newassign)
            #print("newassign type:", type(newassign))
            newpot.table[tuple(newassign)] = pot.table[tuple(oldassign)]

    return newpot

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
from brml import dag
from brml import FactorGraph

# variables
a,b,c,d,e = zip(range(5), [random.randint(2,4) for _ in range(5)])
vars=[a,b,c,d,e]

pots = [ #define some potentials
      TablePotential([a,b], npr.random([a[1],b[1]]))
    , TablePotential([b,c,d], npr.random([b[1],c[1],d[1]]))
    , TablePotential([c], npr.random([c[1]]))
    , TablePotential([d,e], npr.random([d[1],e[1]]))
    , TablePotential([d], npr.random([d[1]]))
    ]
for p in pots: print(p)

graph=FactorGraph(pots)
graph.pretty_draw()
plt.show()

sumprod=[None]*5
for i in range(5):
    sumprod[i]=graph.sumprod(vars[i])

import operator, functools
jointpot=functools.reduce(operator.mul, pots)

margpot=[None]*5
for i in range(5):
    # compute p(vars[i]), ie marginalize away all other variables
    newvars=list(vars)
    newvars.pop(i)
    margpot[i]=jointpot.marginalize(newvars)
    print("Marginal of variable ",vars[i],":")
    print(margpot[i].normalize())
    # sumprod[i] is p(vars[i]) as computed by the sum-product alg.
    print("According to sum-prod alg:")
    print(sumprod[i].normalize())

# FG on reduced variables
# TODO
"""
% FG on reduced variables (as a demonstation of changing variables):
disp('compute p(c|a=1) by FG on reduced variables (column 1) and raw summation (column 2):')
[pot2 newvars oldvars]=squeezepots(setpot(pot,a,1));
marg2=sumprodFG(pot2,FactorGraph(pot2));
str1=disptable(condpot(marg2{newvars(oldvars==c)}),[]);
str2=disptable(condpot(multpots(setpot(pot,a,1)),c),[]);
disp([char(str1) char(str2)])

fprintf(1,'\n\nCompute the marginals for the variables connected to each potential.\n')
% compute marginals for the factors:
for f=1:length(pot)
    potmarg{f}=condpot(multpots([pot(f) mess(mess2fact(f,A))])); % marginal is the potential multiplied by incoming messages
    potmargcheck{f}=condpot(sumpot(jointpot,pot{f}.variables,0));
    str1=disptable(potmarg{f});
    str2=disptable(potmargcheck{f});
    fprintf(1,'marginal table on potential %d (left sumprod, right check)\n',f); disp([char(str1) char(str2)])
end
"""

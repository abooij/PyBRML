#!/usr/bin/env python

# this is a test case for sample codes

from brml.potential import Potential
from brml.variable import Variable


p = Potential(1,1)
print("var POTENTIAL.p created")
print("p.variable = ", p.variables)
print("p.table = ", p.table)

v = Variable('butler',['hehe', 'heihei'])
# v = variable('butler',['murderer','not murderer'])
print("var VARIABLE.v created")
print("v.name = ", v.name)
print("v.domain = ", v.domain)

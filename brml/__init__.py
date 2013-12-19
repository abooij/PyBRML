"""
PyBRML is a Python version of BRML toolbox for Bayesian Reasoning and Machine Learning

Thanks to Dr. David Barber's book Bayesian Reasoning and Machine Learning and his original design of the toolbox as an accompanying code for the book.

The BRMLtoolbox is provided to help readers see how mathematical models translate into actual MATLAB code. There are a large number of demos that a lecturer may wish to use or adapt to help illustrate the material. In addition many of the exercises make use of the code, helping the reader gain confidence in the concepts and their application. Along with complete routines for many Machine Learning methods, the philosophy is to provide low level routines whose composition intuitively follows the mathematical description of the algorithm. In this way students may easily match the mathematics with the corresponding algorithmic implementation.

.. note:: Most methods are functional in the sense that class methods rarely change objects. Rather, they return new objects with the requested modifications.
"""

from .potential import Potential, TablePotential
from .dag import dag
from .fg import FactorGraph
